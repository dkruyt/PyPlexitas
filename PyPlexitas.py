import os
import re
import json
import hashlib
import logging
import argparse
import asyncio
from aiohttp import ClientSession, ClientError, ClientSSLError
from typing import List, Dict, Optional

import aiohttp
from lxml import html
from langchain import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Constants
DEFAULT_BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
CHUNK_SIZE = 1000
DIMENSION = 1536

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request class
class Request:
    def __init__(self, query: str):
        self.query = query
        self.search_map: Dict[str, SearchResult] = {}
        self.chunk_id_chunk_map: Dict[int, str] = {}
        self.chunk_id_to_search_id: Dict[int, str] = {}

    def add_search_result(self, search_result: "SearchResult"):
        url_hash = hash_string(search_result.url)
        self.search_map[url_hash] = search_result

    def add_webpage_content(self, url: str, content: str):
        url_hash = hash_string(url)
        if url_hash in self.search_map:
            self.search_map[url_hash].content = content

    def add_id_to_chunk(self, chunk: str, search_result_id: str, chunk_id: int):
        self.chunk_id_chunk_map[chunk_id] = chunk
        self.chunk_id_to_search_id[chunk_id] = search_result_id

    def get_chunks(self, ids: List[int]) -> List["Chunk"]:
        chunks = []
        for chunk_id in ids:
            chunk_content = self.chunk_id_chunk_map[chunk_id]
            search_id = self.chunk_id_to_search_id[chunk_id]
            search_result = self.search_map[search_id]
            if search_result:
                chunk = Chunk(
                    content=chunk_content,
                    name=search_result.name,
                    url=search_result.url,
                )
                chunks.append(chunk)
        return chunks

# SearchResult class
class SearchResult:
    def __init__(self, name: str, url: str, content: Optional[str] = None):
        self.name = name
        self.url = url
        self.content = content

# Chunk class
class Chunk:
    def __init__(self, content: str, name: str, url: str):
        self.content = content
        self.name = name
        self.url = url

# Utility functions
def hash_string(input_string: str) -> str:
    return hashlib.sha256(input_string.encode()).hexdigest()

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)

# Bing search
async def fetch_web_pages(request: Request, search_count: int):
    query = request.query
    mkt = "en-US"
    count_str = str(search_count)

    params = {
        "mkt": mkt,
        "q": query,
        "count": count_str,
    }

    bing_endpoint = os.getenv("BING_ENDPOINT", DEFAULT_BING_ENDPOINT)
    bing_api_key = os.environ["BING_SUBSCRIPTION_KEY"]

    headers = {
        "Ocp-Apim-Subscription-Key": bing_api_key,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(bing_endpoint, params=params, headers=headers) as response:
            if response.status == 200:
                json_data = await response.json()
                logger.info(f"Bing search returned: {len(json_data['webPages']['value'])} results")

                for wp in json_data["webPages"]["value"]:
                    search_result = SearchResult(
                        name=wp["name"],
                        url=wp["url"],
                    )
                    request.add_search_result(search_result)

                logger.debug(f"JSON result from Bing: {json.dumps(json_data, indent=2)}")
            else:
                raise Exception(f"Request failed with status code: {response.status}")

# Content scraping
async def fetch_url_content(session: ClientSession, url: str, max_retries: int = 3, retry_delay: int = 1) -> str:
    logger.info(f"Scraping content from URL: {url}")
    retries = 0
    while retries < max_retries:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    full_text = await response.text()
                    document = html.fromstring(full_text)
                    selector_p = "body p, body h1, body h2, body h3, article p, div p, span p"
                    elements = document.cssselect(selector_p)
                    main_text = "\n".join([clean_text(element.text_content()) for element in elements if element.text_content()])
                    logger.debug(f"Extracted content: {main_text}")
                    return main_text
                else:
                    logger.warning(f"Request failed with status code: {response.status}")
                    return ""
        except (ClientError, ClientSSLError) as e:
            logger.warning(f"Error occurred while scraping URL: {url}. Error: {str(e)}")
            retries += 1
            await asyncio.sleep(retry_delay)
    logger.error(f"Failed to scrape content from URL: {url} after {max_retries} retries.")
    return ""

async def process_urls(request: Request):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for search_result in request.search_map.values():
            url = search_result.url
            task = asyncio.create_task(fetch_url_content(session, url))
            tasks.append(task)

        webpage_contents = await asyncio.gather(*tasks)

        for url, content in zip(request.search_map.keys(), webpage_contents):
            request.add_webpage_content(url, content)

# Embedding generation and upsert
async def insert_embedding(vector_client: QdrantClient, embedding: List[float], chunk_id: int):
    vector_client.upsert(
        collection_name="embeddings",
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=embedding,
            )
        ],
    )

async def generate_upsert_embeddings(request: Request, vector_client: QdrantClient):
    tasks = []
    shared_counter = 0

    for url_hash, search_result in request.search_map.items():
        content = search_result.content or ""
        chunks = [" ".join(content.split()[i:i+CHUNK_SIZE]) for i in range(0, len(content.split()), CHUNK_SIZE)]
        logger.info(f"Chunked content into {len(chunks)} chunks for url: {search_result.url}")
        logger.info(f"Generating embedding for url: {search_result.url}")

        for chunk in chunks:
            task = asyncio.create_task(process_chunk(request, vector_client, shared_counter, url_hash, chunk))
            tasks.append(task)
            shared_counter += 1

    await asyncio.gather(*tasks)

async def process_chunk(request: Request, vector_client: QdrantClient, shared_counter: int, url_hash: str, chunk: str):
    embeddings = OpenAIEmbeddings()
    embedding = embeddings.embed_query(chunk)

    chunk_id = shared_counter

    request.add_id_to_chunk(chunk, url_hash, chunk_id)

    await insert_embedding(vector_client, embedding, chunk_id)

# LLM agent
class LLMAgent:
    def __init__(self):
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        api_key = os.environ["OPENAI_API_KEY"]

        self.local_mode = "localhost" in base_url
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(openai_api_key=api_key)

    def chunk_to_documents(self, chunks: List[Chunk]) -> List[str]:
        documents = []
        for chunk_id, chunk in enumerate(chunks, start=1):
            chunk_yaml = f"Name: {chunk.name}\nurl: {chunk.url}\nfact: {chunk.content}\nid: {chunk_id}\n\n"
            documents.append(chunk_yaml)
        return documents

    async def answer_question_stream(self, query: str, chunks: List[Chunk]):
        logger.info(f"\nAnswering your query: {query} 🙋\n")
        documents = self.chunk_to_documents(chunks)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            CONTEXT:
            {context}

            QUESTION:
            {question}

            INSTRUCTIONS:
            You are a helpful AI assistant that helps users answer questions using the provided context. If the answer is not in the context, say you don't know rather than making up an answer.

            Please provide a detailed answer to the question above only using the context provided.
            Include in-text citations like this [1] for each significant fact or statement at the end of the sentence.
            At the end of your response, list all sources in a citation section with the format: [citation number] Name - URL.
            """,
        )
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        
        result = chain({"input_documents": documents, "question": query}, return_only_outputs=True)
        print(result["output_text"])

async def main():
    parser = argparse.ArgumentParser(description="fyin.app - Open source CLI alternative to Perplexity AI.")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search Query")
    parser.add_argument("-s", "--search", type=int, default=10, help="Number of search results to parse")
    args = parser.parse_args()

    query = args.query
    search_count = args.search

    logger.info(f"Searching for: {query}")
    request = Request(query)
    llm_agent = LLMAgent()

    # Fetch search results
    logger.info("Fetching search results from Bing...")
    await fetch_web_pages(request, search_count)

    # Scrape content
    logger.info("Scraping content from search results...")
    await process_urls(request)

    # Generate and upsert embeddings
    logger.info("Embedding content...")
    dimension = len(llm_agent.embeddings.embed_query(query))
    vector_client = QdrantClient(host="localhost", port=6333)
    vector_client.recreate_collection(
        collection_name="embeddings",
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    await generate_upsert_embeddings(request, vector_client)

    # Search across embeddings
    prompt_embedding = llm_agent.embeddings.embed_query(query)
    search_result = vector_client.search(
        collection_name="embeddings",
        query_vector=prompt_embedding,
        limit=10,
    )
    chunk_ids = [result.id for result in search_result]
    chunks = request.get_chunks(chunk_ids)

    # Answer the question
    await llm_agent.answer_question_stream(query, chunks)

if __name__ == "__main__":
    asyncio.run(main())