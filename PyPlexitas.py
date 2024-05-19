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
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv()

# Constants
# Default endpoint for Bing Search API
DEFAULT_BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
# Default base URL for OpenAI API
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Default chunk size for processing text data
CHUNK_SIZE = 1000
# Dimension of the embedding vectors
DIMENSION = 1536
# Default maximum number of tokens for input/output in API requests
DEFAULT_MAX_TOKENS = 1024  

# Initialize logger
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

    def add_webpage_content(self, url_hash: str, content: str):
        if (search := self.search_map.get(url_hash, None)) is not None:
            search.content = content
            logger.debug(f"Added content for URL hash {url_hash}")

    def add_id_to_chunk(self, chunk: str, search_result_id: str, chunk_id: int):
        self.chunk_id_chunk_map[chunk_id] = chunk
        self.chunk_id_to_search_id[chunk_id] = search_result_id
        logger.debug(f"Chunk added for search result ID {search_result_id} with chunk ID {chunk_id}")

    def get_chunks(self, ids: List[int]) -> List["Chunk"]:
        chunks = []
        for chunk_id in ids:
            chunk_content = self.chunk_id_chunk_map.get(chunk_id)
            search_id = self.chunk_id_to_search_id.get(chunk_id)
            search_result = self.search_map.get(search_id)
            if search_result and chunk_content:
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

# Function to hash strings, typically used for URL hashing or similar purposes.
def hash_string(input_string: str) -> str:
    return hashlib.sha256(input_string.encode()).hexdigest()

# Function to clean text by removing extra whitespace and normalizing spaces.
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)

# Function to perform web search using Bing's search API.
async def fetch_web_pages(request: Request, search_count: int):
    logger.info("Starting Bing search...")
    
    query = request.query
    mkt = "en-US"
    count_str = str(search_count)

    logger.debug(f"Search parameters - Query: {query}, Market: {mkt}, Count: {count_str}")

    params = {
        "mkt": mkt,
        "q": query,
        "count": count_str,
    }

    bing_endpoint = os.getenv("BING_ENDPOINT", DEFAULT_BING_ENDPOINT)
    bing_api_key = os.getenv("BING_SUBSCRIPTION_KEY")
    if not bing_api_key:
        logger.error("Bing API key is missing. Please set BING_SUBSCRIPTION_KEY in your environment variables.")
        return

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
                logger.error(f"Request failed with status code: {response.status}")
                raise Exception(f"Request failed with status code: {response.status}")

# Function to scrape content from given URLs by making HTTP requests.
async def fetch_url_content(session: ClientSession, url: str, max_retries: int = 3, retry_delay: int = 1) -> str:
    logger.info(f"Scraping content from URL: {url}")
    retries = 0
    while retries < max_retries:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    full_text = await response.text()
                    document = html.fromstring(full_text)

                    selectors = [
                        "article",  # good for blog posts/articles
                        "div.main-content",  # a more specific div that usually holds main content
                        "body",  # generic selector
                    ]

                    main_text = ""
                    for selector in selectors:
                        elements = document.cssselect(selector)
                        if elements:
                            main_text = "\n".join([clean_text(element.text_content()) for element in elements if element.text_content()])
                            break

                    logger.debug(f"Extracted content: '{main_text}' from URL: {url}")
                    if not main_text.strip():
                        logger.warning(f"No content extracted from URL: {url}")
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

# Function to handle the URL processing by scraping content from each URL found in the search results.
async def process_urls(request: Request):
    logger.info("Processing URLs to scrape content...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for search_result in request.search_map.values():
            url = search_result.url
            task = asyncio.create_task(fetch_url_content(session, url))
            tasks.append(task)

        webpage_contents = await asyncio.gather(*tasks)
        
        for search_result, content in zip(request.search_map.values(), webpage_contents):
            url_hash = hash_string(search_result.url)
            logger.debug(f"Adding webpage content for URL hash {url_hash}. Content length: {len(content)}")
            request.add_webpage_content(url_hash, content)

# Embedding generation and upsert
async def insert_embedding(vector_client: QdrantClient, embedding: List[float], chunk_id: int):
    logger.debug(f"Inserting embedding for chunk ID {chunk_id}")
    vector_client.upsert(
        collection_name="embeddings",
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=embedding,
            )
        ],
    )
    
# Function to generate embeddings for web page content and upsert them into the vector database.
# This function iterates over the search results, chunks the content, and processes each chunk to create and store embeddings.
async def generate_upsert_embeddings(request: Request, vector_client: QdrantClient):
    logger.info("Generating and upserting embeddings...")
    tasks = []
    shared_counter = 0

    for url_hash, search_result in request.search_map.items():
        content = search_result.content or ""
        logger.debug(f"Content length for URL hash {url_hash}: {len(content)}")

        if not content.strip():
            logger.warning(f"Skipping URL {search_result.url} due to empty or non-relevant content")
            continue

        # Split content into chunks
        chunks = []
        words = content.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            chunks.append(chunk)

        if not chunks:
            chunks = [content]

        logger.info(f"Chunked content into {len(chunks)} chunks for url: {search_result.url}")
        logger.info(f"Generating embedding for url: {search_result.url}")

        for chunk in chunks:
            task = asyncio.create_task(process_chunk(request, vector_client, shared_counter, url_hash, chunk))
            tasks.append(task)
            shared_counter += 1

    await asyncio.gather(*tasks)

# Function to process a content chunk, generate its embedding, and upsert the embedding into the vector database.
async def process_chunk(request: Request, vector_client: QdrantClient, shared_counter: int, url_hash: str, chunk: str):
    logger.debug(f"Processing chunk with ID {shared_counter} for URL hash {url_hash}")
    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        embeddings = OllamaEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()
    embedding = embeddings.embed_query(chunk)

    chunk_id = shared_counter

    request.add_id_to_chunk(chunk, url_hash, chunk_id)
    logger.debug(f"Processed and added chunk with ID {chunk_id} for URL hash {url_hash}")

    await insert_embedding(vector_client, embedding, chunk_id)

# LLM agent
class LLMAgent:
    def __init__(self):
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "llama3")
        chat_model_name = os.getenv("CHAT_MODEL_NAME", "gpt-4o")
        ollama_chat_model_name = os.getenv("OLLAMA_CHAT_MODEL_NAME", "llama3")

        self.local_mode = use_ollama
        if self.local_mode:
            self.embeddings = OllamaEmbeddings(model='llama3')
            self.llm = Ollama(model=ollama_chat_model_name)
        else:
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=chat_model_name)
        
        logger.info(f"LLM Agent initialized. Local mode: {self.local_mode}")

    def chunk_to_documents(self, chunks: List[Chunk], max_tokens: int) -> List[Document]:
        documents = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(chunk.content.split())  # Simple token estimation using word count
            if current_tokens + chunk_tokens > max_tokens:
                break
            documents.append(Document(page_content=chunk.content, metadata={"name": chunk.name, "url": chunk.url}))
            current_tokens += chunk_tokens
        
        logger.debug(f"Converted {len(chunks)} chunks into {len(documents)} documents")
        return documents

    async def answer_question_stream(self, query: str, chunks: List[Chunk], max_tokens: int):
        logger.info(f"\nAnswering your query: {query} 🙋\n")
        documents = self.chunk_to_documents(chunks, max_tokens)
        logger.debug(f"Documents metadata:\n{[doc.metadata for doc in documents]}")
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
        result = chain.invoke({"input_documents": documents, "question": query}, return_only_outputs=True)
        logger.debug(f"Generated answer: {result['output_text']}")
        print(result["output_text"])

async def main():
    parser = argparse.ArgumentParser(description="PyPlexitas - Open source CLI alternative to Perplexity AI.")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search Query")
    parser.add_argument("-s", "--search", type=int, default=10, help="Number of search results to parse")
    parser.add_argument("-l", "--log-level", type=str, default="ERROR", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("-t", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum token limit for model input")
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())
    logging.basicConfig(level=args.log_level.upper())
    
    logger.info("Application started")
    logger.debug(f"Received arguments: {args}")

    query = args.query
    search_count = args.search
    max_tokens = args.max_tokens

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

    collection_name = "embeddings"
    if vector_client.collection_exists(collection_name):
        vector_client.delete_collection(collection_name)

    vector_client.create_collection(
        collection_name=collection_name,
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
    await llm_agent.answer_question_stream(query, chunks, max_tokens)

if __name__ == "__main__":
    asyncio.run(main())
