# 🌟 PyPlexitas

PyPlexitas is a Python script that is designed to create an open-source alternative to Perplexity AI, a tool that provides users with detailed answers to their queries by searching the web, extracting relevant content, and using advanced language models to generate responses.

The script operates by first taking a user’s query and using search engines like Bing or Google and GMAIL to find relevant web pages or e-mails. It then scrapes the content from these web pages or mails, processes the text into manageable chunks, and generates vector embeddings for these chunks. Vector embeddings are mathematical representations of text that allow for efficient searching and comparison of content. These embeddings are stored in a database, enabling quick retrieval of relevant information based on the user’s query.

Once the content is processed and stored, the script uses a language model to generate a detailed answer to the user’s query, using the information extracted from the web pages. This response is designed to be accurate and informative, drawing directly from the content found during the search process.

**Example:**
```bash
python PyPlexitas.py -q "When will the next model GPT-5 be released" -s 10 --engine google 
```

Expected Output:
```
Searching for 🔎: When will the next model GPT-5 be released using google
Starting Google search ⏳
Google search returned 🔗: 10 results
From domains 🌐: mashable.com  www.reddit.com  www.tomsguide.com  www.datacamp.com  medium.com  www.standard.co.uk  www.theverge.com  arstechnica.com  
Scraping content from search results...
Embedding content ✨
Total embeddings 📊: 10
Total chunks processed 🧩: 7

Answering your query: When will the next model GPT-5 be released 🙋

The release date for GPT-5 is currently expected to be sometime in mid-2024, likely during the summer, according to a report from Business Insider [1][2]. OpenAI representatives have not provided a specific release date, and the timeline may be subject to change depending on the duration of safety testing and other factors [1][2]. OpenAI CEO Sam Altman has indicated that a major AI model will be released this year, but it is unclear whether it will be called GPT-5 or something else [1].

### Sources
1. Benj Edwards - https://arstechnica.com/information-technology/2024/03/gpt-5-might-arrive-this-summer-as-a-materially-better-update-to-chatgpt/
2. Saqib Shah - https://www.standard.co.uk/tech/openai-chatgpt-5-release-date-b1076129.html
```

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Getting API Keys](#getting-api-keys)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Web Search**: Perform web searches using Bing, Google, or Gmail APIs.
- **Content Scraping**: Scrape content from search results.
- **Embedding Generation**: Generate embeddings for content using OpenAI or Ollama models.
- **Question Answering**: Answer questions based on the scraped content.
- **Vector Database**: Use Qdrant for storing and querying embeddings.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/dkruyt/PyPlexitas.git
    cd PyPlexitas
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the Qdrant service using Docker:
    ```bash
    docker-compose up -d
    ```

## Configuration
Configure your environment variables by creating a `.env` file in the project root. Use the provided `example.env` as a template:
```bash
cp example.env .env
```
Fill in your API keys and other necessary details in the `.env` file:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_CX`
- `BING_SUBSCRIPTION_KEY`

## Getting API Keys

### OpenAI API Key
1. Sign up or log in to your [OpenAI account](https://www.openai.com/).
2. Go to the API section and generate a new API key.
3. Copy the API key and add it to the `OPENAI_API_KEY` field in your `.env` file.

### Google Custom Search API Key and CX
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing project.
3. Enable the Custom Search API in the API & Services library.
4. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page and create an API key.
5. Copy the API key and add it to the `GOOGLE_API_KEY` field in your `.env` file.
6. To get the Custom Search Engine (CX) ID, go to the [Custom Search Engine](https://cse.google.com/cse/) page.
7. Create a new search engine or select an existing one.
8. Copy the Search Engine ID (CX) and add it to the `GOOGLE_CX` field in your `.env` file.

### Bing Search API Key
1. Sign up or log in to your [Microsoft Azure account](https://azure.microsoft.com/).
2. Create a new Azure resource for Bing Search v7.
3. Go to the Keys and Endpoint section to find your API key.
4. Copy the API key and add it to the `BING_SUBSCRIPTION_KEY` field in your `.env` file.

### Gmail API Key
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing project.
3. Enable the Gmail API in the API & Services library.
4. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page and create OAuth 2.0 credentials.
5. Download the credentials file and save it as `credentials.json` in the project root.

## Usage
Run the PyPlexitas script with your query:
```bash
python PyPlexitas.py -q "Your search query" -s 10 --engine bing
```
Options:
- `-q, --query`: Search query (required)
- `-s, --search`: Number of search results to parse (default: 10)
- `--engine`: Search engine to use (`bing`, `google`, or `gmail`, default: `bing`)
- `-l, --log-level`: Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `ERROR`)
- `-t, --max-tokens`: Maximum token limit for model input (default: 1024)
- `--quiet`: Suppress print messages

## Project Structure
- `PyPlexitas.py`: Main script for running the application.
- `example.env`: Example configuration file for environment variables.
- `docker-compose.yml`: Docker Compose configuration for Qdrant.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the GPL 3.0 License. See the [LICENSE](LICENSE) file for details.
