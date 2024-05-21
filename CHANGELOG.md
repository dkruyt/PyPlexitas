# Changelog

## [Unreleased]

## [v0.2.0] - 2024-05-22

### Added
- Optional LLM query parameter `--llm-query`.
- New TODO.md file outlining tasks related to speed optimizations, configurable parameters review, and Gmail integration improvements.
- Integration of Gmail search functionality into the main function, allowing users to search and fetch emails from Gmail using specified query and count, including email content retrieval.

### Changed
- Updated `README.md` to reflect the new optional LLM query parameter and increased default max-tokens limit.
- Updated default constants for chunk size and max tokens.
- Enhanced `fetch_url_content` function with a timeout parameter.

### Fixed
- Use correct query variable for embedding calculation in PyPlexitas.
- Added missing Google libraries to `requirements.txt`.

### Removed
- Removed token.json and credentials.json from version control by adding them to `.gitignore`.

## [v0.1.0] - 2024-05-20

### Added
- Initial project files and configurations.
- Docker Compose configuration for Qdrant container including services, configs, volumes, and networks.
- Functionality to load environment variables from `.env` file and handle missing Bing API key gracefully.
- Embedding selection logic based on the value of `USE_OLLAMA`.
- Logging and debug statements throughout the codebase.
- Google search functionality with the necessary endpoint and parameters.

### Changed
- Refactored imports in `PyPlexitas.py` to include both OpenAI and OpenAIEmbeddings from `langchain_openai`.
- Updated logging configuration to set the logging level to DEBUG for more detailed logging information.
- Enhanced LLMAgent class by adding an optional debug parameter to print document content for debugging purposes.
- Adjusted the argparse description to reflect the project name PyPlexitas.
- Implemented collection handling in `vector_client`, including deleting the collection if it already exists before recreating it with the appropriate configurations.

### Fixed
- Corrected API key and model names in environment configuration.
- Set global User-Agent for all API requests.
- Added missing import statement for `hashlib` in `hash_string` function.

### Documentation
- Added `README.md` for PyPlexitas including features, installation instructions, configuration details, and usage guidelines.
- Updated `README.md` formatting and added license information.

### Chore
- Improved logging and print messages for search operations, including emoji indicators.
