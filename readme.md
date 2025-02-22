# Venice AI Document QA System

A question-answering system that uses Venice AI's LLM capabilities combined with document embeddings to provide accurate answers from a repository of documents. This implementation specifically targets the Venice AI API documentation repository.

## Features

- Document loading from Git repositories
- Local or OpenAI embeddings support
- Vector storage using FAISS
- Interactive Q&A interface
- Configurable model parameters through environment variables

## Prerequisites

- Python 3.11+
- Git installed on your system
- Venice AI API key
- (Optional) Hugging Face API key for alternative models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/veniceai/venice-api-docs.git
cd venice-api-docs
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

4. Update the `.env` file with your credentials:

```
env
VENICE_MODEL_NAME=llama-3.3-70b
VENICE_TEMPERATURE=0.5
VENICE_BASE_URL=https://api.venice.ai/api/v1/
VENICE_API_KEY=your_Venice_api_key_here
```

## Configuration

The system can be configured through environment variables:

- `VENICE_MODEL_NAME`: The Venice AI model to use (default: llama-3.3-70b)
- `VENICE_TEMPERATURE`: Temperature setting for response generation (default: 0.5)
- `VENICE_BASE_URL`: Venice AI API base URL
- `VENICE_API_KEY`: Your Venice AI API key
- `EMBED_HF_MODEL`: Hugging Face model for embeddings (default: BAAI/bge-small-en-v1.5)

## Usage

Run the main script to start the interactive QA system:

```bash
python3 main.py
```


The system will:
1. Clone the Venice AI API documentation repository (if not present)
2. Process and embed the documentation
3. Start an interactive Q&A session

Type your questions and press Enter. Type `/exit` to quit the program.

## Project Structure

- `src/main.py`: Main application entry point and QA system setup
- `src/llm_setup.py`: Venice AI LLM client configuration
- `src/embeddings.py`: Embeddings management for document processing
- `.env.example`: Environment variable template

## Error Handling

The system includes comprehensive error handling for:
- API connection issues
- Document processing errors
- Query execution failures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
