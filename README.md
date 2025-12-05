# Gemini Documentation Tutor

A lightweight, NotebookLM-style assistant that lets you build a local knowledge index from documentation and answer questions using the Gemini API.

## Features
- Loads Markdown or text files from any folder.
- Splits content into overlapping chunks for better retrieval quality.
- Builds and saves an embedding index using the `text-embedding-004` model.
- Answers questions with `gemini-1.5-flash` while citing the most relevant sources.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide your Gemini API key in a `.env` file or environment variable:
   ```bash
   echo "GEMINI_API_KEY=your-key-here" > .env
   ```

## Usage
Build an index from your documentation:
```bash
python main.py build-index sample_docs data/index.json
```

Ask a question against the index:
```bash
python main.py ask data/index.json "How does the workflow operate?"
```

### Customizing
- Change `sample_docs` to any documentation folder.
- Adjust `--chunk-size` and `--overlap` to tune chunking for your content.
- Regenerate the index whenever the source docs change.

## Project Structure
- `gemini_agent/` – core code for loading documents, embedding, and generation.
- `sample_docs/` – example documentation to try the pipeline.
- `main.py` – CLI for building the index and asking questions.

## Notes
- The agent uses a simple in-memory vector search with cosine similarity. For larger corpora, consider swapping in a dedicated vector database.
- Ensure outbound internet access is available for Gemini API calls.
