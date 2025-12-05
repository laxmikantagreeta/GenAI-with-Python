# Gemini Documentation Tutor

This example project demonstrates how to build a lightweight retrieval-augmented agent using the Gemini API.

## Workflow
1. Load markdown or text documentation from the `sample_docs` folder.
2. Chunk the text with a small overlap so context stays coherent.
3. Embed each chunk with the `text-embedding-004` model.
4. Store embeddings alongside metadata in a JSON file.
5. At query time, embed the question, retrieve the most similar chunks, and ask `gemini-1.5-flash` to craft a helpful answer.

## Configuration
Set the environment variable `GEMINI_API_KEY` in a `.env` file to authenticate requests to the Gemini API.

## Usage Tips
- Keep chunks under 1000 characters for best retrieval quality.
- Include the file names in your prompt to help the model cite sources.
- Refresh the index when documentation changes.
