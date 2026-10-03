Udacity _NASA Intelligence Chat_ project: RAG Q&A over Apollo 11, Apollo 13, and Challenger mission documents, with ChromaDB retrieval and RAGAS scoring in a Streamlit chat.

Setup with [uv](https://docs.astral.sh/uv/): `uv venv && source venv/bin/activate && uv pip install -r requirements.txt`

> Note: RAGAS Version 0.4.3, which is the last version as of 2026-09-29, has a bug in the imports which causes `import ragas` to fail. After installing, run `python fix_ragas.py` to fix the bug.

Set `OPENAI_API_KEY` in the environment / .env file.

The mission text files are not included in this repo. Copy `data_text/` from the [starter repository](https://github.com/udacity/cd13318-exercises-project/tree/main/Project-NASA-Mission-Intelligence-Starter/data_text) into this directory.

Then index the documents with `python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path ./data_text` and start the chat with `streamlit run chat.py` and it will open in your browser.

To see the Bleu and Rouge scores you need to use the batch mode by uploading a json formatted file with questions and expected answers. There is a sample file `evaluation_dataset.txt` that you can upload via the web interface. 

