# Challenges Report


## Software used


### Streamlit
- It took me a while to understand and how to customize Streamlit with what I wanted to do. So there was somewhat of a learning curve. There were also some issues like it was impossible to clear the uploaded etc. things that are easily done in normal web programming (e.g. in react etc.)

### RAGAS
- There were some issues with RAGAS and the ChatVertexAI import. 
- RAGAS returns NaN for a failed score and that crashed the progress bar.


## AI Backend

### Vocareum
- The code was using openai (the full api) but we were given a vocareum key. That did not work of course but some boilerplate code to check the key and change the url of the API fixed that.
- Also vocareum did not support the n=3 that RAGAS uses to get three completions generated. Setting bypass_n=True in ragas_evaluator.py fixed it. 


## Starter code / logic

- I needed a reference (expected) answer for bleu and rouge scores but the chat interface only asks for the question. Adding the batch mode with a file upload (that allows providing the expected answer) fixed that.
- The starter code for the chat used the default chroma model (a locally running all-MiniLM-L6-v2) for embedding while the embedding pipeline uses text-embedding-3-small via OpenAI. 

