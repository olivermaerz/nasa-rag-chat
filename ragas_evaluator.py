from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness, RougeScore
    
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], openai_api_key: str, expected_answer: str = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=openai_api_key,
            base_url="https://openai.vocareum.com/v1" if openai_api_key.startswith("voc-") else None,         
        ),
        bypass_n=True, # send three separate requests to the LLM (voacreum does not like n=3 for some reason)
    )
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
            base_url="https://openai.vocareum.com/v1" if openai_api_key.startswith("voc-") else None,         
        )
    )
    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm, name="context_precision"),
    ]
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    # Bleu and Rouge scored only work if we have an 'expected answer' during batch run
    if expected_answer:
        metrics.extend([BleuScore(), RougeScore()])
        sample.reference = expected_answer
    dataset = EvaluationDataset(samples=[sample])
    # Evaluate the response using the metrics
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=False,
    )
    # Return the evaluation results
    return result.scores[0]

