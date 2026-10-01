from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
    # TODO: Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    # TODO: Define an instance for each metric to evaluate
    metrics = [
        # Answer Relevancy (wether answer adresses the question)
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        # Faithfulness (wether answer is from the retrieved chunks)
        Faithfulness(llm=evaluator_llm),    
    ]
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
    ])
    # TODO: Evaluate the response using the metrics
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=False,
    )
    # TODO: Return the evaluation results
    return result.scores[0]

