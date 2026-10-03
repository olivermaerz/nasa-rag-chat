#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import streamlit as st
import os
import json
import pandas as pd

import ragas_evaluator
import rag_client
import llm_client

from pathlib import Path
from typing import Dict, List, Optional

# load the api key from .env (optional)
from dotenv import load_dotenv
load_dotenv()

# RAGAS imports
try:
    from ragas import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    st.warning("RAGAS not available. Install with: pip install ragas")

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""

    return rag_client.discover_chroma_backends()

#@st.cache_resource
def initialize_rag_system(chroma_dir: str, collection_name: str, openai_api_key: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
       return rag_client.initialize_rag_system(chroma_dir, collection_name, openai_api_key)
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict], distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""
    
    return rag_client.format_context(documents, metadatas, distances)

def generate_response(openai_key, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"

def evaluate_response_quality(question: str, answer: str, contexts: List[str], openai_api_key: str, expected_answer: str = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(
            question, answer, contexts, openai_api_key, expected_answer
        )
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def display_evaluation_metrics(scores: Dict[str, float], question: str = None):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return
    st.sidebar.subheader("📊 Response Quality")
    if question: # display the question with the score (for batch processing)
        st.sidebar.caption(question)
    
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            # Color code based on score
            if score >= 0.8:
                color = "green"
            elif score >= 0.6:
                color = "orange"
            else:
                color = "red"
            
            label = metric_name.replace("_", " ").title()
            # Add some html to format it nicer and give it the 3 colors (green, orange, red)
            st.sidebar.markdown(
                f"""
                <div style="margin-bottom:0.75rem;">
                <div style="color:{color}; font-size:0.8rem;">{label} {score:.3f}</div>
                <div style="background:rgba(255,255,255,0.15); border-radius:4px; height:8px;">
                    <div style="width:{score * 100}%; background:{color}; height:8px; border-radius:4px;"></div>
                </div>
                </div>
                """,
                unsafe_allow_html=True, # make streamlit render it as html 
            )
            



def load_questions_and_answers(uploaded_file) -> List[Dict[str, str]]:
    """Load questions (mandatory) and expected answers (optional) from a JSON formatted file"""
    list_of_questions_and_answers = json.loads(uploaded_file.getvalue().decode('utf-8'))
    if not list_of_questions_and_answers or not isinstance(list_of_questions_and_answers, list):
        raise ValueError("Invalid JSON format. Please upload a valid JSON formatted file with .json or .txt suffix.")
    
    items = [] # this list will hold the questions and if present the expected answers

    for index, questions_and_answer in enumerate(list_of_questions_and_answers):
        if not isinstance(questions_and_answer, dict):
            raise ValueError(f"Item {index + 1} must be an object with a question.")
        question = str(questions_and_answer.get("question") or "").strip()
        answer = str(questions_and_answer.get("expected_answer") or "").strip()
        if not question:
            raise ValueError(f"Question {index} is missing. Please add a question.")
        # we always have a question (or it would have errored out earlier)
        item = {"question": question}
        # optional expected answer
        if answer:
            item["expected_answer"] = answer
        items.append(item)
    
    return items


def reset_question_upload() -> None:
    """Remove the uploaded file from sidebar and switch back to chat."""
    st.session_state.pending_questions = None
    st.session_state.loaded_file = None
    st.session_state.load_error = None
    st.session_state.batch_running = False
    st.session_state.batch_index = 0
    st.session_state.batch_messages = []
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def save_questions_and_answers(list_of_questions_and_answers: List[Dict[str, str]], uploaded_file) -> None:
    """Save questions (mandatory) and expected answers (optional) to a JSON formatted file"""
    json.dump(list_of_questions_and_answers, uploaded_file, ensure_ascii=False, indent=4)


def answer_question(collection, question, openai_key, model, n_docs, mission,
                    history, enable_evaluation, expected_answer=None):
    """Answer a question using the RAG system. This function is used by both the chat and the batch processing"""
    docs_result = retrieve_documents(collection, question, n_docs, mission)
    context, contexts_list = "", []
    if docs_result and docs_result.get("documents"):
        distances = (docs_result.get("distances") or [[]])[0]
        documents, metadatas = rag_client.order_retrieved_chunks(
            docs_result["documents"][0],
            docs_result["metadatas"][0],
            distances,
        )
        # already sorted by distance 
        context = format_context(documents, metadatas)
        contexts_list = documents
    response = generate_response(openai_key, question, context, history, model)
    scores = None
    if enable_evaluation and RAGAS_AVAILABLE:
        scores = evaluate_response_quality(
            question, response, contexts_list, openai_key, expected_answer
        )
    return {"role": "assistant", "content": response, "scores": scores}


def show_message(message, enable_evaluation):
    """Display a message in the chat window"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if enable_evaluation and message.get("scores"):
            show_response_quality(message["scores"])


def show_response_quality(scores):
    """Display the reponse quality/metrics in the chat window"""
    st.markdown("### Response Quality Metrics:")
    if "error" in scores:
        st.error(scores["error"])
    for name, score in scores.items():
        if isinstance(score, (int, float)) and score == score:
            st.write(f"{name.replace('_', ' ').title()}: {score:.3f}")


def summarize_batch_scores(messages):
    """Summarize the scores for a batch run"""
    values = {}
    for message in messages:
        scores = message.get("scores") or {}
        if not isinstance(scores, dict) or "error" in scores:
            continue
        for name, score in scores.items():
            if isinstance(score, (int, float)) and score == score:
                values.setdefault(name, []).append(score)
    if not values:
        return None
    lines = ["### Batch scores"]
    for name, scores in values.items():
        # mean score
        mean = sum(scores) / len(scores)
        # number of score above 0.8
        high = sum(score >= 0.8 for score in scores)
        # number of score between 0.6 and 0.8
        mid = sum(0.6 <= score < 0.8 for score in scores)
        # number of score below 0.6
        low = sum(score < 0.6 for score in scores)
        # label for the metric
        label = name.replace("_", " ").title()
        lines.append(
            f"{label}: mean {mean:.3f} over {len(scores)} "
            f"({high} at 0.8 or above, {mid} between 0.6 and 0.8, {low} below 0.6)"
        )
    return "\n\n".join(lines)

# - - - - - - - - - - - - - - - - - - - MAIN FUNCTION - - - - - - - - - - - - - - - - - - - #

def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    
    # - - - - - - - - - - - - - - -  SIDEBAR CONFIGURATION - - - - - - - - - - - - - - - #
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()
        
        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n`python embedding_pipeline.py`")
            st.stop()
        
        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )
        
        selected_backend = available_backends[selected_backend_key]
        
        # API Key input
        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        else:
            os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
        
        # Model selection
        model_choice = st.selectbox(
            "🧠 OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the OpenAI model for responses"
        )
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        # adding the mission choice dropdown
        mission_choice = st.selectbox(
            "Mission",
            options=["all", "apollo_11", "apollo_13", "challenger"],
            format_func=lambda mission: {
                "all": "All missions",
                "apollo_11": "Apollo 11",
                "apollo_13": "Apollo 13",
                "challenger": "Challenger",
            }[mission],
            help="Limit retrieval to one mission, or search all of them",
        )
        
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)
        


        st.header("💾 Upload Questions")

        # add a file uploader for the questions and answers (.json or .txt)
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Upload Questions (mandatory) and Expected Answers (optional)",
            type=["json", "txt"],
            key=st.session_state.uploader_key,
        )
        if uploaded_file is None:
            st.session_state.pending_questions = None
            st.session_state.loaded_file = None
            st.session_state.load_error = None
            st.session_state.batch_running = False
        else:
            file_token = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("loaded_file") != file_token:
                st.session_state.batch_running = False
                st.session_state.batch_index = 0
                st.session_state.batch_messages = []
                try:
                    st.session_state.pending_questions = load_questions_and_answers(uploaded_file)
                    st.session_state.load_error = None
                except (ValueError, UnicodeDecodeError) as e:
                    st.session_state.pending_questions = None
                    st.session_state.load_error = str(e)
                st.session_state.loaded_file = file_token
            if st.session_state.get("load_error"):
                st.error(st.session_state.load_error)

    # - - - - - - - - - - - - - - - - - - - END SIDEBAR - - - - - - - - - - - - - - - - - - #

    # Initialize RAG system when backend changes
    if (st.session_state.current_backend != selected_backend_key):
        st.session_state.current_backend = selected_backend_key
        # Clear cache to force reinitialization
        st.cache_resource.clear()

    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):

        collection, success, error = initialize_rag_system(
            selected_backend["directory"], 
            selected_backend["collection"],
            openai_key
        )
    
    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()
    
    # Display evaluation metrics if available
    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(
            st.session_state.last_evaluation,
            st.session_state.get("last_question"), # add the question to the score sidebar
        )
    
    for message in st.session_state.messages:
        show_message(message, enable_evaluation)

    pending = st.session_state.get("pending_questions")
    batch_running = st.session_state.get("batch_running", False)
    if pending:
        next_index = st.session_state.get("batch_index", 0)
        for item in pending[next_index:]:
            with st.chat_message("user"):
                st.markdown(item["question"])
        start_column, cancel_column = st.columns(2)
        if start_column.button("Batch start", disabled=batch_running, use_container_width=True):
            st.session_state.batch_running = True
            # only remember the start on a fresh batch so retry still includes the earlier answers
            if st.session_state.get("batch_index", 0) == 0:
                st.session_state.batch_message_start = len(st.session_state.messages)
            st.rerun()
        if cancel_column.button("Cancel", use_container_width=True):
            reset_question_upload()
            st.rerun()

        if st.session_state.get("batch_running"):
            index = st.session_state.get("batch_index", 0)
            item = pending[index]
            try:
                with st.spinner(f"Answering {index + 1} of {len(pending)}"):
                    result = answer_question(
                        collection, item["question"], openai_key, model_choice,
                        n_docs, mission_choice, [], enable_evaluation,
                        item.get("expected_answer"),
                    )
            except Exception as e:
                st.session_state.batch_running = False
                st.error(f"Batch stopped: {e}")
            else:
                st.session_state.messages.append({"role": "user", "content": item["question"]})
                st.session_state.messages.append(result)
                st.session_state.last_evaluation = result["scores"]
                st.session_state.last_question = item["question"]
                st.session_state.batch_index = index + 1
                if st.session_state.batch_index >= len(pending):
                    start = st.session_state.get("batch_message_start", 0)
                    summary = summarize_batch_scores(st.session_state.messages[start:])
                    if summary:
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                    reset_question_upload()
                st.rerun()
    elif prompt := st.chat_input("Ask about NASA space missions..."):
        # existing single-question block
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching documents and generating response..."):
            result = answer_question(
                collection, prompt, openai_key, model_choice, n_docs, mission_choice,
                st.session_state.messages[:-1], enable_evaluation,
            )
        st.session_state.messages.append(result)
        st.session_state.last_evaluation = result["scores"]
        st.session_state.last_question = prompt
        st.rerun()


if __name__ == "__main__":
    main()
