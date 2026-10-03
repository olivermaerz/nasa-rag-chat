import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction # to use the text-embedding-3-small encoder


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        path for path in current_dir.iterdir()
        if path.is_dir() and path.name.startswith("chroma_db")
    ]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings = Settings(anonymized_telemetry=False),
            )
            # Retrieve list of available collections from the database
            collections = client.list_collections()
        # Catch connection errors etc.
        except Exception as e:
            # Default entry in case of error
            key = chroma_dir.name
            backends[key] = {
                # Store directory path as string
                "directory": str(chroma_dir),
                # Store collection name
                "collection": "",
                # Include error information in display name with truncation
                "display_name": f"{chroma_dir.name} (Error: {str(e)[:50]})",
                # Set appropriate fallback values for missing information
                "document_count": 0,
            }
            continue

        # Loop through each collection found
        for collection in collections:
            # Create unique identifier key combining directory and collection names
            key = f"{chroma_dir.name}_{collection.name}"
            # Build information dictionary containing:
            try:
                # Get document count with fallback for unsupported operations
                document_count = collection.count()
            except Exception:
                document_count = 0
            collection_info = {
                # Store directory path as string
                "directory": str(chroma_dir),
                # Store collection name
                "collection": collection.name,
                # Create user-friendly display name
                "display_name": f"{chroma_dir.name}_{collection.name}",
                "document_count": document_count,
            }
            # Add collection information to backends dictionary
            backends[key] = collection_info
    # Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str, openai_api_key: str):
    """Initialize the RAG system with specified backend (cached for performance). Needs openai_api_key to be passed!"""
    try:
        # Create a chomadb persistentclient
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # set the correct embedding function, otherwise it will use some weird all-MiniLM-L6-v2 ...
        embedding_function = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=openai_api_key,
            api_base="https://openai.vocareum.com/v1" if openai_api_key.startswith("voc-") else None,         
        )
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function, # the text_embedding_3_small model we used when chunking the documents
        )
        # Return the collection with the collection_name
        return collection, True, None
    except Exception as e:
        return None, False, str(e)
    
    
def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
    # If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter = {"mission": mission_filter}

    # Execute database query with the following parameters:
    results = collection.query(
        # Pass search query in the required format
        query_texts=[query],
        # Set maximum number of results to return
        n_results=n_results,
        # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where=filter,
    )
    # Return query results to caller
    return results


def order_retrieved_chunks(documents: List[str], metadatas: List[Dict], distances: Optional[List[float]] = None):
    """Sort chunks by retrieval distance and drop repeated text."""
    distances = distances or []
    metadatas = metadatas or []
    ranked = []
    for index, document in enumerate(documents or []):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = distances[index] if index < len(distances) else None
        # missing distance goes last, then closer matches, then the original order
        ranked.append((distance is None, distance if distance is not None else 0, index, document, metadata))
    ranked.sort()

    seen = set()
    ordered_documents = []
    ordered_metadatas = []
    for _, _, _, document, metadata in ranked:
        key = document.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered_documents.append(document)
        ordered_metadatas.append(metadata)
    return ordered_documents, ordered_metadatas


def format_context(documents: List[str], metadatas: List[Dict], distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Sort by retrieval score and drop duplicate chunks before building the context
    documents, metadatas = order_retrieved_chunks(documents, metadatas, distances)

    # Initialize list with header text for context section
    context_parts = ["Retrieved NASA documents related to the query:"]

    # Loop through paired documents and their metadata using enumeration
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "Unknown")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()
        # Extract source information from metadata with fallback value
        source = metadata.get("source", "Unknown")
        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "Unknown")
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()
        # Create formatted source header with index number and extracted information
        source_header = f"Source {index + 1}: {mission} - {source} ({category})"
        # Add source header to context parts list
        context_parts.append(source_header)
        # Check document length and truncate if necessary
        if len(document) > 500:
            document = document[:500] + "..."
        # Add truncated or full document content to context parts list
        context_parts.append(document)

    # Join all context parts with newlines and return formatted string
    return "\n\n".join(context_parts)