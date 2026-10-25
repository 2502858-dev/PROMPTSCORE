
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class fileproschema(TypedDict, total=False):
    # PDF ingestion
    file_path: str
    documents: List[Document]
    text: str
    chunks: List[Document]
    embedded: bool

    # User input
    question: str
    intended_purpose: str
    task_description: str

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    sources: List[Dict[str, str]]
    retrieval_grade: str
    source_type: str

    # Generation
    evaluation: Dict[str, Any]
    answer: str
