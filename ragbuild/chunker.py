
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbuild.states import fileproschema


class Chunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def _strip_references(self, text: str) -> str:
        # Only remove a references/bibliography heading when it
        # appears as a standalone line.
        pattern = (
            r"(?im)^\s*\**\s*"
            r"(references|bibliography)"
            r"\s*\**\s*$"
        )
        match = re.search(pattern, text)

        return text[:match.start()].strip() if match else text.strip()

    async def __call__(self, state: fileproschema) -> dict:
        documents = state.get("documents", [])

        if not documents:
            raise ValueError(
                "No PDF documents found in state. Run PDFLoader first."
            )

        cleaned_documents = []

        for doc in documents:
            cleaned_text = self._strip_references(doc.page_content or "")

            if not cleaned_text:
                continue

            cleaned_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=dict(doc.metadata),
                )
            )

        if not cleaned_documents:
            raise ValueError(
                "No usable text remains after cleaning the PDF documents."
            )

        chunks = self.splitter.split_documents(cleaned_documents)

        # Remove very short chunks that usually contain little useful context.
        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.page_content.strip()) >= 50
        ]

        if not chunks:
            raise ValueError(
                "No valid chunks were produced. Check the PDF text and chunk settings."
            )

        return {
            **state,
            "chunks": chunks,
        }
