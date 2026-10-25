import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragbuild.states import fileproschema

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _strip_references(self, text: str) -> str:
        match = re.search(r"\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n", text)
        return text[:match.start()] if match else text

    async def __call__(self, state: fileproschema):
        documents = state["documents"]

        for doc in documents:
            doc.page_content = self._strip_references(doc.page_content)

        chunks = self.splitter.split_documents(documents)

        return {
            **state,
            "chunks": chunks,
        }