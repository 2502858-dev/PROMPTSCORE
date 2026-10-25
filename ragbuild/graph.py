
from langgraph.graph import StateGraph, START, END
import time

from ragbuild.states import fileproschema
from ragbuild.pdfloader import PDFLoader
from ragbuild.chunker import Chunker
from ragbuild.embeding import Embedder
from ragbuild.retriver import Retriever
from ragbuild.generator import Generator


def route(state: fileproschema):
    """Choose PDF ingestion or prompt evaluation."""
    file_path = state.get("file_path")
    question = state.get("question")

    if file_path:
        if not file_path.lower().endswith(".pdf"):
            raise ValueError(
                "PromptScore knowledge ingestion currently supports PDF files only."
            )
        return "ingest"

    if question and question.strip():
        return "evaluate"

    raise ValueError(
        "State must contain either a PDF 'file_path' or a prompt 'question'."
    )


class Graph_manager:
    def __init__(self):
        print("Starting PromptScore Graph_manager init...")

        t0 = time.time()
        self.pdf_loader = PDFLoader()
        print(f"  PDFLoader: {time.time() - t0:.2f}s")

        t0 = time.time()
        self.chunker = Chunker()
        print(f"  Chunker: {time.time() - t0:.2f}s")

        t0 = time.time()
        self.embedder = Embedder()
        print(f"  Embedder: {time.time() - t0:.2f}s")

        t0 = time.time()
        self.retriever = Retriever(self.embedder)
        print(f"  Retriever: {time.time() - t0:.2f}s")

        t0 = time.time()
        self.generator = Generator()
        print(f"  Generator: {time.time() - t0:.2f}s")

        t0 = time.time()
        graph = StateGraph(fileproschema)

        # Register nodes
        graph.add_node("PDFLoader", self.pdf_loader)
        graph.add_node("Chunker", self.chunker)
        graph.add_node("Embedder", self.embedder)
        graph.add_node("Retriever", self.retriever)
        graph.add_node("Generator", self.generator)

        # Route from START
        graph.add_conditional_edges(
            START,
            route,
            {
                "ingest": "PDFLoader",
                "evaluate": "Retriever",
            },
        )

        # Knowledge ingestion path
        graph.add_edge("PDFLoader", "Chunker")
        graph.add_edge("Chunker", "Embedder")
        graph.add_edge("Embedder", END)

        # Prompt evaluation path
        graph.add_edge("Retriever", "Generator")
        graph.add_edge("Generator", END)

        self.graph = graph.compile()

        print(f"  Graph build + compile: {time.time() - t0:.2f}s")
        print("PromptScore Graph_manager init complete.\n")

    async def run(self, initial_state: dict) -> dict:
        """Run the graph asynchronously."""
        return await self.graph.ainvoke(initial_state)
