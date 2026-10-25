
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from ragbuild.states import fileproschema


load_dotenv(override=True)  # Load environment variables from .env file, if present

BM25_CACHE_PATH = Path(__file__).resolve().parent / "bm25_params.json"


class Embedder:
    def __init__(
        self,
        index_name: str = "promptevaluation",
        namespace: str = "promptscore",
    ):
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")

        if not openai_key:
            raise ValueError("OPENAI_API_KEY is missing from your .env file.")

        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY is missing from your .env file.")

        self.namespace = namespace

        # 1. Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_key)

        existing_indexes = self.pc.list_indexes().names()

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        self.index = self.pc.Index(index_name)

        # 2. Initialize dense embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key,
        )

        # 3. Initialize sparse BM25 encoder
        if BM25_CACHE_PATH.exists():
            self.bm25_encoder = BM25Encoder()
            self.bm25_encoder.load(str(BM25_CACHE_PATH))
        else:
            self.bm25_encoder = BM25Encoder().default()
            self.bm25_encoder.dump(str(BM25_CACHE_PATH))

    @staticmethod
    def make_chunk_id(source: str, page, text: str) -> str:
        # Full text helps avoid collisions between similar chunks.
        raw = f"{source}|{page}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def __call__(self, state: fileproschema) -> dict:
        chunks = state.get("chunks", [])

        if not chunks:
            raise ValueError(
                "No chunks found in state. Run PDFLoader and Chunker first."
            )

        # Filter out empty/short chunks before embedding.
        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.page_content.strip()) >= 50
        ]

        if not chunks:
            raise ValueError("No usable chunks are available for embedding.")

        batch_size = 64
        total_upserted = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            texts = [doc.page_content.strip() for doc in batch]

            # Dense semantic vectors.
            dense_vectors = self.embeddings.embed_documents(texts)

            # Sparse lexical vectors for hybrid search.
            sparse_vectors = self.bm25_encoder.encode_documents(texts)

            records = []

            for doc, text, dense, sparse in zip(
                batch, texts, dense_vectors, sparse_vectors
            ):
                metadata = dict(doc.metadata)

                source = str(metadata.get("source", "unknown"))
                page = metadata.get("page", 0)

                # Store human-readable source metadata and chunk text.
                metadata["source"] = source
                metadata["text"] = text
                metadata["chunk_start_index"] = int(
                    metadata.get("start_index", 0)
                )

                records.append(
                    {
                        "id": self.make_chunk_id(source, page, text),
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": metadata,
                    }
                )

            self.index.upsert(
                vectors=records,
                namespace=self.namespace,
            )

            total_upserted += len(records)

        return {
            **state,
            "embedded": True,
            "embedded_chunk_count": total_upserted,
        }

