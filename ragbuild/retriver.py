
import os
from ragbuild.states import fileproschema


class Retriever:
    def __init__(self, embedder):
        # Reuse the existing Pinecone index and encoders
        self.index = embedder.index
        self.embeddings = embedder.embeddings
        self.bm25_encoder = embedder.bm25_encoder

    async def __call__(self, state: fileproschema):
        question = (state.get("question") or "").strip()

        if not question:
            raise ValueError(
                "The prompt to evaluate ('question') cannot be empty."
            )

        # Dense + sparse query vectors
        dense_vec = self.embeddings.embed_query(question)
        sparse_vec = self.bm25_encoder.encode_queries([question])[0]

        results = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=5,
            include_metadata=True,
        )

        matches = results.get("matches", [])

        context_blocks = []
        sources = []

        for i, match in enumerate(matches):
            meta = match.get("metadata") or {}

            source_path = meta.get("source", "unknown")
            source_title = os.path.basename(source_path)
            raw_page = meta.get("page", "unknown")
            text = meta.get("text", "").strip()

            if not text:
                continue

            # PyMuPDFLoader commonly stores zero-based page indices.
            # Convert numeric indices to human-readable page numbers.
            try:
                page = str(int(raw_page) + 1)
            except (TypeError, ValueError):
                page = str(raw_page)

            context_blocks.append(
                f"[Source {len(sources) + 1}]\n"
                f"Title: {source_title}\n"
                f"Page: {page}\n"
                f"Excerpt:\n{text}"
            )

            sources.append({
                "title": source_title,
                "page": page,
            })

        context = "\n\n".join(context_blocks)

        return {
            **state,
            "retrieved_chunks": matches,
            "context": context,
            "sources": sources,
        }
