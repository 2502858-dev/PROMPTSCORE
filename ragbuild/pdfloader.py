
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from ragbuild.states import fileproschema


class PDFLoader:
    def __init__(self, ocr_language: str = "eng"):
        self.ocr_language = ocr_language

    def _extract_tables(self, page) -> str:
        """Extract detectable PDF tables into readable row format."""
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", [])

            output = []

            for table_number, table in enumerate(tables, start=1):
                rows = table.extract()

                if not rows:
                    continue

                output.append(f"\n[TABLE {table_number}]")

                for row in rows:
                    cells = [
                        str(cell).strip().replace("\n", " ")
                        if cell is not None else ""
                        for cell in row
                    ]
                    output.append(" | ".join(cells))

            return "\n".join(output)

        except Exception as exc:
            print(f"Table extraction warning on page {page.number + 1}: {exc}")
            return ""

    def _extract_page_text(self, page) -> tuple[str, str]:
        """Prefer native text; use OCR when little text is available."""
        text = page.get_text("text").strip()
        extraction_method = "native"

        # OCR fallback for scanned pages or pages without usable text.
        if len(text) < 40:
            try:
                text_page = page.get_textpage_ocr(
                    language=self.ocr_language,
                    dpi=200,
                    full=True,
                )
                text = page.get_text(
                    "text",
                    textpage=text_page,
                ).strip()
                extraction_method = "ocr"

            except Exception as exc:
                print(
                    f"OCR warning on page {page.number + 1}: {exc}"
                )

        return text, extraction_method

    async def __call__(self, state: fileproschema) -> dict:
        file_path = state.get("file_path")

        if not file_path:
            raise ValueError("PDF file_path is missing from state.")

        pdf_path = Path(file_path).expanduser().resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("PDFLoader accepts PDF files only.")

        documents = []
        full_text = []

        with pymupdf.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf):
                text, method = self._extract_page_text(page)

                # Extract tables when detectable, including text-based tables.
                table_text = self._extract_tables(page)

                # Avoid adding table content twice when native text already
                # contains it. Add structured table rows when detected.
                if table_text and table_text not in text:
                    text = f"{text}\n\n{table_text}".strip()

                if not text.strip():
                    print(
                        f"Skipping page {page_number + 1}: "
                        "no text extracted."
                    )
                    continue

                metadata = {
                    "source": str(pdf_path),
                    "page": page_number,
                    "extraction_method": method,
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

                full_text.append(text)

                print(
                    f"Page {page_number + 1}: "
                    f"{len(text)} characters ({method})"
                )

        if not documents:
            raise ValueError(
                f"No extractable text found in {pdf_path.name}. "
                "Check OCR installation or use a dedicated OCR/table "
                "recognition tool."
            )

        print(
            f"Loaded {len(documents)} pages from {pdf_path.name}"
        )

        return {
            **state,
            "file_path": str(pdf_path),
            "documents": documents,
            "text": "\n\n".join(full_text),
        }
