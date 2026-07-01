# take a PDF file path
# extract all text
# return clean text

import fitz  # PyMuPDF


def load_pdf_text(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    doc = fitz.open(file_path)

    full_text = ""

    for page in doc:
        text = page.get_text()
        full_text += text + "\n"

    doc.close()

    return full_text