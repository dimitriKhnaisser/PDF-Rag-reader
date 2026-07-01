from app.services.pdf_loader import load_pdf_text

text = load_pdf_text("data/pdfs/sample.pdf")

print(text[:1000])