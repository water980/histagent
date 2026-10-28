from pypdf import PdfReader

reader = PdfReader("histbook.pdf")


with open("histbooktext.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        print(f"Processing Page {i + 1}...")
        
        text = page.extract_text()
        if text:
            f.write(text)
            f.write("\n") 
