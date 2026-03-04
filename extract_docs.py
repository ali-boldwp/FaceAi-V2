import os
import docx

doc_dir = r"d:\Python\FaceAiv2\doc\new"
out_file = r"d:\Python\FaceAiv2\extracted_docs.txt"

with open(out_file, "w", encoding="utf-8") as f:
    for filename in sorted(os.listdir(doc_dir)):
        if filename.endswith(".docx"):
            f.write(f"\n\n=================================\n--- {filename} ---\n=================================\n")
            doc = docx.Document(os.path.join(doc_dir, filename))
            for para in doc.paragraphs:
                if para.text.strip():
                    f.write(para.text.strip() + "\n")
