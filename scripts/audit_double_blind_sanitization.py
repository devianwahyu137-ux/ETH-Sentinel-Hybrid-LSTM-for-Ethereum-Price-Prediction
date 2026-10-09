import os
import zipfile
import pypdf
import docx

def audit_double_blind():
    print("===========================================================================")
    print("AUDIT KEPATUHAN DOUBLE-BLIND SANITIZATION: JAI UMY")
    print("===========================================================================")

    keywords = ["devian", "wahyu", "arsya", "github.com/devianwahyu"]
    
    # 1. Check main_paper_JAI.docx
    docx_path = "manuscript/main_paper_JAI.docx"
    docx_leaks = []
    if os.path.exists(docx_path):
        # Check text via python-docx
        doc = docx.Document(docx_path)
        for p in doc.paragraphs:
            for kw in keywords:
                if kw in p.text.lower():
                    docx_leaks.append(f"Paragraph leak: '{kw}' in text: {p.text[:60]}...")
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for kw in keywords:
                        if kw in cell.text.lower():
                            docx_leaks.append(f"Table cell leak: '{kw}' in cell: {cell.text[:60]}...")
        
        # Check raw zip contents (relationships and document xml)
        with zipfile.ZipFile(docx_path) as z:
            for name in z.namelist():
                if name.endswith('.xml') or name.endswith('.rels'):
                    content = z.read(name).decode('utf-8', errors='ignore').lower()
                    for kw in keywords:
                        if kw in content:
                            docx_leaks.append(f"XML/Rels leak in [{name}]: keyword '{kw}'")
    else:
        docx_leaks.append(f"File not found: {docx_path}")

    # 2. Check main_paper_JAI.pdf
    pdf_path = "manuscript/main_paper_JAI.pdf"
    pdf_leaks = []
    if os.path.exists(pdf_path):
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").lower()
            for kw in keywords:
                if kw in text:
                    pdf_leaks.append(f"PDF Page {i+1} leak: keyword '{kw}'")
    else:
        pdf_leaks.append(f"File not found: {pdf_path}")

    # Report Main Paper
    print(f"\n1. Main Paper DOCX ({docx_path}):")
    if docx_leaks:
        print(f"   [FAIL] {len(docx_leaks)} leaks detected:")
        for l in docx_leaks:
            print(f"     - {l}")
    else:
        print("   [PASS] 0 leaks found (100% Anonymized & Double-Blind Compliant)")

    print(f"\n2. Main Paper PDF ({pdf_path}):")
    if pdf_leaks:
        print(f"   [FAIL] {len(pdf_leaks)} leaks detected:")
        for l in pdf_leaks:
            print(f"     - {l}")
    else:
        print("   [PASS] 0 leaks found (100% Anonymized & Double-Blind Compliant)")

    # 3. Check Title Page (Unblinded document)
    print("\n3. Unblinded Title Page Audit (Title Page MUST contain the official project repo link):")
    tp_docx = "manuscript/title_page.docx"
    tp_pdf = "manuscript/title_page.pdf"
    
    tp_docx_has_link = False
    if os.path.exists(tp_docx):
        doc_tp = docx.Document(tp_docx)
        for p in doc_tp.paragraphs:
            if "github.com/devianwahyu" in p.text.lower():
                tp_docx_has_link = True
                break
    
    tp_pdf_has_link = False
    if os.path.exists(tp_pdf):
        reader_tp = pypdf.PdfReader(tp_pdf)
        for page in reader_tp.pages:
            if "github.com/devianwahyu" in (page.extract_text() or "").lower():
                tp_pdf_has_link = True
                break
                
    print(f"   - {tp_docx}: GitHub repo link present = {tp_docx_has_link}")
    print(f"   - {tp_pdf}: GitHub repo link present = {tp_pdf_has_link}")
    if tp_docx_has_link and tp_pdf_has_link:
        print("   [PASS] Verified: GitHub repo link isolated strictly to Title Page!")
    else:
        print("   [INFO] Pending re-compilation of Title Page to embed link.")

    print("\n===========================================================================")
    if not docx_leaks and not pdf_leaks:
        print("OVERALL SANITIZATION VERDICT: 100% PASS")
    else:
        print("OVERALL SANITIZATION VERDICT: FAIL")
    print("===========================================================================")

if __name__ == "__main__":
    audit_double_blind()
