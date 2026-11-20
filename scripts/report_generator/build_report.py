"""
Master Orchestration Script for GreenTech Phase 3 Final Report Generation.
Loads GreenTech_Phase3_Report_Format_TEMPLATE_BACKUP.docx,
preserves Section 0 guidelines & updates cover page,
updates Section 1 preliminary pages with roman numerals,
rebuilds Section 2 with complete, exhaustive Chapters 1 to 8, References, and Appendices,
and saves to GreenTech_Phase3_Report_Format.docx.
"""

import os
import sys
import docx
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl

# Ensure scripts/report_generator is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preliminary import update_section0_and_section1
from chapter1_intro import build_chapter1
from chapter2_litreview import build_chapter2
from chapter3_requirements import build_chapter3
from chapter4_design import build_chapter4
from chapter5_implementation import build_chapter5
from chapter6_testing import build_chapter6
from chapter7_results import build_chapter7
from chapter8_conclusion import build_chapter8
from references_appendices import build_references_and_appendices

TEMPLATE_BACKUP = r"E:\Paladin_media\docs\reports\GreenTech_Phase3_Report_Format_TEMPLATE_BACKUP.docx"
FINAL_OUTPUT = r"E:\Paladin_media\docs\reports\GreenTech_Phase3_Report_Format.docx"

def main():
    print(f"Loading template backup from: {TEMPLATE_BACKUP}")
    doc = docx.Document(TEMPLATE_BACKUP)

    print("Step 1: Updating Section 0 (Cover) and Section 1 (Preliminary Pages)...")
    update_section0_and_section1(doc)

    print("Step 2: Clearing placeholder elements from Section 2...")
    body = doc._body._element
    body_children = list(body)
    
    # Locate p85 (end of Section 1)
    p_count = 0
    p85_idx = None
    for idx, child in enumerate(body_children):
        if isinstance(child, CT_P):
            if p_count == 85:
                p85_idx = idx
                break
            p_count += 1
            
    if p85_idx is None:
        raise RuntimeError("Could not locate paragraph 85 in template.")

    print(f"Located p85 at body index {p85_idx}. Total body elements: {len(body_children)}")
    
    # Remove all elements after p85 except the final sectPr
    to_remove = body_children[p85_idx + 1 : -1]
    print(f"Removing {len(to_remove)} placeholder elements...")
    for elem in to_remove:
        body.remove(elem)
        
    print(f"Remaining elements before chapter generation: {len(body)}")

    print("Step 3: Building Chapter 1 (Introduction)...")
    build_chapter1(doc)

    print("Step 4: Building Chapter 2 (Literature Review & SDG Alignment)...")
    build_chapter2(doc)

    print("Step 5: Building Chapter 3 (Requirement Analysis & Scrum Project Management)...")
    build_chapter3(doc)

    print("Step 6: Building Chapter 4 (System Design)...")
    build_chapter4(doc)

    print("Step 7: Building Chapter 5 (Implementation — Phase 3)...")
    build_chapter5(doc)

    print("Step 8: Building Chapter 6 (Testing and Validation)...")
    build_chapter6(doc)

    print("Step 9: Building Chapter 7 (Results, Green Impact Assessment & Discussion)...")
    build_chapter7(doc)

    print("Step 10: Building Chapter 8 (Conclusion and Future Scope)...")
    build_chapter8(doc)

    print("Step 11: Building References and Appendices A through E...")
    build_references_and_appendices(doc)

    print(f"Step 12: Saving finalized report to: {FINAL_OUTPUT}")
    doc.save(FINAL_OUTPUT)
    print("Report saved successfully!")

    # Audit generated document
    print("\n=== GENERATED DOCUMENT AUDIT ===")
    doc_check = docx.Document(FINAL_OUTPUT)
    print(f"Total Sections: {len(doc_check.sections)}")
    for i, s in enumerate(doc_check.sections):
        print(f"  Section {i}: Left Margin={s.left_margin.inches:.2f}\", Right Margin={s.right_margin.inches:.2f}\", "
              f"Top Margin={s.top_margin.inches:.2f}\", Bottom Margin={s.bottom_margin.inches:.2f}\"")
        print(f"  Section {i} Footer text: {[p.text for p in s.footer.paragraphs]}")

    print(f"Total Paragraphs: {len(doc_check.paragraphs)}")
    print(f"Total Tables: {len(doc_check.tables)}")
    
    # Calculate word count
    total_words = sum(len(p.text.split()) for p in doc_check.paragraphs)
    for t in doc_check.tables:
        for row in t.rows:
            for cell in row.cells:
                total_words += sum(len(p.text.split()) for p in cell.paragraphs)
                
    print(f"Total Word Count across entire document: {total_words:,} words")
    print(f"Estimated Page Count (at ~280-320 words/page): {total_words / 300:.1f} pages")

if __name__ == "__main__":
    main()
