"""Preflight utility script for PaladinMedia.

Parses specifications in docs/specifications/ and inspects carrier dataset in MVP 0 Dataset/,
updating docs/project/mvp0_manifest.json.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import hashlib
import json
import sys
import subprocess

try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
    from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs", "specifications")
MANIFEST_PATH = os.path.join(REPO_ROOT, "docs", "project", "mvp0_manifest.json")

# Default to canonical dataset/MVP 0 Dataset, preserving CLI override or legacy fallback
if len(sys.argv) > 1 and sys.argv[1].strip() and not sys.argv[1].startswith("-"):
    DATASET_DIR = os.path.abspath(sys.argv[1])
else:
    canonical_dir = os.path.join(REPO_ROOT, "dataset", "MVP 0 Dataset")
    legacy_dir = os.path.join(REPO_ROOT, "MVP 0 Dataset")
    DATASET_DIR = canonical_dir if os.path.exists(canonical_dir) else legacy_dir


# 1. Inspect docx
def extract_text_from_docx(docx_path: str) -> str:
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = []
            for paragraph in tree.iterfind('.//w:p', namespace):
                texts = [node.text for node in paragraph.iterfind('.//w:t', namespace) if node.text]
                if texts:
                    text.append(''.join(texts))
            return '\n'.join(text)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

docs = [
    "Architecture_Specification.docx",
    "Product_Vision_Draft.docx",
    "Sprint-0_Implementation_Readiness_Specification_v0.1.docx",
    "System_and_Research_Document_Draft.docx"
]

print("Validating specifications in docs/specifications/...")
for doc in docs:
    path = os.path.join(DOCS_DIR, doc)
    if os.path.exists(path):
        text = extract_text_from_docx(path)
        print(f"  [OK] Validated {doc} ({len(text)} characters extracted)")
    else:
        print(f"  [MISSING] {doc}")

# 2. Inspect MVP 0 Dataset
print("Inspecting MVP 0 Dataset carriers...")
manifest = []
if os.path.exists(DATASET_DIR):
    for filename in sorted(os.listdir(DATASET_DIR)):
        if filename.endswith(".png"):
            filepath = os.path.join(DATASET_DIR, filename)
            with open(filepath, "rb") as f:
                file_bytes = f.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    img_format = img.format
                    mode = img.mode
            except Exception as e:
                width, height, channels, img_format, mode = -1, -1, -1, str(e), "ERROR"

            manifest.append({
                "image_id": filename,
                "width": width,
                "height": height,
                "channels": channels,
                "mode": mode,
                "format": img_format,
                "dataset_version": "MVP0_v1",
                "sha256": file_hash
            })

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done parsing {len(manifest)} carrier images -> {MANIFEST_PATH}")
else:
    print(f"Dataset directory not found: {DATASET_DIR}")
