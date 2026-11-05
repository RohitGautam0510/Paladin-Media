# PaladinMedia — Repository Simplicity, Redundancy & File Hygiene Audit

**Audit Timestamp:** 2026-10-18T15:58:00Z  
**Auditor Role:** Senior Software Architect, Repository Maintainer, and Codebase Simplification Auditor  
**Audit Mode:** STRICT READ-ONLY AUDIT MODE (No project files deleted, moved, renamed, or modified)  
**Target Repository:** `E:\Paladin_media`  
**Git Branch / Commit:** `master` @ `9555996` (feat(audit): resolve P1 gate blockers with E14 lossy baseline, div2k split manifest, and acceptance criteria reconciliation)  

---

## 1. Executive Summary

A comprehensive repository simplicity, redundancy, and file hygiene audit was performed on PaladinMedia (`E:\Paladin_media`). The inspection covered the entire directory tree, Git working-tree status, content-hash deduplication, line-of-code metrics, import/dependency relationships, experiment assets, script lifecycles, and documentation sprawl.

### Current System Health & Operational Integrity
- **Test Suite Status:** 103 passed out of 103 collected tests in 2.30s (`pytest`). All core unit, boundary, fail-closed security, error-correction, and media roundtrip tests pass cleanly.
- **Verification Script Status:** Standalone programmatic verification (`python scripts/verify_sprint1.py`) passes all 24/24 architectural and acceptance checks in 0.286s.
- **Source Code State:** The backend source code (`src/`, 22 files, 2,345 code lines) is tightly scoped, modular, and maintains strict separation of concerns between Phase 1 cryptography/ECC and Phase 2 wavelet transform/carrier modulation.
- **Active Task Safeguard:** Detected an in-progress E14 limitations-revision task affecting 4 files. Per architectural instructions, these files were treated as strictly read-only and excluded from any modification.

### Primary Audit Findings
1. **Root Directory Clutter & Redundancy:** The repository root contains 5 binary documents (3 DOCX, 2 PDF) totaling ~504 KB. These are duplicated across `docs/reports/` and `CIAs/`, or generated repeatedly under alternate filenames.
2. **Identical Content Duplicates (Hash-Verified):**
   - Exact hash `8802dfe05cb4`: `PaladinMedia_Project_Status_Report_IMRaD.docx` (root), `docs/reports/PaladinMedia_Project_Status_Report.docx`, and `docs/reports/PaladinMedia_Project_Status_Report_IMRaD.docx` are 100% byte-for-byte identical.
   - Exact hash `98cd6f59542b`: `CIAs/PaladinMedia_Sprint_and_Scrum_Log.docx` and `docs/project/PaladinMedia_Sprint_and_Scrum_Log.docx` are identical.
   - Exact hash `bed44df13fa1`: `CIAs/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx` and `docs/project/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx` are identical.
   - Exact byte size 10,708 B: Four 2-page status PDFs across root and `docs/reports/` share the exact same generated visual content.
3. **Orphaned Binary Clutter in `scripts/`:** Two tracked PNG files (`scripts/page_1.png` [361 KB] and `scripts/page_2.png` [431 KB], totaling 793 KB) are static renders of the 2-page PDF report. They are unreferenced in any script or markdown document.
4. **Generator Script Fan-Out:** Generator scripts (`generate_status_docx.py`, `generate_status_pdf.py`, `generate_week7_doc.py`) hardcode multiple simultaneous output paths, scattering duplicate copies across root and subdirectories on every execution.
5. **Dataset Path Discrepancy:** While datasets are correctly segregated under `dataset/` (and ignored by `.gitignore`), legacy scripts (`scripts/preflight.py`, `scripts/experiments/run_e03.py`, `run_e05.py`..`run_e10.py`, and `src/api/server.py`) still hardcode paths to `MVP 0 Dataset/` directly at repository root. Consequently, `preflight.py` reports `Dataset directory not found` and default server fixture loading fails.
6. **Dependency Manifest Gaps:** `pyproject.toml` declares only 4 core runtime packages (`cryptography`, `reedsolo`, `numpy`, `pillow`), omitting undeclared dependencies used by the API (`fastapi`, `uvicorn`, `pydantic`), testing (`pytest`), and reporting scripts (`python-docx`, `reportlab`, `python-pptx`, `psutil`).
7. **Documentation & Submission Sprawl:** Coursework deliverables and student sprint submissions are fragmented across root, `CIAs/`, and untracked `submission/CIA3_Sprint1/` (19 files, 2.69 MB). In `docs/experiments/`, superseded audit snapshots (`historical_E12*`, `historical_E13*`) sit alongside active experiment reports.

---

## 2. Actual Repository Inventory

Measurements were taken directly from the filesystem, excluding `.git/`, `frontend/node_modules/`, `frontend/dist/`, `.pytest_cache/`, `__pycache__/`, and unversioned image datasets in `dataset/`.

### 2.1 File Counts & Storage Distribution

| Directory | Files | Total Size | Total Lines | Code / Text Lines | Comments | Blank Lines | Primary Content |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Root (`.`)** | 8 | 512.7 KB | 202 | 183 | 0 | 19 | `README.md`, `pyproject.toml`, `.gitignore`, 5 binary reports |
| **`CIAs/`** | 4 | 682.2 KB | 0 | 0 | 0 | 0 | Academic submission artifacts (PDF, DOCX) |
| **`docs/`** | 50 | 1.44 MB | 24,756 | 24,325 | 0 | 431 | Specifications, audits, experiment markdown & JSON results |
| **`frontend/`** (excl node_modules) | 22 | 160.0 KB | 3,547 | 3,435 | 7 | 105 | React/Vite UI, Tailwind config, assets, TypeScript adapter |
| **`scripts/`** | 25 | 1.21 MB | 8,267 | 6,895 | 335 | 1,037 | Experiment runners (E03-E14), doc generators, 2 binary PNGs |
| **`src/`** | 22 | 113.2 KB | 2,950 | 2,345 | 108 | 497 | Core security, carrier, pipeline, API modules |
| **`submission/`** (untracked) | 19 | 2.69 MB | 317 | 251 | 0 | 66 | CIA-3 Sprint-1 submission package, 13 preview PNGs |
| **`tests/`** | 19 | 99.0 KB | 2,575 | 1,796 | 208 | 571 | Unit and integration test suite, fixtures, KAT vectors |
| **Total Tracked & Repo Files** | **169** | **7.41 MB** | **42,614** | **39,230** | **658** | **2,726** | *(Excludes external datasets and node_modules)* |

*Note on Unversioned Large Directories:*
- `dataset/`: Contains 1,661 image files (800 DIV2K HR, 800 DIV2K LR X2, 61 MVP 0 images). Appropriately ignored by `.gitignore` (line 42).
- `frontend/node_modules/`: Contains 7,249 files. Appropriately ignored by `.gitignore` (line 32).

### 2.2 File Counts by Extension

| Extension | Count | Total Size | Lines | Description |
| :--- | :---: | :---: | :---: | :--- |
| `.py` | 63 | 627.7 KB | 13,792 | Python source, scripts, experiment runners, test files |
| `.json` | 24 | 712.9 KB | 25,213 | Experiment results, split manifests (`div2k_hr_split.json`), configs |
| `.md` | 26 | 206.1 KB | 2,593 | Documentation, audits, experiment reports, root README |
| `.png` | 17 | 3.32 MB | 0 | Fixture (`secret_small.png`), hero images, preview renders |
| `.docx` | 13 | 898.7 KB | 0 | Architectural specs, project status reports, sprint logs |
| `.pdf` | 9 | 1.02 MB | 0 | Two-page status reports, CIA submissions, demo PDFs |
| `.ts` / `.tsx` | 4 | 32.9 KB | 762 | Frontend laboratory application and backend adapter |
| `.css` / `.html` / `.js` | 5 | 4.1 KB | 236 | Vite & Tailwind frontend infrastructure |
| `.svg` | 4 | 27.4 KB | 0 | Static UI icons |
| `.pptx` | 1 | 59.7 KB | 0 | CIA-3 Sprint-1 demo presentation |
| `.toml` | 1 | 345 B | 18 | `pyproject.toml` packaging configuration |
| *(no ext)* | 2 | 663 B | 0 | `.gitignore` files (root and frontend) |

### 2.3 Largest Files in Repository

| Size (Bytes) | File Path | Category / Status |
| :---: | :--- | :--- |
| 431,526 | `scripts/page_2.png` | Orphaned preview render (REMOVE CANDIDATE) |
| 361,615 | `scripts/page_1.png` | Orphaned preview render (REMOVE CANDIDATE) |
| 328,891 | `PaladinMedia_23215050.pdf` | Coursework report at root (ARCHIVE / RELOCATE) |
| 307,816 | `submission/CIA3_Sprint1/preview/report_page_1.png` | Untracked submission preview PNG |
| 300,864 | `CIAs/23215050_PaladinMedia_WEEK 6.pdf` | Academic CIA coursework deliverable (KEEP in CIAs) |
| 298,672 | `CIAs/23215050_PaladinMedia_Week7_Sprint1_Testing_and_Review.pdf` | Academic CIA coursework deliverable (KEEP in CIAs) |
| 291,234 | `docs/specifications/Architecture_Specification.docx` | Core architecture specification (KEEP) |
| 289,895 | `submission/CIA3_Sprint1/preview/report_page_3.png` | Untracked submission preview PNG |
| 284,338 | `submission/CIA3_Sprint1/preview/report_page_2.png` | Untracked submission preview PNG |
| 282,811 | `docs/project/div2k_hr_split.json` | Canonical DIV2K dataset partition manifest (KEEP / PROTECT) |

---

## 3. Confirmed Duplicates and Near-Duplicates

### 3.1 Exact Content-Hash Duplicates (SHA-256)

1. **Hash Group `8802dfe05cb4` (3 Identical Files):**
   - `PaladinMedia_Project_Status_Report_IMRaD.docx` (Root)
   - `docs/reports/PaladinMedia_Project_Status_Report.docx`
   - `docs/reports/PaladinMedia_Project_Status_Report_IMRaD.docx`
   - *Evidence:* Exact byte size (44,079 B) and identical SHA-256 hash. Produced simultaneously by `scripts/generate_status_docx.py`.
   - *Recommendation:* Keep single canonical file at `docs/reports/PaladinMedia_Project_Status_Report.docx`. Remove root file and redundant `_IMRaD` named copy.

2. **Hash Group `98cd6f59542b` (2 Identical Files):**
   - `CIAs/PaladinMedia_Sprint_and_Scrum_Log.docx`
   - `docs/project/PaladinMedia_Sprint_and_Scrum_Log.docx`
   - *Evidence:* Exact byte size (40,524 B) and identical SHA-256 hash.
   - *Recommendation:* Consolidate into `CIAs/PaladinMedia_Sprint_and_Scrum_Log.docx` (the coursework record) and remove copy from `docs/project/`.

3. **Hash Group `bed44df13fa1` (2 Identical Files):**
   - `CIAs/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx`
   - `docs/project/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx`
   - *Evidence:* Exact byte size (42,161 B) and identical SHA-256 hash. Produced by `scripts/generate_week7_doc.py`.
   - *Recommendation:* Consolidate into `CIAs/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx`. Remove copy from `docs/project/`.

### 3.2 Semantic / Functional Duplicates (Identical Size & Structure)

4. **Two-Page Status PDFs (4 Identical Visual Copies):**
   - `PaladinMedia_Project_Status_Report_2Page.pdf` (Root, 10,708 B)
   - `PaladinMedia_Project_Status_and_Completion_Report_2Page.pdf` (Root, 10,708 B)
   - `docs/reports/PaladinMedia_Project_Status_Report_2Page.pdf` (10,708 B)
   - `docs/reports/PaladinMedia_Project_Status_and_Completion_Report_2Page.pdf` (10,708 B)
   - *Evidence:* All four files have identical byte size (10,708 B) and identical visual page layout generated by `scripts/generate_status_pdf.py`. Subtle hash divergence is caused solely by ReportLab's sequential `CreationDate` timestamp in the PDF trailer.
   - *Recommendation:* Retain only `docs/reports/PaladinMedia_Project_Status_Report_2Page.pdf`. Remove the other 3 copies.

5. **Root Status Report DOCX Variants:**
   - `PaladinMedia_Project_Status_Report.docx` (Root, 109,827 B, 35 paragraphs, 1 embedded diagram)
   - `PaladinMedia_Project_Status_Report_IMRaD.docx` (Root, 44,079 B, 26 paragraphs, text only)
   - *Evidence:* Both documents share the exact same 9 section and subsection headings. The 109 KB version incorporates an inline system diagram; the 44 KB version is text-only.
   - *Recommendation:* Consolidate into `docs/reports/PaladinMedia_Project_Status_Report.docx` with the diagram retained. Remove the text-only `_IMRaD` redundancy.

### 3.3 Historical Experiment Report Pairs

6. **E12 Continuous Payload Sweep:**
   - Active: `docs/experiments/E12_payload_distortion_sweep.md` and `e12_results.json` (2026-10-11T18:41:00Z)
   - Historical: `docs/experiments/historical_E12_payload_distortion_sweep.md` and `historical_e12_results.json` (2026-10-11T18:26:01Z)
   - *Evidence:* 95% text similarity. The historical files represent an audit baseline cited in `PHASE3_ENTRY_GATE_AUDIT_2026-10-12.md` §6.1.
   - *Recommendation:* ARCHIVE into `docs/experiments/historical/` to preserve audit citations without cluttering the primary experiment ledger.

7. **E13 Multi-Carrier Repeatability:**
   - Active: `docs/experiments/E13_carrier_repeatability_generalization.md` and `e13_results.json`
   - Historical: `docs/experiments/historical_E13_carrier_repeatability_generalization.md` and `historical_e13_results.json`
   - *Evidence:* Historical 90-run run record cited in `PHASE3_ENTRY_GATE_AUDIT_2026-10-12.md` §6.2.
   - *Recommendation:* ARCHIVE into `docs/experiments/historical/`.

8. **E04 Spatial Scaling Triad:**
   - `E04_carrier_spatial_scaling.md` (initial 512x512 exploratory run)
   - `E04_hr_spatial_scaling.md` (native DIV2K HR 512/1024 resolution expansion)
   - `E04_hr_audit_remediation.md` (mathematical reconciliation of 23,010-byte RS chunking)
   - *Status:* All 3 represent genuinely distinct empirical phases of the spatial scaling investigation. KEEP all three, but clarify their sequential progression in `docs/experiments/PHASE2_SYNTHESIS_REPORT.md`.

---

## 4. Documentation Redundancy Map

### 4.1 Target Documentation Hierarchy vs. Current Reality

```text
Target Hierarchy:
1. Root README.md                      -> Primary navigation, quickstart, verified status
2. docs/specifications/                -> Authoritative architecture & acceptance criteria
3. docs/experiments/                   -> Canonical experiment ledgers (E03-E14) & synthesis
4. docs/audits/                        -> Formal gate audits (Entry Gate, Simplicity Audit)
5. docs/reports/                       -> High-level status & completion reports
6. CIAs/                               -> Academic coursework submissions & evaluation logs
```

### 4.2 Document Sprawl Findings & Conflicts

1. **Root Directory Clutter:** The root directory currently contains 5 coursework/status files that violate root simplicity. The root should contain only `README.md`, `pyproject.toml`, `.gitignore`, and standard project configuration.
2. **README.md Desynchronization:**
   - Line 124 states: *"Comprehensive test suite (99 tests)"* — The test suite has grown to **103 passing tests**.
   - Line 74 references only `E03_media_steganography.md` under `docs/experiments/`, completely omitting E04 through E14.
   - Omits `docs/audits/` and `dataset/` structure.
3. **Competing Project Logs:** Sprint logs are stored in duplicate under both `docs/project/` and `CIAs/`. The academic coursework logs should be consolidated in `CIAs/`, while `docs/project/` retains technical metadata (`ORIGINAL_REQUEST.md`, `div2k_hr_split.json`, `mvp0_manifest.json`).
4. **Specification Format Imbalance:** All foundational specifications (`Architecture_Specification.docx`, `Product_Vision_Draft.docx`, `Sprint-0_Implementation_Readiness_Specification_v0.1.docx`, `System_and_Research_Document_Draft.docx`) exist exclusively as binary `.docx` files. This prevents text-based diffing, automated linting, and git grep search.
   - *Note:* These documents represent foundational project contracts and must NOT be deleted. A future non-destructive conversion to markdown is recommended.

---

## 5. Source-Code Simplification Opportunities

The production codebase in `src/` (22 files, 2,345 code lines) is compact and well-engineered. However, specific simplifications and bug fixes were identified:

### 5.1 Unused / Wrapper Code in `src/`

1. **Unreferenced RS Primitives (`src/security/ecc/encoder.py`):**
   - `encode_block(chunk: bytes)` (lines 338-360) and `decode_block(block: bytes)` (lines 362-404) are single-block primitives unreferenced by `SecurityPipeline`, `MediaStegoPipeline`, or `tests/unit/test_ecc.py`.
   - *Assessment:* Although unreferenced, they provide clean, self-contained single-block RS primitives complementing the framed `encode`/`decode` methods.
   - *Recommendation:* KEEP. Add unit tests in `tests/unit/test_ecc.py` rather than deleting valid cryptographic primitives.
2. **Structured Crypto Wrappers (`src/security/crypto/provider.py`):**
   - `encrypt_structured` (lines 293-302) and `decrypt_structured` (lines 304-311) are thin wrappers unpacking and packing `CryptoResult`.
   - *Assessment:* They add minimal overhead (18 lines) and provide typed interfaces.
   - *Recommendation:* KEEP.

### 5.2 Definite Bugs & Path Fragility in `src/`

3. **Broken Dataset Fixture Path in `src/api/server.py`:**
   - Line 34: `COVER_FIXTURE_PATH = os.path.join(REPO_ROOT, "MVP 0 Dataset", "0001x2.png")`
   - *Problem:* `MVP 0 Dataset` is located in `dataset/MVP 0 Dataset/0001x2.png`. As a result, `GET /api/fixtures/defaults` fails to load the default cover image.
   - *Recommendation:* Update path to `os.path.join(REPO_ROOT, "dataset", "MVP 0 Dataset", "0001x2.png")`.

### 5.3 Architectural Layering Validation

- `SecurityPipeline` (`src/core/pipeline.py`, 117 lines) vs. `MediaStegoPipeline` (`src/core/media_pipeline.py`, 261 lines):
  - *Finding:* `SecurityPipeline` encapsulates pure Phase 1 security operations (Serialization -> AES-256-GCM -> Reed-Solomon), while `MediaStegoPipeline` composes `SecurityPipeline` with 2D Haar DWT, capacity gating, and transform-domain carrier embedding.
  - *Verdict:* This separation is architecturally sound and modular. Do NOT merge them.

---

## 6. Scripts & Experiment Infrastructure

### 6.1 Orphaned and Redundant Files in `scripts/`

1. **Tracked Binary Images (`scripts/page_1.png` & `scripts/page_2.png`):**
   - Combined size: **793.1 KB**.
   - Created on 10/1/2026 as rendered preview pages of the 2-page status report PDF.
   - Unreferenced in any Python script or Markdown document.
   - *Recommendation:* REMOVE CANDIDATE. Completely redundant binary artifacts in a script directory.
2. **Multi-Target Output Generation in `scripts/generate_status_docx.py`:**
   - Lines 433-437 write identical files to 4 different target paths in root and `docs/reports/`.
   - *Recommendation:* Refactor `main` to write solely to `docs/reports/PaladinMedia_Project_Status_Report.docx` with an optional `--output` argument.
3. **Multi-Target Output Generation in `scripts/generate_status_pdf.py`:**
   - Lines 506-510 write identical PDFs to 4 paths across root and `docs/reports/`.
   - *Recommendation:* Refactor `main` to write solely to `docs/reports/PaladinMedia_Project_Status_Report_2Page.pdf`.
4. **Machine-Specific Hardcoded Path in `scripts/generate_sprint1_demo_presentation.py`:**
   - Line 24: `sys.path.insert(0, r"E:\pip_packages")`
   - *Problem:* Non-portable absolute path.
   - *Recommendation:* Remove hardcoded path and rely on standard virtual environment packages.

### 6.2 Broken Dataset Paths Across Scripts

5. **`scripts/preflight.py`:**
   - Line 23: `DATASET_DIR = os.path.join(REPO_ROOT, "MVP 0 Dataset")`
   - Running `python scripts/preflight.py` reports `Dataset directory not found: E:\Paladin_media\MVP 0 Dataset`.
   - *Recommendation:* Correct to `os.path.join(REPO_ROOT, "dataset", "MVP 0 Dataset")`.
6. **Experiment Runners (`run_e03.py`, `run_e05.py`, `run_e06.py`, `run_e07.py`, `run_e08.py`, `run_e09.py`, `run_e10.py`):**
   - All 7 runners reference `os.path.join("MVP 0 Dataset", ...)` instead of `os.path.join("dataset", "MVP 0 Dataset", ...)`.
   - *Recommendation:* Update cover path lookup to check `dataset/MVP 0 Dataset` first with fallback to `MVP 0 Dataset`.

---

## 7. Dependencies & Configuration Hygiene

### 7.1 Undeclared Dependencies in `pyproject.toml`

`pyproject.toml` currently declares:
```toml
[project]
dependencies = [
    "cryptography>=41.0.0",
    "reedsolo>=1.7.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
]
```

The audit detected active imports across the repository that are missing from `pyproject.toml`:
1. **API Server (`src/api/`):** `fastapi`, `uvicorn`, `pydantic`.
2. **Automated Testing (`tests/`):** `pytest`.
3. **Experiment Telemetry (`scripts/experiments/`):** `psutil`.
4. **Report & Presentation Generation (`scripts/`):** `python-docx`, `reportlab`, `python-pptx`.

### 7.2 Recommended Dependency Configuration (Optional Groups)

Without altering installed runtime environments, `pyproject.toml` should be updated to declare optional dependency groups:
```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
]
test = [
    "pytest>=7.0.0",
]
reports = [
    "python-docx>=0.8.11",
    "reportlab>=4.0.0",
    "python-pptx>=0.6.21",
    "psutil>=5.9.0",
]
all = [
    "paladinmedia[api,test,reports]",
]
```

---

## 8. Proposed Target Repository Structure

The proposed structure eliminates root clutter, consolidates duplicate reports, archives historical audit files, and maintains all functional, testing, and dataset paths:

```text
Paladin_media/
├── .gitignore
├── pyproject.toml                         # Updated with optional dependency groups
├── README.md                              # Clean root navigation, accurate 103 test count
│
├── src/                                   # Core production package (clean & verified)
│   ├── api/                               # FastAPI server & observability adapter
│   │   ├── event_adapter.py
│   │   ├── server.py                      # Path bug resolved
│   │   └── __init__.py
│   ├── carrier/                           # Wavelet transform & carrier modulation
│   │   ├── capacity.py
│   │   ├── dwt.py
│   │   ├── embedding.py
│   │   ├── extraction.py
│   │   ├── header.py
│   │   ├── mask.py
│   │   └── __init__.py
│   ├── core/                              # Pipeline orchestration & serialization
│   │   ├── exceptions.py
│   │   ├── media_pipeline.py
│   │   ├── pipeline.py
│   │   ├── serialization/
│   │   └── __init__.py
│   └── security/                          # Cryptography & Reed-Solomon ECC
│       ├── crypto/
│       ├── ecc/
│       └── __init__.py
│
├── tests/                                 # 103 passing automated tests
│   ├── conftest.py
│   ├── fixtures/
│   ├── integration/
│   └── unit/
│
├── scripts/                               # Operational scripts & runners
│   ├── experiments/                       # Experiment runners E03-E14
│   │   ├── run_e03.py ... run_e14.py
│   ├── generate_div2k_split.py            # Canonical DIV2K split generator
│   ├── generate_status_docx.py            # Cleaned: writes to docs/reports/ only
│   ├── generate_status_pdf.py             # Cleaned: writes to docs/reports/ only
│   ├── preflight.py                       # Path bug resolved
│   ├── verify_sprint1.py                  # Standalone 24-check acceptance verifier
│   └── __init__.py
│
├── docs/                                  # Canonical documentation hierarchy
│   ├── architecture/                      # (Target for converted markdown specs)
│   ├── specifications/                    # Authoritative specifications
│   │   ├── Architecture_Specification.docx
│   │   ├── Phase3_Acceptance_Criteria.md
│   │   ├── Product_Vision_Draft.docx
│   │   ├── Sprint-0_Implementation_Readiness_Specification_v0.1.docx
│   │   └── System_and_Research_Document_Draft.docx
│   ├── audits/                            # Formal project audits
│   │   ├── PHASE3_ENTRY_GATE_AUDIT_2026-10-12.md
│   │   └── REPOSITORY_SIMPLICITY_AUDIT.md
│   ├── experiments/                       # Canonical experiment ledgers & results
│   │   ├── E03_media_steganography.md ... E14_lossy_channel_degradation.md
│   │   ├── e03_results.json ... e14_results.json
│   │   ├── PHASE2_SYNTHESIS_REPORT.md
│   │   ├── PREFLIGHT_INSPECTION_REPORT.md
│   │   └── historical/                    # Archived audit backups
│   │       ├── historical_E12_payload_distortion_sweep.md
│   │       ├── historical_e12_results.json
│   │       ├── historical_E13_carrier_repeatability_generalization.md
│   │       └── historical_e13_results.json
│   ├── reports/                           # Canonical project reports
│   │   ├── PaladinMedia_Project_Status_Report.docx
│   │   └── PaladinMedia_Project_Status_Report_2Page.pdf
│   ├── testing/
│   │   └── TEST_INFRA.md
│   └── project/                           # Dataset manifests & provenance
│       ├── ORIGINAL_REQUEST.md
│       ├── div2k_hr_split.json
│       └── mvp0_manifest.json
│
├── CIAs/                                  # Academic coursework submissions & logs
│   ├── 23215050_PaladinMedia_WEEK 6.pdf
│   ├── 23215050_PaladinMedia_Week7_Sprint1_Testing_and_Review.pdf
│   ├── PaladinMedia_23215050.pdf          # Relocated from root
│   ├── PaladinMedia_Sprint_and_Scrum_Log.docx
│   └── PaladinMedia_Week7_Sprint1_Testing_and_Review.docx
│
├── submission/                            # CIA-3 Sprint-1 submission package
│   └── CIA3_Sprint1/                      # (Ignored or tracked per human preference)
│
├── frontend/                              # Vite + React Laboratory UI
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
└── dataset/                               # Local/external image data (git-ignored)
    ├── DIV2K/
    └── MVP 0 Dataset/
```

---

## 9. Ranked Cleanup Table

Candidates are classified as:
- **KEEP:** Required, canonical, unique, or necessary for reproducibility.
- **CONSOLIDATE:** Content should be merged into a single clearly identified canonical destination.
- **ARCHIVE:** Historical material retained but moved out of primary navigation path.
- **REMOVE CANDIDATE:** Strong evidence indicates the file is redundant, duplicate, generated, or unused.

| Priority & Rank | Current File Path | Category | Purpose | Evidence of Redundancy / Issue | Proposed Action & Destination | Expected Benefit | Risk Level | Validation Required | Human Approval Needed? |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **P1-01** | `scripts/page_1.png` | REMOVE CANDIDATE | Rendered preview of page 1 of 2-page report | 361.6 KB binary file, unreferenced anywhere in code or docs, tracked in Git | Delete from `scripts/` | Removes 361 KB tracked binary clutter | **Low** | Verify test suite and scripts unaffected | No |
| **P1-02** | `scripts/page_2.png` | REMOVE CANDIDATE | Rendered preview of page 2 of 2-page report | 431.5 KB binary file, unreferenced anywhere in code or docs, tracked in Git | Delete from `scripts/` | Removes 431 KB tracked binary clutter | **Low** | Verify test suite and scripts unaffected | No |
| **P1-03** | `PaladinMedia_Project_Status_Report_IMRaD.docx` (root) | REMOVE CANDIDATE | Root status docx | Exact SHA-256 match (`8802dfe05cb4`) to `docs/reports/PaladinMedia_Project_Status_Report.docx` | Delete from root; canonical copy in `docs/reports/` | Cleans root directory | **Low** | Verify `docs/reports/` canonical copy intact | No |
| **P1-04** | `docs/reports/PaladinMedia_Project_Status_Report_IMRaD.docx` | REMOVE CANDIDATE | Duplicate report name in `docs/reports/` | Exact SHA-256 match (`8802dfe05cb4`) to sibling `PaladinMedia_Project_Status_Report.docx` | Delete redundant `_IMRaD` named copy | Eliminates duplicate naming | **Low** | Verify sibling docx intact | No |
| **P1-05** | `PaladinMedia_Project_Status_Report_2Page.pdf` (root) | REMOVE CANDIDATE | 2-page status PDF in root | Identical 10,708 B size and visual content to copy in `docs/reports/` | Delete from root; canonical copy in `docs/reports/` | Cleans root directory | **Low** | Verify `docs/reports/` PDF opens | No |
| **P1-06** | `PaladinMedia_Project_Status_and_Completion_Report_2Page.pdf` (root) | REMOVE CANDIDATE | Duplicate-named 2-page PDF in root | Identical 10,708 B size and visual content to copy in `docs/reports/` | Delete from root | Cleans root directory | **Low** | Verify `docs/reports/` PDF opens | No |
| **P1-07** | `docs/reports/PaladinMedia_Project_Status_and_Completion_Report_2Page.pdf` | REMOVE CANDIDATE | Duplicate-named 2-page PDF in `docs/reports/` | Identical 10,708 B size and visual content to sibling `PaladinMedia_Project_Status_Report_2Page.pdf` | Delete redundant alternate name | Eliminates duplicate naming | **Low** | Verify sibling PDF opens | No |
| **P1-08** | `docs/project/PaladinMedia_Sprint_and_Scrum_Log.docx` | REMOVE CANDIDATE | Sprint log in `docs/project/` | Exact SHA-256 match (`98cd6f59542b`) to `CIAs/PaladinMedia_Sprint_and_Scrum_Log.docx` | Delete from `docs/project/`; canonical copy in `CIAs/` | Eliminates cross-directory duplication | **Low** | Verify `CIAs/` copy intact | No |
| **P1-09** | `docs/project/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx` | REMOVE CANDIDATE | Week 7 test review docx in `docs/project/` | Exact SHA-256 match (`bed44df13fa1`) to `CIAs/PaladinMedia_Week7_Sprint1_Testing_and_Review.docx` | Delete from `docs/project/`; canonical copy in `CIAs/` | Eliminates cross-directory duplication | **Low** | Verify `CIAs/` copy intact | No |
| **P1-10** | `PaladinMedia_Project_Status_Report.docx` (root) | CONSOLIDATE | Root status report with diagram (109.8 KB) | Duplicates status report in `docs/reports/`, but contains an embedded architecture diagram | Overwrite `docs/reports/PaladinMedia_Project_Status_Report.docx` with this diagram version; remove root file | Preserves visual diagram in single canonical report | **Low** | Verify diagram renders in Word | **Yes** (Human preference on diagram vs text) |
| **P1-11** | `PaladinMedia_23215050.pdf` (root) | ARCHIVE | BCA411 progress report submission for student 23215050 | Coursework submission PDF sitting in root; belongs with other student submissions | Move to `CIAs/PaladinMedia_23215050.pdf` | Cleans root; keeps coursework records grouped | **Low** | Verify PDF intact in `CIAs/` | **Yes** (Confirm placement in `CIAs/`) |
| **P2-01** | `docs/experiments/historical_E12_payload_distortion_sweep.md` | ARCHIVE | Pre-remediation E12 experiment report | Superseded by canonical `E12_payload_distortion_sweep.md`; cited in Entry Gate audit | Move to `docs/experiments/historical/` | Unclutters active experiment ledger | **Low** | Verify link in Entry Gate audit updated | No |
| **P2-02** | `docs/experiments/historical_e12_results.json` | ARCHIVE | Pre-remediation E12 raw JSON results | Superseded by `e12_results.json`; cited in Entry Gate audit §6.1 | Move to `docs/experiments/historical/` | Preserves scientific evidence cleanly | **Low** | Verify audit reference updated | No |
| **P2-03** | `docs/experiments/historical_E13_carrier_repeatability_generalization.md` | ARCHIVE | Pre-remediation E13 experiment report | Superseded by canonical `E13_carrier_repeatability_generalization.md`; cited in Entry Gate audit | Move to `docs/experiments/historical/` | Unclutters active experiment ledger | **Low** | Verify audit reference updated | No |
| **P2-04** | `docs/experiments/historical_e13_results.json` | ARCHIVE | Pre-remediation E13 raw JSON results | Superseded by `e13_results.json`; cited in Entry Gate audit §6.2 | Move to `docs/experiments/historical/` | Preserves scientific evidence cleanly | **Low** | Verify audit reference updated | No |
| **P2-05** | `scripts/generate_status_docx.py` (lines 433-437) | CONSOLIDATE | Status report generator script | Hardcodes writing to 4 distinct paths simultaneously | Refactor to write to single canonical `docs/reports/PaladinMedia_Project_Status_Report.docx` | Stops regenerative file duplication | **Low** | Run script and verify output | No |
| **P2-06** | `scripts/generate_status_pdf.py` (lines 506-510) | CONSOLIDATE | Status PDF generator script | Hardcodes writing to 4 distinct paths simultaneously | Refactor to write to single canonical `docs/reports/PaladinMedia_Project_Status_Report_2Page.pdf` | Stops regenerative file duplication | **Low** | Run script and verify output | No |
| **P2-07** | `scripts/generate_week7_doc.py` (lines 365-372) | CONSOLIDATE | Week 7 doc generator script | Saves to both root and `docs/project/` | Refactor to save to `CIAs/` only | Stops root file regeneration | **Low** | Run script and verify output | No |
| **P2-08** | `scripts/preflight.py` (line 23) | BUG FIX | Preflight verification script | Hardcodes `MVP 0 Dataset` at root, failing carrier checks | Update to `os.path.join(REPO_ROOT, "dataset", "MVP 0 Dataset")` | Restores functional dataset verification | **Low** | Run `python scripts/preflight.py` | No |
| **P2-09** | `src/api/server.py` (line 34) | BUG FIX | FastAPI backend server | Hardcodes `MVP 0 Dataset` at root, breaking `/api/fixtures/defaults` | Update to `os.path.join(REPO_ROOT, "dataset", "MVP 0 Dataset", "0001x2.png")` | Fixes cover fixture loading in UI | **Low** | Test endpoint with `pytest` / `requests` | No |
| **P2-10** | `scripts/experiments/run_e03.py` .. `run_e10.py` | BUG FIX | 7 experiment runner scripts | Hardcodes `MVP 0 Dataset/` relative path | Update cover path resolution to `dataset/MVP 0 Dataset/` | Enables experiment execution from root | **Low** | Verify paths resolve | No |
| **P2-11** | `scripts/generate_sprint1_demo_presentation.py` (line 24) | BUG FIX | Demo PPTX generator script | Machine-specific `sys.path.insert(0, r"E:\pip_packages")` | Remove hardcoded path; rely on environment | Improves portability | **Low** | Run script in active venv | No |
| **P2-12** | `pyproject.toml` | CONSOLIDATE | Dependency manifest | Omits `fastapi`, `uvicorn`, `pydantic`, `pytest`, `reportlab`, `docx`, `pptx`, `psutil` | Add `[project.optional-dependencies]` groups | Clarifies runtime and dev requirements | **Low** | Verify `pip install -e .[test]` | No |
| **P2-13** | `README.md` | CONSOLIDATE | Root navigation and overview | Stale test count (99 vs 103), missing E04-E14 experiment register | Update test count and directory navigation | Restores single source of truth | **Low** | Inspect rendered markdown | No |
| **P3-01** | `submission/CIA3_Sprint1/preview/*.png` (13 files) | ARCHIVE / GITIGNORE | Rendered PNG previews of submission report and slides | 2.52 MB of untracked binary preview PNGs | Add `submission/*/preview/` to `.gitignore` or archive | Prevents accidental binary commit | **Medium** | Verify git status | **Yes** (Human decision on submission tracking) |

---

## 10. Files That Must Be Protected

The following files represent verified scientific evidence, frozen architectural contracts, core cryptographic engines, or active user edits and **must never be deleted, renamed, or modified during cleanup**:

### 10.1 Active In-Progress Tasks (Excluded & Protected)
- `docs/audits/PHASE3_ENTRY_GATE_AUDIT_2026-10-12.md`: Active E14 limitations revisions.
- `docs/experiments/E14_lossy_channel_degradation.md`: Active E14 empirical findings and channel qualification.
- `docs/experiments/PHASE2_SYNTHESIS_REPORT.md`: Active Phase 2 synthesis updates.
- `docs/specifications/Phase3_Acceptance_Criteria.md`: Active gate 3 channel robustness definition.

### 10.2 Core Algorithmic & Security Invariants
- `src/carrier/dwt.py`: Haar DWT/IDWT 2D transform implementation.
- `src/carrier/embedding.py`: Baseline transform-domain embedder (2*HH parity modulation).
- `src/carrier/extraction.py`: Parity-bit carrier extractor.
- `src/carrier/header.py`: 16-byte fixed canonical carrier header codec (`PLDH`).
- `src/carrier/mask.py`: Deterministic PRNG position generator.
- `src/carrier/capacity.py`: Capacity hard gate and allocation bounds.
- `src/core/pipeline.py`: `SecurityPipeline` (Serialize -> AES-GCM -> Reed-Solomon).
- `src/core/media_pipeline.py`: `MediaStegoPipeline` end-to-end orchestration.
- `src/core/serialization/serializer.py`: Canonical payload wire framing (`PLDS`, CRC-32).
- `src/core/exceptions.py`: System exception hierarchy.
- `src/security/crypto/provider.py`: Authenticated AES-256-GCM provider.
- `src/security/ecc/encoder.py`: Reed-Solomon GF(2^8) encoder/decoder.

### 10.3 Scientific Reproducibility & Experiment Ledgers
- `docs/experiments/e03_results.json` through `e14_results.json`: Canonical empirical result records (12 JSON files).
- `docs/experiments/e04_hr_results.json` & `e04_hr_audit_results.json`: HR spatial scaling result records.
- `docs/experiments/E03_media_steganography.md` through `E14_lossy_channel_degradation.md`: Primary experiment reports.
- `docs/project/div2k_hr_split.json` (282.8 KB): Cryptographically frozen train (600) / val (100) / test (100) DIV2K HR split.
- `docs/project/mvp0_manifest.json` (15.7 KB): Carrier image checksum and dimension ledger.
- `docs/project/ORIGINAL_REQUEST.md`: Provenance of original architectural mandate.

### 10.4 Test Suite & Validation Infrastructure
- `tests/` (all 19 files, 103 tests): Entire unit and integration test suite.
- `tests/fixtures/vectors.py`: Official NIST SP 800-38D KAT vectors.
- `tests/fixtures/secret_small.png`: Ground-truth test image fixture.
- `scripts/verify_sprint1.py`: Standalone 24-check acceptance verifier.

---

## 11. Risks and Validation Plan

### 11.1 Identified Risks & Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
| :--- | :---: | :--- |
| **Breaking Import Paths:** Deleting or moving Python modules or scripts breaks external runners or tests. | High | No modules in `src/` or `tests/` are proposed for deletion or relocation. All proposed deletions are redundant reports, duplicate docs, and orphaned PNGs. |
| **Broken Audit Links:** Moving `historical_e12*` and `historical_e13*` breaks citations in `PHASE3_ENTRY_GATE_AUDIT_2026-10-12.md`. | Medium | Ensure relative Markdown links in the audit document are updated to point to `docs/experiments/historical/` before moving files. |
| **Loss of Coursework Records:** Removing DOCX/PDF files from root or `docs/project/` deletes required academic submission records. | Medium | All unique academic records are consolidated into `CIAs/` (e.g. `PaladinMedia_23215050.pdf`). No academic submission is permanently deleted without human approval. |
| **Regenerative Duplication:** Running generator scripts recreates duplicate files in root and `docs/reports/`. | Medium | Refactor generator scripts (`generate_status_docx.py`, `generate_status_pdf.py`, `generate_week7_doc.py`) prior to deleting the generated duplicates. |

### 11.2 Post-Cleanup Verification Checklist

After any authorized cleanup is executed, the following non-destructive verification sequence must be completed:
1. **Run Automated Test Suite:**
   ```bash
   pytest -v
   ```
   *Pass Criterion:* Exactly 103 passed in < 3.0s with 0 failures or warnings.
2. **Run Programmatic Sprint 1 Verifier:**
   ```bash
   python scripts/verify_sprint1.py
   ```
   *Pass Criterion:* All 24 checks pass.
3. **Run Dataset Preflight:**
   ```bash
   python scripts/preflight.py
   ```
   *Pass Criterion:* All 4 specifications validated, all 61 MVP 0 dataset carriers inspected with checksums matching `docs/project/mvp0_manifest.json`.
4. **Verify Git Working Tree:**
   ```bash
   git status
   ```
   *Pass Criterion:* No untracked binary clutter in root or `scripts/`.

---

## 12. Estimated Reductions

*Note: All figures below are engineering estimates until authorized changes are executed and measured.*

| Category | Current State | Target Post-Cleanup | Estimated Reduction |
| :--- | :---: | :---: | :---: |
| **Root Directory File Count** | 8 files | 3 files (`README.md`, `pyproject.toml`, `.gitignore`) | **-5 files (-62.5%)** |
| **Tracked Binary Clutter** | ~2.1 MB across root, scripts, docs | ~0.7 MB (canonical reports & specs) | **-1.43 MB (-68.1%)** |
| **Duplicate DOCX / PDF Documents** | 9 redundant copies | 0 redundant copies | **-9 duplicate files (-100%)** |
| **Orphaned Script PNG Renders** | 2 files (793.1 KB) | 0 files | **-2 files (-793 KB)** |
| **Active Experiment Ledger Files** | 22 files in `docs/experiments/` | 18 active + 4 archived | **4 files archived out of primary path** |
| **Generator Script Output Fan-Out** | 8 hardcoded write targets | 2 single canonical targets | **-6 redundant write paths** |
| **Total Tracked Repo Files** | 140 files | ~129 files | **-11 files (-7.8%)** |
| **Untracked Clutter (Submission Previews)** | 13 preview PNGs (2.52 MB) | Git-ignored or archived | **-2.52 MB untracked binary clutter** |

---

## 13. Outstanding Conflicts & Items Requiring Human Decisions

The following architectural and structural items require explicit user approval before execution in a subsequent task:

1. **Root Status Report Diagram Retention (`PaladinMedia_Project_Status_Report.docx`):**
   - The 109 KB root DOCX includes an embedded architecture diagram, whereas the 44 KB version in `docs/reports/` is text-only.
   - *Decision:* Should the 109 KB diagram version overwrite the 44 KB version as the single canonical status document in `docs/reports/PaladinMedia_Project_Status_Report.docx`? *(Recommended: Yes)*.
2. **Coursework Deliverables Directory (`CIAs/` vs. `submission/`):**
   - The repository currently has both `CIAs/` (tracked) and `submission/CIA3_Sprint1/` (untracked).
   - *Decision:* Should `PaladinMedia_23215050.pdf` (from root) be relocated into `CIAs/`, and should `submission/CIA3_Sprint1/` remain local/untracked or be added to `.gitignore`? *(Recommended: Relocate root PDF to `CIAs/`, add `submission/*/preview/` to `.gitignore`)*.
3. **Specification Format Migration (`.docx` to `.md`):**
   - The four core specifications in `docs/specifications/` are in binary `.docx` format.
   - *Decision:* Should these specifications be converted to Markdown in a future task to enable Git text diffing and grep search, while retaining the `.docx` originals in an archive? *(Recommended: Yes, in a subsequent dedicated documentation task)*.
4. **Dataset Location Standardization:**
   - Datasets currently reside in `dataset/` (which is correctly ignored by Git). However, scripts and server code have legacy references to `MVP 0 Dataset/` at root.
   - *Decision:* Confirm that all code and scripts should be updated to point to `dataset/MVP 0 Dataset/` rather than creating symlinks or moving datasets back to root. *(Recommended: Update code paths to `dataset/MVP 0 Dataset/`)*.

---

**Audit Conclusion:** The PaladinMedia repository is functionally sound with verified cryptographic and signal processing implementations and a 100% passing test suite. The identified cleanup plan provides a clear, zero-risk pathway to eliminate ~1.43 MB of tracked binary duplicates and orphaned assets, resolve path bugs, and establish a clean, uncluttered root directory without altering a single line of core algorithmic code or compromising scientific traceability.
