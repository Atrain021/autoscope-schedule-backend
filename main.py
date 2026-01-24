# main.py - AutoScope Schedule Backend (Production Ready) 

import os
import re
import gc
import json
import uuid
import base64
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field


import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI
from typing import Optional, Dict, List

import re
import json
import base64
from typing import Dict, Optional

# -----------------------------
# Helper for Drawing Type - need to expand for all sets
# -----------------------------

ARCH_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "CODE_LIFE_SAFETY",
    "SITE_PLAN",
    "FLOOR_PLAN",
    "ENLARGED_PLAN",
    "REFLECTED_CEILING_PLAN",
    "ROOF_PLAN",
    "LIFE_SAFETY_EGRESS_PLAN",
    "ELEVATIONS",
    "SECTIONS",
    "DETAILS",
    "WALL_TYPES_ASSEMBLIES",
    "DOOR_WINDOW_SCHEDULES",
    "ROOM_FINISH_SCHEDULES",
    "OTHER_SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

ARCH_ALLOWED_SET = set(ARCH_SHEET_TYPES)

def _clean_json_text(text: str) -> str:
    """Remove ```json fences if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r'^```[a-zA-Z]*\n', '', t)
        t = re.sub(r'\n```$', '', t)
        t = t.strip()
    return t


def _normalize_arch_type(value: str) -> str:
    v = (value or "").strip().upper()
    v = v.replace(" ", "_").replace("-", "_")
    # common aliases
    aliases = {
        "FINISH_SCHEDULE": "ROOM_FINISH_SCHEDULES",
        "FINISH_SCHEDULES": "ROOM_FINISH_SCHEDULES",
        "ROOM_FINISH_SCHEDULE": "ROOM_FINISH_SCHEDULES",
        "RFS": "ROOM_FINISH_SCHEDULES",
        "DOOR_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
        "WINDOW_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
        "DOOR_WINDOW_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
        "GENERAL_NOTES": "SHEET_INDEX_GENERAL_NOTES",
        "SHEET_INDEX": "SHEET_INDEX_GENERAL_NOTES",
        "CODE": "CODE_LIFE_SAFETY",
        "LIFE_SAFETY": "CODE_LIFE_SAFETY",
        "EGRESS": "LIFE_SAFETY_EGRESS_PLAN",
        "PLAN": "FLOOR_PLAN",
        "DETAIL": "DETAILS",
        "SECTION": "SECTIONS",
        "ELEVATION": "ELEVATIONS",
        "WALL_TYPES": "WALL_TYPES_ASSEMBLIES",
        "ASSEMBLY_TYPES": "WALL_TYPES_ASSEMBLIES",
    }
    if v in aliases:
        v = aliases[v]
    return v if v in ARCH_ALLOWED_SET else "OTHER"


def _override_by_title(sheet_title: str, sheet_id: str, base_type: str) -> str:
    """
    Deterministic overrides based on sheet title / id.
    This is where we eliminate obvious mislabels (egress plans, schedules, etc.).
    """
    title = (sheet_title or "").strip().upper()
    sid = (sheet_id or "").strip().upper()

    # If title is blank, keep base_type
    if not title and not sid:
        return base_type

    # Hard keyword buckets (highest confidence)
    if any(k in title for k in ["COVER SHEET"]):
        return "COVER_SHEET"

    if any(k in title for k in ["SHEET INDEX", "GENERAL NOTES", "ABBREVIATIONS", "LEGEND"]):
        return "SHEET_INDEX_GENERAL_NOTES"

    # Life safety / egress plans
    if any(k in title for k in ["EGRESS", "EXIT", "LIFE SAFETY", "LIFE-SAFETY", "EVACUATION"]):
        return "LIFE_SAFETY_EGRESS_PLAN"

    # Code / analysis pages (not necessarily “plans”)
    if any(k in title for k in ["CODE", "OCCUPANCY", "UNIT MIX", "FIRE RATING", "ACCESSIBILITY", "ADA", "IBC", "NFPA"]):
        return "CODE_LIFE_SAFETY"

    # Finish schedules
    if any(k in title for k in ["FINISH SCHEDULE", "ROOM FINISH", "FINISH LEGEND", "FINISH PLAN LEGEND"]):
        return "ROOM_FINISH_SCHEDULES"

    # Door/window schedules
    if any(k in title for k in ["DOOR SCHEDULE", "WINDOW SCHEDULE", "STOREFRONT SCHEDULE", "FRAME SCHEDULE"]):
        return "DOOR_WINDOW_SCHEDULES"

    # RCP
    if any(k in title for k in ["REFLECTED CEILING", "RCP", "CEILING PLAN"]):
        return "REFLECTED_CEILING_PLAN"

    # Roof plan
    if any(k in title for k in ["ROOF PLAN", "PENTHOUSE ROOF", "ROOF"]):
        # avoid misfiring on "roof details"
        if "DETAIL" not in title and "SECTION" not in title:
            return "ROOF_PLAN"

    # Site plan
    if any(k in title for k in ["SITE PLAN", "GRADING", "SITE", "UTILITY PLAN", "EROSION", "SEDIMENT"]):
        return "SITE_PLAN"

    # Elevations / sections / details
    if "ELEVATION" in title or title.startswith("ELEVATIONS"):
        return "ELEVATIONS"

    if "SECTION" in title or title.startswith("SECTIONS"):
        return "SECTIONS"

    if "DETAIL" in title or title.startswith("DETAILS"):
        return "DETAILS"

    # Wall types / assemblies
    if any(k in title for k in ["WALL TYPE", "PARTITION TYPE", "ASSEMBLY", "TYPICAL WALL"]):
        return "WALL_TYPES_ASSEMBLIES"

    # Enlarged plans (toilet rooms, kitchens, etc.)
    if "ENLARGED" in title or any(k in title for k in ["TOILET ROOM PLAN", "KITCHEN PLAN", "BATHROOM PLAN"]):
        return "ENLARGED_PLAN"

    # If the model said "FLOOR_PLAN" but title screams it's not, downgrade:
    if base_type == "FLOOR_PLAN":
        if any(k in title for k in ["SITE", "CODE", "SCHEDULE", "DETAIL", "SECTION", "ELEVATION", "NOTES", "INDEX"]):
            # pick the closest based on title; otherwise OTHER
            # (most are handled above, this is just safety)
            return "OTHER"

    return base_type


# -----------------------------
# Configuration
# -----------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(
    title="AutoScope Schedule Service",
    description="AI-powered construction schedule extraction",
    version="2.0.0"
)

class PageIndexPage(BaseModel):
    page_number: int
    sheet_identifier: Optional[str] = None
    sheet_title: Optional[str] = None
    classification: Optional[str] = None  # current: FINISH_SCHEDULE / FLOOR_PLAN / OTHER


class PageIndexPayload(BaseModel):
    filename: str
    pdf_url: Optional[str] = None
    taxonomy_version: str = "v1-coarse"
    pages: List[PageIndexPage] = Field(default_factory=list)


def _page_index_qc_flags(pages: List[PageIndexPage]) -> Dict[str, Any]:
    # Basic QC flags that help you quickly spot issues.
    missing_sheet_id_pages = [p.page_number for p in pages if not (p.sheet_identifier and p.sheet_identifier.strip())]
    low_info_title_pages = [p.page_number for p in pages if not (p.sheet_title and p.sheet_title.strip())]

    # Duplicate sheet IDs (common when OCR/classifier mistakes happen)
    sheet_map: Dict[str, List[int]] = {}
    for p in pages:
        sid = (p.sheet_identifier or "").strip()
        if not sid:
            continue
        sheet_map.setdefault(sid, []).append(p.page_number)
    duplicates = {sid: nums for sid, nums in sheet_map.items() if len(nums) > 1}

    # Classification counts
    counts: Dict[str, int] = {}
    for p in pages:
        c = (p.classification or "UNKNOWN").strip()
        counts[c] = counts.get(c, 0) + 1

    return {
        "page_count": len(pages),
        "classification_counts": counts,
        "missing_sheet_identifier_pages": missing_sheet_id_pages[:50],  # cap so response doesn't blow up
        "missing_sheet_title_pages": low_info_title_pages[:50],
        "duplicate_sheet_identifiers": duplicates,  # may be large but usually small
    }

@app.post("/page-index/qc-summary")
def page_index_qc_summary(payload: PageIndexPayload) -> Dict[str, Any]:
    flags = _page_index_qc_flags(payload.pages)

    # A simple "success criteria" you can use in UI:
    # - no missing sheet IDs
    # - page_count matches expected
    ok = (len(flags["missing_sheet_identifier_pages"]) == 0)

    return {
        "ok": ok,
        "filename": payload.filename,
        "taxonomy_version": payload.taxonomy_version,
        "stats": flags,
    }


from fastapi import Request
from fastapi.responses import Response

@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    return Response(status_code=200)


# Serve uploaded PDFs publicly so Base44 can pass a URL into InvokeLLM
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

from fastapi.middleware.cors import CORSMiddleware

# CORS for Base44 + local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https://.*\.base44\.app$",
    allow_credentials=False,  # IMPORTANT (no cookies needed)
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class ExtractByTagsRequest(BaseModel):
    filename: str
    page_number: int
    tags: List[str]

# -----------------------------
# Core Extraction Logic
# -----------------------------
class ScheduleExtractor:
    """Handles conversion of PDF pages to images and GPT-4V extraction"""
    
    def __init__(self):
        self.client = client
    
    import fitz  # PyMuPDF
    import gc

    def pdf_page_to_image(
        self,
        pdf_path: str,
        page_number: int,
        dpi: int = 300,
        *,
        purpose: str = "extract",     # "extract" or "classify"
        jpeg_quality: int = 70
    ) -> bytes:
        """
        Convert a PDF page to an image.

        - purpose="extract": high-quality PNG (default) for reading small schedule text
        - purpose="classify": low-memory grayscale JPEG for page classification

        Returns: image bytes
        """
        doc = None
        try:
            doc = fitz.open(pdf_path)

            if page_number < 0 or page_number >= len(doc):
                raise ValueError(f"Invalid page number {page_number}. PDF has {len(doc)} pages.")

            page = doc[page_number]

            # Classification should NOT render at high DPI (memory killer)
            if purpose == "classify":
                dpi = min(int(dpi), 72)  # hard cap for safety (60–72 is plenty)

            # Render
            mat = fitz.Matrix(dpi / 72, dpi / 72)

            if purpose == "classify":
                # Grayscale drastically reduces memory; JPEG reduces bytes further
                pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
                img_bytes = pix.tobytes("jpeg", jpg_quality=int(jpeg_quality))
            else:
                # Extraction mode: keep crisp PNG
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")

            # Free big objects ASAP
            del pix
            del page
            gc.collect()

            return img_bytes

        except Exception as e:
            raise Exception(f"PDF to image conversion failed: {str(e)}")
        finally:
            try:
                if doc is not None:
                    doc.close()
            except Exception:
                pass

    
    def extract_schedule_tags(self, image_bytes: bytes, requested_tags: List[str]) -> Dict[str, str]:
        """
        Two-pass extraction to prevent hallucinations:
        Pass 1: Read all visible text on page
        Pass 2: Extract tags using only text from Pass 1
        """
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # PASS 1: Force GPT to read what's actually there
        verification_prompt = """Look at this construction finish schedule image.

    List ALL manufacturer names (MFR) you can see on this page.

    Return ONLY a JSON array of manufacturer names visible in the image:

    ["MANUFACTURER1", "MANUFACTURER2", "MANUFACTURER3"]

    Do NOT guess. Do NOT use training data. Read ONLY what is visible in THIS image."""

        try:
            # Pass 1: Get visible manufacturers
            verify_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": verification_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.0,
            )

            verify_text = verify_response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if verify_text.startswith("```"):
                verify_text = re.sub(r"^```[a-zA-Z]*\n", "", verify_text)
                verify_text = re.sub(r"\n```$", "", verify_text)
                verify_text = verify_text.strip()

            visible_manufacturers = json.loads(verify_text)
            mfr_list = ", ".join(visible_manufacturers)

            # PASS 2: Extract using only verified manufacturers
            tag_list = ", ".join(requested_tags)

            extraction_prompt = f"""You are extracting finish schedule data.
    VERIFIED MANUFACTURERS visible on this page: {mfr_list}
    REQUESTED TAGS: {tag_list}
    For each requested tag:
    1. Locate the tag marker (circle with tag name)
    2. Read ALL text associated with that tag
    3. Include: MFR, PROD, PATT, COLOR, SIZE, LOC, HEIGHT, etc.
    CRITICAL RULES:
    • Use ONLY manufacturers from the verified list above
    • If a tag uses a manufacturer NOT in the list, return "ERROR: Unverified manufacturer"
    • If tag shows DELETED, return "DELETED"
    • If tag not found, return empty string ""
    • Read text EXACTLY as shown
    Return JSON:
    STARTJSON {{"TAG": "complete description text"}} ENDJSON"""

            # Pass 2: Extract with constraint
            extract_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )

            result_text = extract_response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if result_text.startswith("```"):
                result_text = re.sub(r"^```[a-zA-Z]*\n", "", result_text)
                result_text = re.sub(r"\n```$", "", result_text)
                result_text = result_text.strip()

            # Strip STARTJSON / ENDJSON markers if the model echoed them
            result_text = result_text.replace("STARTJSON", "")
            result_text = result_text.replace("ENDJSON", "")
            result_text = result_text.strip()

            result = json.loads(result_text)
            return result

        except json.JSONDecodeError as e:
            raise Exception(
                f"JSON parse failed: {str(e)}\nResponse: {result_text[:300]}"
            )
        except Exception as e:
            raise Exception(f"Extraction failed: {str(e)}")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AutoScope Schedule Service",
        "version": "2.0.0",
        "status": "operational"
    }
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AutoScope Schedule Extractor"
    }
# -----------------------------
# /upload PDF endpoint
# -----------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and store it with unique filename.
    Base44 calls this first to upload the PDF.

    Returns: {"filename": "unique_id.pdf"}
    """
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.pdf"
        filepath = UPLOAD_DIR / filename
        
        # Save uploaded file
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        pdf_url = f"{base_url}/uploads/{filename}" if base_url else f"/uploads/{filename}"


        return JSONResponse({
            "filename": filename,
            "pdf_url": pdf_url,
            "size": len(content),
            "original_filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
# -----------------------------
# /extract-finish-schedule-by-tags
# -----------------------------
@app.post("/extract-finish-schedule-by-tags")
async def extract_finish_schedule_by_tags(request: ExtractByTagsRequest):
    """
    Extract specific tags from a finish schedule page.
    This is the main endpoint Base44 uses.

    Request body:
    {
        "filename": "uuid.pdf",
        "page_number": 0,
        "tags": ["AC-01", "PT-02F", "WD-01"]
    }

    Returns:
    {
        "blocks": [
            {"tag": "AC-01", "block_text": "MFR: Armstrong..."},
            {"tag": "PT-02F", "block_text": "MFR: Sherwin Williams..."}
        ],
        "num_blocks": 2
    }
    """
    try:
        filepath = UPLOAD_DIR / request.filename
        if not filepath.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PDF file not found: {request.filename}"
            )
        
        extractor = ScheduleExtractor()
        
        image_bytes = extractor.pdf_page_to_image(
            str(filepath),
            request.page_number
        )
        
        tag_results = extractor.extract_schedule_tags(
            image_bytes,
            request.tags
        )
        
        blocks = []
        for tag in request.tags:
            normalized_tag = normalize_tag(tag)
            
            block_text = ""
            for result_tag, text in tag_results.items():
                if normalize_tag(result_tag) == normalized_tag:
                    block_text = text
                    break
            
            blocks.append({
                "tag": normalized_tag,
                "block_text": block_text
            })
        
        return JSONResponse({
            "blocks": blocks,
            "num_blocks": len(blocks),
            "page_number": request.page_number
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )

class ClassifyPdfRequest(BaseModel):
    filename: str
    start_page: int = 1          # 1-based inclusive
    end_page: Optional[int] = None  # 1-based inclusive (None = through end)
    dpi: int = 72               # lower DPI is faster for classification

def _clean_json_text(text: str) -> str:
    """Remove ```json fences if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r'^```[a-zA-Z]*\n', '', t)
        t = re.sub(r'\n```$', '', t)
        t = t.strip()
    return t


def _classify_single_page(image_bytes: bytes) -> Dict[str, str]:
    """
    Classify a single PDF page image into an Architectural sheet type.
    Also attempt to read sheet_identifier and sheet_title if visible.
    Then apply deterministic overrides based on title/id.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    type_list = ", ".join(ARCH_SHEET_TYPES)

    prompt = f"""You are looking at ONE page from a construction drawing PDF (Architectural set).

Your job:
1) Visually classify this page into exactly ONE of these types:
{type_list}

2) If visible, extract:
- sheet_identifier (examples: A101, A0.01, AS-201, etc.)
- sheet_title (examples: "LEVEL 2 FLOOR PLAN", "DOOR SCHEDULE", etc.)

Return ONLY valid JSON in exactly this format:
{{
  "classification": "FLOOR_PLAN",
  "sheet_identifier": "A101",
  "sheet_title": "LEVEL 1 FLOOR PLAN"
}}

Rules:
- Do NOT output markdown.
- If sheet_identifier or sheet_title are not clearly visible, return "" for them.
- Choose the most specific type available (e.g., if it’s an EGRESS plan, choose LIFE_SAFETY_EGRESS_PLAN, not FLOOR_PLAN).
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=400,
        temperature=0.0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    cleaned = _clean_json_text(raw)
    data = json.loads(cleaned)

    sheet_id = (data.get("sheet_identifier") or "").strip()
    sheet_title = (data.get("sheet_title") or "").strip()

    base_type = _normalize_arch_type(data.get("classification") or "OTHER")
    final_type = _override_by_title(sheet_title, sheet_id, base_type)

    return {
        "classification": final_type,
        "sheet_identifier": sheet_id,
        "sheet_title": sheet_title,
    }



@app.post("/classify-pdf")
async def classify_pdf(request: ClassifyPdfRequest):
    """
    Classify pages in an uploaded PDF (stored in /uploads) WITHOUT using Base44 InvokeLLM.
    This is specifically to handle PDFs >10MB that Base44 rejects.
    """
    try:
        filepath = UPLOAD_DIR / request.filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {request.filename}")

        doc = fitz.open(str(filepath))
        total_pages = len(doc)

        start = max(1, int(request.start_page))

        # ✅ NEW: if Base44 asks beyond the last page, return empty list (don't error)
        if start > total_pages:
            doc.close()
            return JSONResponse({
                "filename": request.filename,
                "total_pages": total_pages,
                "pages": []
            })

        end = int(request.end_page) if request.end_page else total_pages
        end = min(end, total_pages)

        if start > end:
            doc.close()
            raise HTTPException(status_code=422, detail="start_page must be <= end_page")


        extractor = ScheduleExtractor()
        pages_out = []

        for page_number in range(start, end + 1):
            page_index = page_number - 1  # 0-based
            image_bytes = extractor.pdf_page_to_image(
                str(filepath),
                page_index,
                dpi=int(request.dpi),
                purpose="classify"
            )


            page_info = _classify_single_page(image_bytes)
            pages_out.append({
                "page_number": page_number,  # keep 1-based in output
                "classification": page_info["classification"],
                "sheet_identifier": page_info["sheet_identifier"],
                "sheet_title": page_info["sheet_title"],
            })

        doc.close()

        return JSONResponse({
            "filename": request.filename,
            "total_pages": total_pages,
            "pages": pages_out
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# -----------------------------
# Utility Functions
# -----------------------------
def normalize_tag(tag: str) -> str:
    """ Normalize finish tag to standard format:
    PREFIX-NUMBER[SUFFIX]
    Examples: AC-01, PT-02F, WD-01, CT-02
    """
    if not tag:
        return ""
    s = tag.strip().upper()
    s = re.sub(r'[\s_]+', '-', s)
    s = s.replace('--', '-')
    s = s.replace('–', '-').replace('—', '-')
    match = re.match(r'^([A-Z]{1,4})-?(\d{1,3})([A-Z]?)$', s.replace('-', ''))
    if match:
        prefix, number, suffix = match.groups()
        return f"{prefix}-{int(number):02d}{suffix}".rstrip()
    return s
# -----------------------------
# Optional Debugging Endpoints
# -----------------------------
@app.get("/list-uploads")
async def list_uploads():
    """List all uploaded PDFs (for debugging)"""
    files = list(UPLOAD_DIR.glob("*.pdf"))
    return {
        "count": len(files),
        "files": [f.name for f in files]
    }

@app.delete("/cleanup")
async def cleanup_uploads():
    """Delete all uploaded PDFs (for testing)"""
    files = list(UPLOAD_DIR.glob("*.pdf"))
    for f in files:
        f.unlink()
    return {
        "deleted": len(files),
        "message": "All uploads cleaned up"
    }
# -----------------------------
# Run Instructions
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
