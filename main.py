# main.py - AutoScope Schedule Backend (Production Ready) 

import os
import re
import gc
import json
import uuid
import base64
from typing import Dict, List, Optional
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from openai import OpenAI
from typing import Optional, Dict, List

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
    Classify a single PDF page image as FINISH_SCHEDULE / FLOOR_PLAN / OTHER.
    Also attempt to read sheet_identifier and sheet_title if visible.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """You are looking at ONE page from a construction drawing PDF.

Classify the page visually into exactly one of:
- FINISH_SCHEDULE (table/schedule with rows/columns, finish tags like AC-01/PT-02/etc)
- FLOOR_PLAN (plan view with rooms/walls/dimensions/tags)
- OTHER (everything else)

Also extract if visible:
- sheet_identifier (ex: I-601, A-201, ID-411, etc.)
- sheet_title (ex: "LEVEL 1 FLOOR PLAN", "FINISH SCHEDULE", etc.)

Return ONLY valid JSON in exactly this format:
{
  "classification": "FINISH_SCHEDULE",
  "sheet_identifier": "I-601",
  "sheet_title": "FINISH SCHEDULE"
}

Rules:
- If sheet_identifier/title are not clearly visible, return "" for them.
- No markdown. No extra text. JSON only.
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

    raw = resp.choices[0].message.content.strip()
    cleaned = _clean_json_text(raw)
    data = json.loads(cleaned)

    # Hard normalize classification
    cls = (data.get("classification") or "").strip().upper()
    if cls not in {"FINISH_SCHEDULE", "FLOOR_PLAN", "OTHER"}:
        cls = "OTHER"

    return {
        "classification": cls,
        "sheet_identifier": (data.get("sheet_identifier") or "").strip(),
        "sheet_title": (data.get("sheet_title") or "").strip(),
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

        # Convert user range (1-based) -> python range (0-based)
        start = max(1, int(request.start_page))
        end = int(request.end_page) if request.end_page else total_pages
        end = min(end, total_pages)

        if start > end:
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
