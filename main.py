# main.py - AutoScope Schedule Backend (Production Ready) 

import os
import re
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
# CORS for Base44 integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    
    def pdf_page_to_image(self, pdf_path: str, page_number: int, dpi: int = 300) -> bytes:
        """
        Convert PDF page to high-resolution PNG image.
        High DPI ensures schedule text is crisp for vision models.
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Validate page number
            if page_number < 0 or page_number >= len(doc):
                raise ValueError(f"Invalid page number {page_number}. PDF has {len(doc)} pages.")
            
            page = doc[page_number]
            
            # Render at high resolution (300 DPI)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            doc.close()
            
            return img_bytes
            
        except Exception as e:
            raise Exception(f"PDF to image conversion failed: {str(e)}")
    
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
