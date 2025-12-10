# main.py
import io
import re
from typing import List, Dict, Any

import pdfplumber
import os
UPLOAD_DIR = "uploads"
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env vars (for later when we add OpenAI)
load_dotenv()

app = FastAPI(
    title="AutoScope Finish Schedule Backend",
    description="External service to read finish schedules row-by-row.",
    version="0.2.0",
)

# Allow calls from browser / Base44 later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SMALL HELPERS ---------- #

TAG_FULL_RE = re.compile(r'^[A-Z]{1,4}-?\d+[A-Z]*$')   # GL-01, RF-03A, PT-06HG
TAG_LETTERS_RE = re.compile(r'^[A-Z]{1,4}$')           # GL, RF, PT
TAG_DIGITS_RE = re.compile(r'^\d+[A-Z]*$')             # 01, 03A, 06HG


def norm_tag(text: str) -> str:
    return text.replace(" ", "").upper().strip()


def is_full_tag_token(text: str) -> bool:
    return bool(TAG_FULL_RE.match(norm_tag(text)))


def looks_like_tag_parts(t1: str, t2: str) -> bool:
    """
    GL + 01  -> GL-01
    WC + 03  -> WC-03
    RF + 03A -> RF-03A
    """
    return bool(TAG_LETTERS_RE.match(t1)) and bool(TAG_DIGITS_RE.match(t2))


# ---------- API: HEALTH CHECK ---------- #

@app.get("/ping")
async def ping():
    return {
        "status": "ok",
        "message": "AutoScope backend is running",
    }


# ---------- CORE: EXTRACT FINISH SCHEDULE BLOCKS ---------- #

@app.get("/extract-finish-schedule")
async def extract_finish_schedule(
    filename: str = Query(..., description="PDF file name inside the 'uploads' folder"),
    page_number: int = Query(..., ge=0, description="0-based page index")
):
    # 1) Resolve PDF path
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found in uploads: {filename}")

    # 2) Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open PDF: {e}")

    # 3) Validate page number
    if page_number < 0 or page_number >= len(doc):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid page_number={page_number}. PDF has {len(doc)} pages (0-based).",
        )

    page = doc[page_number]

    # 4) Get raw word objects from PyMuPDF
    words = page.get_text("words")  # (x0, y0, x1, y1, "text", block_no, line_no, word_no)

    blocks = []

    # Simple tag patterns like DGF-02, GL-01, WC-03, PT-13F, etc.
    tag_pattern = re.compile(r"^[A-Z]{1,4}[-\s]\d{1,3}[A-Z]?$")

    for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
        raw = text.strip()
        if not tag_pattern.match(raw):
            continue

        # Normalize tag text (GL 01 → GL-01)
        tag_text = raw.replace(" ", "-")

        # 5) Define a horizontal band around this row
        y_center = (y0 + y1) / 2.0
        region_top = y_center - 10
        region_bottom = y_center + 40

        # 6) Collect all words whose vertical center lies in this band
        line_words = []
        for wx0, wy0, wx1, wy1, wtext, wblock, wline, wword in words:
            wy_center = (wy0 + wy1) / 2.0
            if region_top <= wy_center <= region_bottom:
                line_words.append((wy_center, wx0, wtext))

        # Sort by vertical, then left-to-right
        line_words.sort(key=lambda t: (t[0], t[1]))

        # Build block text
        block_text = " ".join(w[2] for w in line_words)

        blocks.append({
            "tag": tag_text,
            "y_center": y_center,
            "region_top": region_top,
            "region_bottom": region_bottom,
            "block_text": block_text,
        })

    doc.close()

    return JSONResponse({
        "page": page_number,
        "num_blocks": len(blocks),
        "blocks": blocks,
    })
