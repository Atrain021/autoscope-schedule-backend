# main.py - minimal, known-good version for finish schedule extraction

import os
import re
from typing import List, Dict, Any  # NEW

from fastapi import FastAPI, HTTPException, Query, UploadFile, File  # UploadFile, File ADDED
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://ta-01kbzevp9h1svdssernwsmwj4v-5173.wo-92h8yghsztgdf4c19ktmss9an.w.modal.host",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# CORS: for now, allow everything (safe for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where your PDFs live
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receive a PDF file, save it into the uploads folder,
    and return the filename that Base44 should use later.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "AutoScope schedule backend is running"}


@app.get("/extract-finish-schedule")
async def extract_finish_schedule(
    filename: str = Query(..., description="PDF file name inside the 'uploads' folder"),
    page_number: int = Query(..., ge=0, description="0-based page index"),
):
    """
    Read a single PDF page and extract candidate finish-schedule blocks,
    grouped by horizontal row band around each detected tag word,
    and LIMITED to the column to the RIGHT of the tag circle.
    """

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
    page_width = page.rect.width  # we’ll use this for the right edge of the column

    # 4) Get raw word objects from PyMuPDF
    # returns: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    words = page.get_text("words")

    # ---- PASS 1: detect tag rows (tag + approximate y_center + block_no + tag_x1) ----

    # Tag pattern AFTER normalization (e.g. "GL01", "DGF02", "PT10S")
    tag_core_pattern = re.compile(r"^([A-Z]{1,4})(\d{1,3})([A-Z]?)$")

    def normalize_candidate(tokens):
        """
        tokens: list of raw text fragments, e.g. ["GL", "01"] or ["PT-10S"]
        returns: normalized tag like "GL-01" or None if not a tag
        """
        combined = "".join(tokens)
        core = combined.replace("-", "").replace(" ", "").upper()

        m = tag_core_pattern.match(core)
        if not m:
            return None

        prefix, num, suffix = m.groups()
        return f"{prefix}-{num}{suffix}"

    row_candidates: List[Dict[str, Any]] = []
    n = len(words)

    for i in range(n):
        x0_i, y0_i, x1_i, y1_i, text_i, blk_i, line_i, word_i = words[i]
        t1 = text_i.strip()
        if not t1:
            continue

        candidates = []

        # 1-word candidate
        tag1 = normalize_candidate([t1])
        if tag1:
            candidates.append((tag1, i, i))

        # 2-word candidate
        if i + 1 < n:
            t2 = words[i + 1][4].strip()
            if t2:
                tag2 = normalize_candidate([t1, t2])
                if tag2:
                    candidates.append((tag2, i, i + 1))

        # 3-word candidate (very defensive; rarely needed)
        if i + 2 < n:
            t2 = words[i + 1][4].strip()
            t3 = words[i + 2][4].strip()
            if t2 and t3:
                tag3 = normalize_candidate([t1, t2, t3])
                if tag3:
                    candidates.append((tag3, i, i + 2))

        for tag_text, start_idx, end_idx in candidates:
            xs0, ys0, xs1, ys1 = [], [], [], []

            # use the block_no of the first word in the tag
            tag_block_no = words[start_idx][5]

            for j in range(start_idx, end_idx + 1):
                x0j, y0j, x1j, y1j, tj, b_j, l_j, w_j = words[j]
                xs0.append(x0j)
                ys0.append(y0j)
                xs1.append(x1j)
                ys1.append(y1j)

            y_center = (min(ys0) + max(ys1)) / 2.0
            tag_x1 = max(xs1)  # RIGHT edge of the tag circle/label

            row_candidates.append(
                {
                    "tag": tag_text,
                    "y_center": y_center,
                    "block_no": tag_block_no,
                    "tag_x1": tag_x1,
                }
            )

    # Deduplicate rows by (tag, rounded y_center, block_no)
    seen_keys = set()
    rows: List[Dict[str, Any]] = []
    for rc in row_candidates:
        key = (rc["tag"], round(rc["y_center"], 1), rc["block_no"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(rc)

    # Sort by vertical position (top to bottom)
    rows.sort(key=lambda r: r["y_center"])

    # ---- PASS 2: for each tag row, define a non-overlapping vertical band
    #              AND a horizontal band to the right of the tag ----

    blocks: List[Dict[str, Any]] = []

    if not rows:
        doc.close()
        return {
            "page": page_number,
            "num_blocks": 0,
            "blocks": []
        }

    for idx, row in enumerate(rows):
        tag_text = row["tag"]
        y = row["y_center"]
        row_block_no = row["block_no"]
        tag_x1 = row["tag_x1"]

        prev_y = rows[idx - 1]["y_center"] if idx > 0 else None
        next_y = rows[idx + 1]["y_center"] if idx < len(rows) - 1 else None

        # Vertical band: halfway between neighbors (same as you had)
        if prev_y is None and next_y is not None:
            region_top = y - (next_y - y) / 2.0
        elif prev_y is None and next_y is None:
            region_top = y - 20.0
        else:
            region_top = (prev_y + y) / 2.0

        if next_y is None and prev_y is not None:
            region_bottom = y + (y - prev_y) / 2.0
        elif next_y is None and prev_y is None:
            region_bottom = y + 20.0
        else:
            region_bottom = (next_y + y) / 2.0

        # Minimum band height
        min_height = 20.0
        if region_bottom - region_top < min_height:
            mid = (region_top + region_bottom) / 2.0
            region_top = mid - min_height / 2.0
            region_bottom = mid + min_height / 2.0

        # Extra vertical padding so we don't miss top/bottom lines like "MFR: DELETED"
        vertical_padding = 8.0  # you can tune this between ~6–12
        region_top -= vertical_padding
        region_bottom += vertical_padding

        # NEW: horizontal band – only take words to the RIGHT of the tag circle
        margin_right = 5.0  # a small gap so we don’t pick up the tag text itself
        horiz_left = tag_x1 + margin_right
        horiz_right = page_width  # up to the right edge of the page

        line_words = []
        for wx0, wy0, wx1, wy1, wtext, wblock, wline, wword in words:
            wy_center = (wy0 + wy1) / 2.0

            # vertical filter (same as before)
            if not (region_top <= wy_center <= region_bottom):
                continue

            # horizontal filter – REJECT left-column & unrelated stuff
            if wx1 < horiz_left or wx0 > horiz_right:
                continue

            line_words.append((wy_center, wx0, wtext))

        # Sort by vertical, then left-to-right
        line_words.sort(key=lambda t: (t[0], t[1]))

        block_text = " ".join(w[2] for w in line_words)

        # --- NEW: try to start text at the first real schedule label ---
        labels = ["MFR:", "LOC:", "COLOR:", "PATT:", "PATTERN:", "FINISH:", "SCALE:", "SKU:", "ITEM:"]
        first_label_pos = None
        upper = block_text.upper()
        for label in labels:
            idx = upper.find(label)
            if idx != -1:
                if first_label_pos is None or idx < first_label_pos:
                    first_label_pos = idx

        if first_label_pos is not None:
            block_text = block_text[first_label_pos:].strip()
        # ---------------------------------------------------------------

        blocks.append(
            {
                "tag": tag_text,
                "y_center": y,
                "region_top": region_top,
                "region_bottom": region_bottom,
                "block_no": row_block_no,
                "block_text": block_text,
            }
        )


    doc.close()

    return JSONResponse(
        {
            "page": page_number,
            "num_blocks": len(blocks),
            "blocks": blocks,
        }
    )

