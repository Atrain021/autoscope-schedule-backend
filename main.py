# main.py - minimal, known-good version for finish schedule extraction

import os
import json
import math
import base64
import re
from collections import defaultdict

import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI


app = FastAPI()

client = OpenAI()
# === Vision-based finish schedule helpers ===

TAG_CORE_RE = re.compile(r"^([A-Z]{1,4})(\d{1,3})([A-Z]?)$")


def normalize_tag_core(tokens):
    """
    Normalize small token sequences like ["AC", "01"] or ["PT-10S"]
    into a canonical tag like "AC-01" or "PT-10S".
    """
    core = "".join(tokens)
    core = core.replace("-", "").replace(" ", "").upper()
    m = TAG_CORE_RE.match(core)
    if not m:
        return None
    prefix, num, suffix = m.groups()
    # preserve zero padding on num (01, 02, etc.)
    return f"{prefix}-{num}{suffix}"


def detect_tag_rows(page):
    """
    Scan page words and detect tag bubbles.

    Returns list of dicts:
      { "tag", "prefix", "y_center", "tag_x0", "tag_x1" }
    """
    # PyMuPDF "words" format:
    # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    words = page.get_text("words")
    rows = []
    used_tags = set()
    n = len(words)

    for i, w in enumerate(words):
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        text = text.strip()
        if not text:
            continue

        candidates = []

        # Single-token candidate, e.g. "AC01" or "PT10S"
        tag1 = normalize_tag_core([text])
        if tag1:
            candidates.append((tag1, x0, x1, y0, y1))

        # Two-token candidate, e.g. "AC" "01"
        if i + 1 < n:
            x0b, y0b, x1b, y1b, text2, *_ = words[i + 1]
            tag2 = normalize_tag_core([text, text2])
            if tag2:
                candidates.append(
                    (tag2,
                     min(x0, x0b),
                     max(x1, x1b),
                     min(y0, y0b),
                     max(y1, y1b))
                )

        if not candidates:
            continue

        tag, tx0, tx1, ty0, ty1 = candidates[0]
        if tag in used_tags:
            continue

        used_tags.add(tag)
        rows.append(
            {
                "tag": tag,
                "prefix": tag.split("-")[0],
                "y_center": (ty0 + ty1) / 2.0,
                "tag_x0": tx0,
                "tag_x1": tx1,
            }
        )

    return rows


def compute_row_rects(page, rows, min_band_height: float = 40.0, margin: float = 4.0):
    """
    Given tag rows, compute rectangular regions for each row.

    - Horizontally: partition page by tag prefix (AC vs WC vs PT, etc.)
      based on average x-position of tags.
    - Vertically: use midpoints between neighboring rows of the same prefix.
    """
    if not rows:
        return []

    page_rect = page.rect
    page_width = page_rect.width

    # Group rows by prefix (AC, WC, PT, etc.)
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prefix[r["prefix"]].append(r)

    # Compute average x-center for each prefix
    prefix_centers: dict[str, float] = {}
    for prefix, group in by_prefix.items():
        centers = [(g["tag_x0"] + g["tag_x1"]) / 2.0 for g in group]
        prefix_centers[prefix] = sum(centers) / max(len(centers), 1)

    # Order prefixes from left to right
    sorted_prefixes = sorted(prefix_centers.items(), key=lambda kv: kv[1])
    prefix_order = [p for p, _ in sorted_prefixes]

    # For each prefix, compute a horizontal slice [left, right]
    prefix_bounds: dict[str, tuple[float, float]] = {}
    for idx, prefix in enumerate(prefix_order):
        center = prefix_centers[prefix]

        if idx == 0:
            left = page_rect.x0
        else:
            prev_center = prefix_centers[prefix_order[idx - 1]]
            left = (prev_center + center) / 2.0

        if idx == len(prefix_order) - 1:
            right = page_width
        else:
            next_center = prefix_centers[prefix_order[idx + 1]]
            right = (center + next_center) / 2.0

        prefix_bounds[prefix] = (left, right)

    row_rects: list[dict] = []

    # Now compute vertical bands within each prefix’s slice
    # Use a fixed-height band around each tag center so we never grab half the page.
    band_height = min_band_height  # interpret min_band_height as our fixed band height

    for prefix, group in by_prefix.items():
        group_sorted = sorted(group, key=lambda r: r["y_center"])
        left, right = prefix_bounds[prefix]

        for r in group_sorted:
            y = r["y_center"]

            # Centered fixed-height band
            top = y - band_height / 2.0
            bottom = y + band_height / 2.0

            # Clamp to page
            top = max(page_rect.y0, top)
            bottom = min(page_rect.y1, bottom)
            if bottom <= top:
                bottom = top + band_height / 2.0

            # Add a small horizontal margin and clamp
            x0 = max(page_rect.x0, left + margin)
            x1 = min(page_rect.x1, right - margin)

            row_rects.append(
                {
                    "tag": r["tag"],
                    "prefix": prefix,
                    "y_center": y,
                    "region_top": top,
                    "region_bottom": bottom,
                    "rect": (x0, top, x1, bottom),
                }
            )

    return row_rects



def vision_transcribe_rows(page, row_rects, batch_size: int = 8):
    """
    For each row rectangle, crop to an image and use OpenAI vision
    to transcribe the row text.

    Returns list of dicts:
      { "tag", "y_center", "region_top", "region_bottom", "block_text" }
    """
    results: list[dict] = []

    for i in range(0, len(row_rects), batch_size):
        batch = row_rects[i : i + batch_size]

        # Build multi-image prompt
        content: list[dict] = []

        instructions = (
            "You are reading interior finish schedule rows from architectural drawings. "
            "For each row IMAGE, I will tell you its TAG. Your job is to transcribe the "
            "row text in normal reading order (left to right, top to bottom) and return "
            "a pure JSON array. Each element in the array must be an object with:\n"
            '{ \"tag\": \"<TAG>\", \"block_text\": \"<ONE_LINE_TEXT>\" }\n\n'
            "Rules:\n"
            "- Use the exact tag I provide; do NOT change it.\n"
            "- For block_text, include all useful information you can read for that row: "
            "manufacturer, product, color, pattern, size, location, notes, etc., in a single line. "
            "You may insert labels like 'MFR:', 'PROD:', 'COLOR:', 'LOC:' if they appear.\n"
            "- If the row appears blank or unreadable, use an empty string for block_text.\n"
            "- Output ONLY JSON. No explanations, no comments."
        )
        content.append({"type": "input_text", "text": instructions})

        # Attach each row image with its tag
        for idx, row in enumerate(batch, start=1):
            rect = fitz.Rect(*row["rect"])
            pix = page.get_pixmap(clip=rect, dpi=250)
            image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

            content.append(
                {
                    "type": "input_text",
                    "text": f"Row {idx} TAG: {row['tag']}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                }
            )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        text = response.output_text

        try:
            batch_rows = json.loads(text)
            if not isinstance(batch_rows, list):
                raise ValueError("Expected a JSON array")
        except Exception:
            # If anything goes wrong, just fallback to empty block_text
            batch_rows = [
                {"tag": row["tag"], "block_text": ""} for row in batch
            ]

        # Build a simple mapping tag -> block_text
        by_tag = {
            str(item.get("tag")): item.get("block_text", "")
            for item in batch_rows
            if isinstance(item, dict) and "tag" in item
        }

        for row in batch:
            block_text = by_tag.get(row["tag"], "")
            results.append(
                {
                    "tag": row["tag"],
                    "y_center": row["y_center"],
                    "region_top": row["region_top"],
                    "region_bottom": row["region_bottom"],
                    "block_text": block_text,
                }
            )

    return results


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
    filename: str,
    page_number: int = Query(..., ge=0),
):
    """
    Vision-based finish schedule extraction.

    1) Detect tag positions (AC-01, WC-08, PT-10S, etc.)
    2) Compute per-tag row rectangles (geometry bands).
    3) Use OpenAI vision to transcribe each row image into clean text.
    4) Return blocks with tag + block_text (one line per row).
    """
    # Adjust this if your upload directory variable has a different name
    upload_dir = UPLOAD_DIR if "UPLOAD_DIR" in globals() else "uploads"
    pdf_path = os.path.join(upload_dir, filename)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found")

    doc = fitz.open(pdf_path)

    if page_number < 0 or page_number >= doc.page_count:
        raise HTTPException(status_code=400, detail="Invalid page number")

    page = doc[page_number]

    # 1. Detect tag rows
    tag_rows = detect_tag_rows(page)

    # 2. Compute row rectangles
    row_rects = compute_row_rects(page, tag_rows)

    # 3. Vision transcription
    blocks = vision_transcribe_rows(page, row_rects)

    return {
        "page": page_number,
        "num_blocks": len(blocks),
        "blocks": blocks,
    }
