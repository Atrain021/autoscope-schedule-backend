
# main.py - AutoScope finish schedule backend

import os
import re
import json
import base64
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import fitz  # PyMuPDF

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()




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
      {
        "tag": str,
        "prefix": str,
        "y_center": float,
        "tag_x0": float,
        "tag_x1": float,
        "tag_y0": float,
        "tag_y1": float
      }
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
                    (
                        tag2,
                        min(x0, x0b),
                        max(x1, x1b),
                        min(y0, y0b),
                        max(y1, y1b),
                    )
                )

        if not candidates:
            continue

        # Take the first candidate
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
                "tag_y0": ty0,
                "tag_y1": ty1,
            }
        )

    return rows



def compute_row_rects(page, rows, vertical_pad: float = 30.0, margin: float = 4.0):
    """
    For each detected tag:

    - Use the tag's own bounding box (tag_y0, tag_y1) as the vertical anchor.
    - Expand that box up and down by a small padding (vertical_pad).
    - Restrict horizontally to that prefix's column slice (AC vs WC vs CT vs PT, etc.).

    This removes all "smart" banding and ties each row rectangle tightly to the
    actual tag bubble on the page.
    """
    if not rows:
        return []

    page_rect = page.rect

    # 1) Group rows by prefix
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prefix[r["prefix"]].append(r)

    # 2) Compute average x-center for each prefix
    prefix_centers: dict[str, float] = {}
    for prefix, group in by_prefix.items():
        centers = [ (g["tag_x0"] + g["tag_x1"]) / 2.0 for g in group ]
        prefix_centers[prefix] = sum(centers) / max(len(centers), 1)

    # 3) Sort prefixes left-to-right and define horizontal bounds between centers
    sorted_prefixes = sorted(prefix_centers.items(), key=lambda kv: kv[1])
    prefix_bounds: dict[str, tuple[float, float]] = {}

    for idx, (prefix, center) in enumerate(sorted_prefixes):
        if idx == 0:
            left = page_rect.x0
        else:
            prev_center = sorted_prefixes[idx - 1][1]
            left = (prev_center + center) / 2.0

        if idx == len(sorted_prefixes) - 1:
            right = page_rect.x1
        else:
            next_center = sorted_prefixes[idx + 1][1]
            right = (center + next_center) / 2.0

        prefix_bounds[prefix] = (left, right)

    # 4) Build one tight rectangle per row using the tag box + padding
    row_rects: list[dict] = []

    for prefix, group in by_prefix.items():
        left, right = prefix_bounds[prefix]

        for r in group:
            tag = r["tag"]
            y0 = float(r["tag_y0"])
            y1 = float(r["tag_y1"])
            y_center = float(r["y_center"])

            # Vertical band = tag box ± vertical_pad
            top = y0 - vertical_pad
            bottom = y1 + vertical_pad

            # Clamp to page
            top = max(page_rect.y0, top)
            bottom = min(page_rect.y1, bottom)
            if bottom <= top:
                # minimal safety height if something weird happens
                bottom = top + max(10.0, (y1 - y0) + 2 * vertical_pad)

            # Horizontal slice for this prefix, with a small margin
            x0 = max(page_rect.x0, left + margin)
            x1 = min(page_rect.x1, right - margin)

            row_rects.append(
                {
                    "tag": tag,
                    "prefix": prefix,
                    "y_center": y_center,
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
    Vision-based finish schedule extraction (full-page).

    Flow:
    1) Open the uploaded PDF and the target page.
    2) Detect all finish tags (AC-01, PT-10S, CT-10, DGF-01, etc.) using text.
    3) Render the entire page to a PNG image.
    4) Ask OpenAI vision: given THIS page image and THIS exact list of tags,
       return a JSON object with one entry per tag:
         { "tag": "<TAG>", "block_text": "<ROW TEXT>" }
    5) Return blocks in the legacy format used by Base44.
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

    # 1) Detect tags on this page
    tag_rows = detect_tag_rows(page)
    if not tag_rows:
        # No tags detected; just return empty
        return {
            "page": page_number,
            "num_blocks": 0,
            "blocks": [],
        }

    # Sort tags in reading order (top to bottom, left to right)
    tag_rows_sorted = sorted(
        tag_rows,
        key=lambda r: (r["y_center"], r["tag_x0"]),
    )
    tag_list = [row["tag"] for row in tag_rows_sorted]

    # 2) Render the entire page as an image, but cap pixel size to avoid OOM
    page_rect = page.rect

    # We want the longer side to be at most ~3000 pixels
    max_pixels = 3000.0
    longer_side_points = max(page_rect.width, page_rect.height)

    # Base resolution is 72 dpi → 1 point = 1 pixel at zoom=1
    # Compute zoom so longer side ≈ max_pixels (but cap at 2x for safety)
    zoom = max_pixels / longer_side_points
    zoom = min(zoom, 2.0)
    if zoom <= 0:
        zoom = 1.0

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    img_data_url = f"data:image/png;base64,{img_b64}"


    # 3) Build the instruction for vision
    tags_str = ", ".join(tag_list)

    instructions = (
        "You are reading an interior finish schedule page from an architectural drawing set.\n\n"
        "The page contains one or more GRIDDed schedules laid out in rows and columns. Each row\n"
        "is identified on the far left by a small circular or boxed TAG label such as AC-01,\n"
        "AC-02, WC-10, PT-07F, CT-10, DGF-01, etc.\n\n"
        "IMPORTANT: For each tag, you MUST do the following EXACTLY:\n"
        "  1. Visually locate the tag label itself (e.g. the circle that says 'AC 01').\n"
        "  2. Identify the horizontal grid ROW that this tag label sits in. That row is bounded\n"
        "     by the horizontal lines immediately above and below it in the table.\n"
        "  3. Follow that row horizontally across the table, staying strictly between those\n"
        "     two horizontal grid lines. Only text inside that band belongs to that tag.\n"
        "  4. Within that band, read the text in normal reading order (left to right, then\n"
        "     the next column, etc.). This typically includes columns such as manufacturer,\n"
        "     product, pattern, color, size, and location (LOC: ...).\n"
        "  5. Completely IGNORE any text that is above or below that tag's row, even if it\n"
        "     mentions similar products (e.g. Armstrong flooring) or looks related.\n"
        "     If the text is not in the same horizontal band as the tag label, it is NOT part\n"
        "     of that tag's row.\n\n"
        "The set of tags that appear on THIS page and that we care about is:\n"
        f"{tags_str}\n\n"
        "Your job:\n"
        "  - For EACH tag in this list, find the corresponding row as described above.\n"
        "  - Read ALL useful text inside that row band (between its horizontal lines), across\n"
        "    ALL relevant columns of that row.\n"
        "  - For each tag, produce an object:\n"
        '      { \"tag\": \"<TAG>\", \"block_text\": \"<ROW_TEXT>\" }\n\n'
        "ROW_TEXT should:\n"
        "  - Be a single line of text that concatenates the row contents in a readable way.\n"
        "  - Include, when present, things like manufacturer, product, pattern, color, size,\n"
        "    and location (LOC: ...), e.g.: \"MFR: ARKTURA PROD: VAPOR PATT: CUMULA COLOR: WHITE\n"
        "    SIZE: 16'7\" X 51'6\" LOC: LOBBY - CEILING DROP HEIGHT TO BOTTOM: 11'-0\" AFF\".\n"
        "  - You may preserve labels like 'MFR:', 'PROD:', 'COLOR:', 'PATT:', 'SIZE:', 'LOC:'.\n"
        "  - MUST NOT include text from other unrelated rows or other schedules on the page.\n\n"
        "Deleted / empty rows:\n"
        "  - If a tag clearly indicates the finish is deleted or has no meaningful info,\n"
        "    you may set block_text to an empty string or a short note like 'DELETED'.\n"
        "  - If you absolutely cannot find a tag on the page, include it with block_text = \"\".\n\n"
        "Output format:\n"
        "  - Return a single JSON object of the exact form:\n"
        "      { \"items\": [ { \"tag\": \"...\", \"block_text\": \"...\" }, ... ] }\n"
        "  - There must be exactly ONE item for each tag in the list I provided.\n"
        "  - Do NOT add any tags I did not list.\n"
        "  - Do NOT include any explanations, comments, or extra text outside that JSON.\n"
    )


    # 4) Call OpenAI vision
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instructions},
                        {"type": "input_image", "image_url": img_data_url},
                    ],
                }
            ],
        )
        raw_text = response.output_text
        data = json.loads(raw_text)
    except Exception as e:
        # If anything goes wrong, fall back to empty block_texts
        data = {"items": []}

    # Build a mapping from tag -> block_text from the model output
    by_tag: dict[str, str] = {}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            t = str(item.get("tag", "")).strip()
            bt = str(item.get("block_text", "")).strip()
            if t:
                by_tag[t] = bt

    # 5) Build the legacy blocks array expected by the frontend
    blocks = []
    for row in tag_rows_sorted:
        tag = row["tag"]
        y = float(row["y_center"])
        # region_top / region_bottom are not trusted for geometry anymore;
        # we just give a small band around the y-center for debugging.
        band_height = 40.0
        top = y - band_height / 2.0
        bottom = y + band_height / 2.0

        block_text = by_tag.get(tag, "")

        blocks.append(
            {
                "tag": tag,
                "y_center": y,
                "region_top": top,
                "region_bottom": bottom,
                "block_no": 0,  # no real meaning here; kept for compatibility
                "block_text": block_text,
            }
        )

    return {
        "page": page_number,
        "num_blocks": len(blocks),
        "blocks": blocks,
    }
