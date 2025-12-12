
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
SECTION_HEADER_RE = re.compile(r"^\s*\d{2}\s+\d{2}\s+\d{2}\s+")
SECTION_KEYWORDS = [
    "ACOUSTICAL", "CEILINGS", "WALL COVERINGS", "WALLCOVERINGS",
    "PAINT", "PAINTING", "TILE", "FLOORING", "CARPET", "BASE",
    "COUNTERTOP", "QUARTZ", "STONE", "GLASS", "DOOR", "MILLWORK"
]

def _group_words_into_lines(words, y_tol=2.0):
    """
    words: list of (x0, y0, x1, y1, text, block_no, line_no, word_no)
    Returns list of dict lines with: y, x0, x1, text, y0, y1
    """
    items = []
    for w in words:
        x0, y0, x1, y1, t, *_ = w
        t = (t or "").strip()
        if not t:
            continue
        y = (y0 + y1) / 2.0
        items.append((y, x0, x1, y0, y1, t))

    items.sort(key=lambda r: (r[0], r[1]))

    lines = []
    cur = []
    cur_y = None

    def flush():
        if not cur:
            return
        cur_sorted = sorted(cur, key=lambda r: r[1])  # by x0
        text = " ".join([r[5] for r in cur_sorted]).strip()
        y0 = min(r[3] for r in cur_sorted)
        y1 = max(r[4] for r in cur_sorted)
        x0 = min(r[1] for r in cur_sorted)
        x1 = max(r[2] for r in cur_sorted)
        y = sum(r[0] for r in cur_sorted) / len(cur_sorted)
        lines.append({"y": y, "y0": y0, "y1": y1, "x0": x0, "x1": x1, "text": text})

    for y, x0, x1, y0, y1, t in items:
        if cur_y is None:
            cur_y = y
            cur = [(y, x0, x1, y0, y1, t)]
            continue
        if abs(y - cur_y) <= y_tol:
            cur.append((y, x0, x1, y0, y1, t))
        else:
            flush()
            cur_y = y
            cur = [(y, x0, x1, y0, y1, t)]

    flush()
    return lines

def detect_section_headers(page) -> List[Dict[str, Any]]:
    """
    Detect schedule section headers using the text layer.
    Returns list of {y0, y1, y, text}
    """
    words = page.get_text("words")
    lines = _group_words_into_lines(words, y_tol=2.5)

    headers = []
    for ln in lines:
        txt = (ln["text"] or "").strip()
        txt_u = txt.upper()

        is_header = False
        if SECTION_HEADER_RE.search(txt):
            is_header = True
        else:
            # keyword fallback
            for kw in SECTION_KEYWORDS:
                if kw in txt_u:
                    # only treat as header if it looks “big / title-ish”
                    # (usually all caps and not too long)
                    if len(txt_u) <= 80:
                        is_header = True
                    break

        if is_header:
            headers.append({"y0": ln["y0"], "y1": ln["y1"], "y": ln["y"], "text": txt})

    headers.sort(key=lambda h: h["y0"])
    return headers

def build_section_bands(page, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build vertical section rectangles from header -> next header.
    Returns list of {top, bottom, header_text}
    """
    rect = page.rect
    bands = []

    if not headers:
        # one band = whole page
        bands.append({"top": rect.y0, "bottom": rect.y1, "header_text": "FULL_PAGE"})
        return bands

    for i, h in enumerate(headers):
        top = max(rect.y0, h["y0"] - 6)  # small padding above header
        if i + 1 < len(headers):
            bottom = min(rect.y1, headers[i + 1]["y0"] - 6)
        else:
            bottom = rect.y1

        if bottom - top < 40:
            continue

        bands.append({"top": top, "bottom": bottom, "header_text": h["text"]})

    return bands

def render_page_clip_base64(page, clip_rect: fitz.Rect, max_long_side_px=2200.0) -> str:
    """
    Render a clipped region of the page to PNG base64 data-url,
    with a pixel cap to avoid memory errors on Render.
    """
    w_pts = clip_rect.width
    h_pts = clip_rect.height
    longer_pts = max(w_pts, h_pts)

    zoom = max_long_side_px / max(longer_pts, 1.0)
    zoom = min(zoom, 3.0)  # safety
    zoom = max(zoom, 0.6)  # avoid too tiny

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

    b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "AutoScope schedule backend is running"}


@app.get("/extract-finish-schedule")
async def extract_finish_schedule(
    filename: str,
    page_number: int = Query(..., ge=0),
):
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found")

    doc = fitz.open(pdf_path)
    if page_number < 0 or page_number >= doc.page_count:
        raise HTTPException(status_code=400, detail="Invalid page number")

    page = doc[page_number]

    # 1) Detect tag rows (you already have this function; keep using it)
    tag_rows = detect_tag_rows(page)
    if not tag_rows:
        return {"page": page_number, "num_blocks": 0, "blocks": []}

    tag_rows_sorted = sorted(tag_rows, key=lambda r: (r["y_center"], r.get("tag_x0", 0)))
    all_tags = [r["tag"] for r in tag_rows_sorted]

    # 2) Detect section headers and build section bands
    headers = detect_section_headers(page)
    bands = build_section_bands(page, headers)

    # 3) Vision extract per band (few calls per page)
    by_tag: Dict[str, str] = {}

    for band in bands:
        top = band["top"]
        bottom = band["bottom"]
        header_text = band["header_text"]

        # Tags whose y_center falls inside this band
        band_tags = [r["tag"] for r in tag_rows_sorted if top <= r["y_center"] <= bottom]
        if not band_tags:
            continue

        clip_rect = fitz.Rect(page.rect.x0, top, page.rect.x1, bottom)
        img_url = render_page_clip_base64(page, clip_rect, max_long_side_px=2200.0)

        tags_str = ", ".join(band_tags)

        prompt = (
            "You are reading a cropped region of an interior finish schedule from construction drawings.\n\n"
            f"SECTION TITLE (context only): {header_text}\n\n"
            "This crop contains a grid/table of schedule rows. Each row is identified on the far left by a\n"
            "small circular/boxed tag label (example: AC-01 shown as 'AC 01').\n\n"
            "TASK:\n"
            "For each tag in the list below, locate that tag label in the crop and read ONLY the text that\n"
            "belongs to that same ROW (bounded by the horizontal grid lines immediately above and below).\n"
            "Then return one JSON item per tag.\n\n"
            "TAGS TO EXTRACT (ONLY THESE):\n"
            f"{tags_str}\n\n"
            "RULES:\n"
            " - Do NOT mix rows. Text must be in the same row band as the tag.\n"
            " - Output must be strict JSON only.\n"
            " - IMPORTANT: Only return block_text = \"DELETED\" if you can literally see the word DELETED\n"
            "   in the same row as that tag. Otherwise, if you are unsure, use an empty string.\n"
            " - If a tag is not visible in this crop, include it with block_text=\"\".\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "{ \"items\": [ { \"tag\": \"AC-01\", \"block_text\": \"...\" }, ... ] }\n"
        )

        try:
            resp = client.responses.create(
                model="gpt-4.1",  # accuracy > cost here; we are doing few calls
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": img_url},
                        ],
                    }
                ],
            )
            raw = resp.output_text
            data = json.loads(raw)
        except Exception:
            data = {"items": []}

        items = data.get("items", []) if isinstance(data, dict) else []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                t = str(it.get("tag", "")).strip()
                bt = str(it.get("block_text", "")).strip()
                if not t:
                    continue

                # Safety: prevent false DELETED unless the model explicitly returned it
                if bt.upper() == "DELETED":
                    by_tag.setdefault(t, "DELETED")
                else:
                    by_tag.setdefault(t, bt)

    # 4) Build response blocks (keep same shape your frontend expects)
    blocks = []
    for row in tag_rows_sorted:
        tag = row["tag"]
        y = float(row["y_center"])
        blocks.append(
            {
                "tag": tag,
                "y_center": y,
                "region_top": y - 20,
                "region_bottom": y + 20,
                "block_no": 0,
                "block_text": by_tag.get(tag, ""),
            }
        )

    return {"page": page_number, "num_blocks": len(blocks), "blocks": blocks}
