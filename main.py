
# main.py - Finish Schedule backend (image-tiling + vision extraction)

import os
import re
import json
import base64
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenAI client (Render uses env var OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# For now, keep CORS permissive so Base44 can hit it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "AutoScope schedule backend running"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)

    return {"filename": safe_name}


# -----------------------------
# Helpers: tag detection (cheap)
# -----------------------------
TAG_CORE_RE = re.compile(r"^([A-Z]{1,4})(\d{1,3})([A-Z]?)$")


def normalize_tag_from_tokens(tokens: List[str]) -> str:
    """
    tokens might be ["AC", "01"] or ["PT-10S"] etc.
    returns normalized "AC-01" / "PT-10S" or "" if not valid.
    """
    raw = "".join(tokens)
    raw = raw.replace(" ", "").replace("-", "").replace("_", "").upper()
    m = TAG_CORE_RE.match(raw)
    if not m:
        return ""
    prefix, num, suffix = m.group(1), m.group(2), m.group(3)
    return f"{prefix}-{num}{suffix}"


def detect_tag_rows(page) -> List[Dict[str, Any]]:
    """
    Detect candidate finish tags + their approximate positions using the PDF text layer.
    Returns rows like:
      { tag, tag_x0, tag_y0, tag_x1, tag_y1, y_center }
    """
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
    # Sort by y then x
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))

    candidates = []
    n = len(words_sorted)

    for i in range(n):
        # Try 1-token tag
        x0, y0, x1, y1, t, *_ = words_sorted[i]
        t = (t or "").strip()
        if t:
            tag = normalize_tag_from_tokens([t])
            if tag:
                candidates.append(
                    {
                        "tag": tag,
                        "tag_x0": float(x0),
                        "tag_y0": float(y0),
                        "tag_x1": float(x1),
                        "tag_y1": float(y1),
                        "y_center": float((y0 + y1) / 2.0),
                    }
                )

        # Try 2-token tag (AC + 01)
        if i + 1 < n:
            x0b, y0b, x1b, y1b, t2, *_ = words_sorted[i + 1]
            t2 = (t2 or "").strip()
            if t and t2:
                # only combine if near each other vertically
                if abs(((y0 + y1) / 2.0) - ((y0b + y1b) / 2.0)) <= 6.0:
                    tag = normalize_tag_from_tokens([t, t2])
                    if tag:
                        candidates.append(
                            {
                                "tag": tag,
                                "tag_x0": float(min(x0, x0b)),
                                "tag_y0": float(min(y0, y0b)),
                                "tag_x1": float(max(x1, x1b)),
                                "tag_y1": float(max(y1, y1b)),
                                "y_center": float((((y0 + y1) / 2.0) + ((y0b + y1b) / 2.0)) / 2.0),
                            }
                        )

    # De-dup: keep the leftmost instance for each tag (usually the circle label)
    best: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        t = c["tag"]
        if t not in best:
            best[t] = c
        else:
            # prefer smaller x0 (tag circles are usually far left of their row)
            if c["tag_x0"] < best[t]["tag_x0"]:
                best[t] = c

    rows = list(best.values())
    rows.sort(key=lambda r: (r["y_center"], r["tag_x0"]))
    return rows


# -----------------------------
# Helpers: tiling + image render
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_tiles_for_landscape(page_rect: fitz.Rect,
                             cols: int = 3,
                             rows: int = 6,
                             col_overlap: float = 0.12,
                             row_overlap: float = 0.15) -> List[fitz.Rect]:
    """
    Return clip rects that tile the page in 2D with overlaps.
    cols=3 is ideal for landscape tables.
    """
    x0, y0, x1, y1 = page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1
    W = x1 - x0
    H = y1 - y0

    col_w = W / cols
    row_h = H / rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = x0 + c * col_w
            right = left + col_w
            top = y0 + r * row_h
            bottom = top + row_h

            # apply overlaps
            pad_x = col_w * col_overlap
            pad_y = row_h * row_overlap

            left2 = clamp(left - pad_x, x0, x1)
            right2 = clamp(right + pad_x, x0, x1)
            top2 = clamp(top - pad_y, y0, y1)
            bottom2 = clamp(bottom + pad_y, y0, y1)

            if right2 - left2 >= 40 and bottom2 - top2 >= 40:
                tiles.append(fitz.Rect(left2, top2, right2, bottom2))

    return tiles


def render_clip_as_data_url(page, clip: fitz.Rect, max_long_side_px: float = 1500.0) -> str:
    """
    Render the clipped region as a PNG data URL, with a pixel cap to avoid memory issues on Render.
    """
    w_pts = clip.width
    h_pts = clip.height
    longer_pts = max(w_pts, h_pts)

    zoom = max_long_side_px / max(longer_pts, 1.0)
    zoom = clamp(zoom, 0.6, 2.5)

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

    b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def tile_contains_any_tags(tile: fitz.Rect, tag_rows: List[Dict[str, Any]]) -> bool:
    for r in tag_rows:
        x = float(r["tag_x0"])
        y = float(r["y_center"])
        if tile.x0 <= x <= tile.x1 and tile.y0 <= y <= tile.y1:
            return True
    return False


def tags_in_tile(tile: fitz.Rect, tag_rows: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in tag_rows:
        x = float(r["tag_x0"])
        y = float(r["y_center"])
        if tile.x0 <= x <= tile.x1 and tile.y0 <= y <= tile.y1:
            out.append(r["tag"])
    # stable order
    return sorted(set(out), key=lambda t: (t.split("-")[0], t.split("-")[1]))


def score_block_text(s: str) -> int:
    """
    Choose the best candidate across overlapping tiles.
    Higher score = more likely a clean row.
    Penalize monster junk.
    """
    if not s:
        return 0
    su = s.upper()
    labels = 0
    for k in ["MFR", "MANUFACTURER", "PROD", "PRODUCT", "COLOR", "PATT", "PAT", "SIZE", "LOC", "FINISH", "SCALE"]:
        if k in su:
            labels += 1
    length = len(s)
    # penalize huge contaminated text
    if length > 700:
        length = 700 - (length - 700)  # starts dropping
    return max(0, length) + labels * 60


# -----------------------------
# Vision extraction per tile
# -----------------------------
def vision_extract_for_tile(image_url: str, tag_list: List[str]) -> Dict[str, str]:
    """
    One OpenAI call. Returns {tag: block_text}.
    Only returns for tags provided.
    """
    if not tag_list:
        return {}

    tags_str = ", ".join(tag_list)

prompt = (
    "You are reading a PNG IMAGE CROP from a construction drawing finish schedule.\n"
    "This schedule lists multiple finish TAGS (e.g., AC-01, WC-10, PT-06F, CT-10) and their associated description text.\n"
    "Tags may appear as 'AC 01', 'AC-01', 'AC01', or similar. They may be inside a circle/box or plain text.\n"
    "The schedule may be arranged in one or more columns and may or may not have visible grid lines.\n\n"
    "TASK:\n"
    "For EACH requested tag below, find that tag in the image. Then extract the description text that belongs to THAT tag.\n"
    "The description text means: the nearby text that is clearly associated with that tag's entry (same row/line block/entry).\n"
    "Include ALL relevant text shown for that tag (any fields such as manufacturer, product, color, pattern, size, location, finish, notes, scale, etc.).\n"
    "Do NOT summarize or paraphrase — transcribe what you see as plain text.\n\n"
    "ASSOCIATION RULE (prevents mixing):\n"
    "Only use text that belongs to that tag's entry. Do NOT include text that belongs to a different tag.\n"
    "If the schedule has rows, stay within that row. If it is column blocks, stay within that tag’s block.\n\n"
    "DELETED RULE:\n"
    "Only return block_text=\"DELETED\" if the word DELETED appears clearly associated with that same tag’s entry.\n"
    "If not sure, do NOT guess DELETED.\n\n"
    "MISSING RULE:\n"
    "If you cannot confidently find the tag in this image crop, return block_text=\"\" for that tag.\n\n"
    f"REQUESTED TAGS (extract ONLY these): {tags_str}\n\n"
    "OUTPUT JSON ONLY (no extra text):\n"
    "{ \"items\": [ { \"tag\": \"AC-01\", \"block_text\": \"...\" }, ... ] }\n"
)



    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )

    raw = resp.output_text
    data = json.loads(raw)

    out: Dict[str, str] = {}
    items = data.get("items", []) if isinstance(data, dict) else []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            t = str(it.get("tag", "")).strip().upper()
            bt = str(it.get("block_text", "")).strip()
            if t:
                out[t] = bt
    return out


# -----------------------------
# Endpoint: extract schedule
# -----------------------------
@app.get("/extract-finish-schedule")
async def extract_finish_schedule(
    filename: str = Query(..., description="PDF file name inside the 'uploads' folder"),
    page_number: int = Query(..., ge=0, description="0-based page index"),
):
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found in uploads: {filename}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open PDF: {e}")

    if page_number >= doc.page_count:
        raise HTTPException(status_code=400, detail=f"Invalid page_number. PDF has {doc.page_count} pages.")

    page = doc[page_number]

    # 1) detect tag anchors (cheap)
    tag_rows = detect_tag_rows(page)
    if not tag_rows:
        return {"page": page_number, "num_blocks": 0, "blocks": []}

    # 2) build tiles (landscape defaults)
    tiles = make_tiles_for_landscape(page.rect, cols=3, rows=10, col_overlap=0.12, row_overlap=0.15)

    # Only keep tiles that contain at least one tag anchor (reduces OpenAI calls)
    tiles = [t for t in tiles if tile_contains_any_tags(t, tag_rows)]

    # Hard cap calls for safety; if too many, increase rows/cols intelligently later
    if len(tiles) > 22:
        tiles = tiles[:22]

    # 3) For each tile, run vision and merge results
    best_by_tag: Dict[str, Dict[str, Any]] = {}  # tag -> {text, score}

    for tile in tiles:
        tile_tags = tags_in_tile(tile, tag_rows)
        if not tile_tags:
            continue

        img_url = render_clip_as_data_url(page, tile, max_long_side_px=2200.0)

        try:
            extracted = vision_extract_for_tile(img_url, tile_tags)
        except Exception:
            extracted = {}

        for t in tile_tags:
            bt = extracted.get(t, "") if extracted else ""
            sc = score_block_text(bt)

            if t not in best_by_tag:
                best_by_tag[t] = {"text": bt, "score": sc}
            else:
                if sc > best_by_tag[t]["score"]:
                    best_by_tag[t] = {"text": bt, "score": sc}

    # 4) Build response blocks in the format Base44 expects
    blocks = []
    for r in tag_rows:
        tag = r["tag"]
        y = float(r["y_center"])
        blocks.append(
            {
                "tag": tag,
                "y_center": y,
                "region_top": y - 20,
                "region_bottom": y + 20,
                "block_no": 0,
                "block_text": (best_by_tag.get(tag, {}).get("text") or "").strip(),
            }
        )

    return {"page": page_number, "num_blocks": len(blocks), "blocks": blocks}
