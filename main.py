
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
    Handles both:
      - inline tags like "AC-01" / "PT10S"
      - stacked tags like "AC" above "01" (common in bubbles)
    Returns rows like:
      { tag, tag_x0, tag_y0, tag_x1, tag_y1, y_center }
    """
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
    words = [(float(x0), float(y0), float(x1), float(y1), (t or "").strip(), b, l, w)
             for (x0, y0, x1, y1, t, b, l, w) in words]
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))

    def clean_token(s: str) -> str:
        s = s.strip().upper()
        # remove common punctuation but keep letters/digits
        s = re.sub(r"[^A-Z0-9]", "", s)
        return s

    candidates: List[Dict[str, Any]] = []

    # ---------
    # A) Single-token tags: "AC-01" / "PT10S" / "WC02"
    # ---------
    for (x0, y0, x1, y1, t, *_rest) in words_sorted:
        ct = clean_token(t)
        if not ct:
            continue
        tag = normalize_tag_from_tokens([ct])
        if tag:
            candidates.append({
                "tag": tag,
                "tag_x0": x0,
                "tag_y0": y0,
                "tag_x1": x1,
                "tag_y1": y1,
                "y_center": (y0 + y1) / 2.0,
            })

    # ---------
    # B) Two-token inline tags: "AC" "01" on same line
    # ---------
    n = len(words_sorted)
    for i in range(n - 1):
        x0, y0, x1, y1, t1, *_ = words_sorted[i]
        x0b, y0b, x1b, y1b, t2, *_ = words_sorted[i + 1]

        ct1 = clean_token(t1)
        ct2 = clean_token(t2)
        if not ct1 or not ct2:
            continue

        yc1 = (y0 + y1) / 2.0
        yc2 = (y0b + y1b) / 2.0

        # same-ish baseline
        if abs(yc1 - yc2) <= 8.0 and (x0b - x1) <= 25.0:
            tag = normalize_tag_from_tokens([ct1, ct2])
            if tag:
                candidates.append({
                    "tag": tag,
                    "tag_x0": min(x0, x0b),
                    "tag_y0": min(y0, y0b),
                    "tag_x1": max(x1, x1b),
                    "tag_y1": max(y1, y1b),
                    "y_center": (yc1 + yc2) / 2.0,
                })

    # ---------
    # C) Stacked tags: "AC" above "01" (x-aligned, vertically separated)
    # ---------
    # Index words by rough x-center to find vertical pairs
    for i in range(n):
        x0a, y0a, x1a, y1a, ta, *_ = words_sorted[i]
        top = clean_token(ta)
        if not top or not re.fullmatch(r"[A-Z]{1,4}", top):
            continue

        xca = (x0a + x1a) / 2.0

        # look for a numeric/suffix token below it
        for j in range(i + 1, min(i + 25, n)):  # local search window
            x0b, y0b, x1b, y1b, tb, *_ = words_sorted[j]
            if y0b <= y1a:  # must be below
                continue

            bot = clean_token(tb)
            if not bot:
                continue

            # bottom should start with digits (01, 02F, 10S etc)
            if not re.fullmatch(r"\d{1,3}[A-Z]?", bot):
                continue

            xcb = (x0b + x1b) / 2.0

            # x-centers aligned (bubble stacks are very aligned)
            if abs(xca - xcb) > 18.0:
                continue

            # vertical gap not crazy (avoid pairing across rows)
            gap = y0b - y1a
            if gap < 0 or gap > 40.0:
                continue

            tag = normalize_tag_from_tokens([top, bot])
            if tag:
                candidates.append({
                    "tag": tag,
                    "tag_x0": min(x0a, x0b),
                    "tag_y0": min(y0a, y0b),
                    "tag_x1": max(x1a, x1b),
                    "tag_y1": max(y1a, y1b),
                    "y_center": ((y0a + y1a) / 2.0 + (y0b + y1b) / 2.0) / 2.0,
                })
                break  # found the stacked pair for this top token

    # ---------
    # D) Universal false-positive filter: keep only prefixes that repeat
    # (real schedules have many tags with same prefix: AC, WC, PT, CT, DGF...)
    # ---------
    # Count prefix frequency
    prefix_counts: Dict[str, int] = {}
    for c in candidates:
        prefix = c["tag"].split("-")[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    # Keep prefixes that appear at least 2 times
    kept = [c for c in candidates if prefix_counts.get(c["tag"].split("-")[0], 0) >= 2]

    # ---------
    # E) De-dup: keep the leftmost occurrence for each tag
    # (bubble tags tend to be furthest-left within the row)
    # ---------
    best: Dict[str, Dict[str, Any]] = {}
    for c in kept:
        t = c["tag"]
        if t not in best or c["tag_x0"] < best[t]["tag_x0"]:
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


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Tries to parse JSON even if wrapped in code fences or extra text.
    """
    if not text:
        raise ValueError("Empty response text")

    t = text.strip()

    # Strip common code fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # If still not pure JSON, extract first {...} block
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        t = m.group(0)

    return json.loads(t)


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
    data = safe_json_loads(raw)

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
    print(f"[detect_tag_rows] found {len(tag_rows)} tags on page {page_number}")
    if tag_rows:
        print("[detect_tag_rows] sample:", [r["tag"] for r in tag_rows[:15]])

    if not tag_rows:
        return {"page": page_number, "num_blocks": 0, "blocks": []}

    # 2) build tiles (landscape defaults)
    tiles = make_tiles_for_landscape(page.rect, cols=3, rows=10, col_overlap=0.12, row_overlap=0.15)

    # Only keep tiles that contain at least one tag anchor (reduces OpenAI calls)
    tag_tiles = [t for t in tiles if tile_contains_any_tags(t, tag_rows)]
    tiles = tag_tiles if tag_tiles else tiles[:6]  # fallback to a few tiles so we can see something


    # Hard cap calls for safety; if too many, increase rows/cols intelligently later
    if len(tiles) > 22:
        tiles = tiles[:22]

    # 3) For each tile, run vision and merge results
    best_by_tag: Dict[str, Dict[str, Any]] = {}  # tag -> {text, score}
    print(f"[tiles] using {len(tiles)} tiles on page {page_number}")
    print(f"[tags] detected {len(tag_rows)} tags on page {page_number}")

    for tile in tiles:
        tile_tags = tags_in_tile(tile, tag_rows)
        if not tile_tags:
            continue

        img_url = render_clip_as_data_url(page, tile, max_long_side_px=2200.0)

        try:
            extracted = vision_extract_for_tile(img_url, tile_tags)
        except Exception as e:
            print("❌ Vision extract failed:", repr(e))
            # (Optional) print the first few tags so we know what tile was being processed
            print("   tile tag sample:", tile_tags[:10])
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
