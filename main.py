
# main.py - Finish Schedule backend (image-tiling + vision extraction)

import numpy as np
import cv2
import math

import io

import os
import re
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
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

def render_page_png(doc: fitz.Document, page_number: int, max_width_px: int = 2200) -> Tuple[bytes, float]:
    """
    Renders a page to a PNG image. Returns (png_bytes, scale_factor_px_per_pdf_unit).
    scale_factor lets us convert pixel bboxes back to PDF coordinates.
    """
    page = doc.load_page(page_number)
    rect = page.rect

    # Scale so the rendered image is ~max_width_px wide
    scale = max_width_px / rect.width
    mat = fitz.Matrix(scale, scale)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    return png_bytes, scale


def b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


def px_bbox_to_pdf_rect(px_bbox: List[float], scale: float) -> fitz.Rect:
    """
    Convert pixel bbox [x0,y0,x1,y1] (in rendered image space) to PDF rect coords.
    """
    x0, y0, x1, y1 = px_bbox
    return fitz.Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale)
def vision_detect_rows(page_png_b64: str) -> List[Dict[str, Any]]:
    """
    Returns a list of rows: [{ "tag": "AC-01", "row_bbox_px": [x0,y0,x1,y1] }, ...]
    row_bbox_px MUST bound only that row (between horizontal grid lines).
    """
    prompt = (
        "You are looking at an IMAGE of a construction drawing finish schedule page.\n"
        "This page contains one or more schedule sections with grid lines.\n"
        "Each schedule ROW begins with a circular (or similar) row marker/bubble containing a tag label.\n\n"
        "TASK:\n"
        "1) Find every schedule row marker/bubble on the page.\n"
        "2) Read the tag label inside it (normalize to format PREFIX-NUMBER with optional suffix, e.g., AC-01, PT-06F).\n"
        "3) For each row marker, return a bounding box (pixel coords) that captures ONLY that row's content\n"
        "   between its horizontal grid lines, extending across the row to include description and location text.\n\n"
        "RULES:\n"
        "- Only include items that are clearly row markers (bubbles) starting a schedule row.\n"
        "- Ignore page titles, sheet numbers (like I-601), notes, or random text that is not a row marker.\n"
        "- If you are unsure a marker is a row marker, do not include it.\n"
        "- Return strict JSON only.\n\n"
        "OUTPUT JSON:\n"
        "{ \"rows\": [ { \"tag\": \"AC-01\", \"row_bbox_px\": [x0,y0,x1,y1] }, ... ] }\n"
    )

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{page_png_b64}"}
                ],
            }
        ],
    )

    raw = resp.output_text.strip()

    # Safe JSON parse (handles fences)
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Row detect: no JSON found. Head: {raw[:200]}")
        raw = m.group(0)

    data = json.loads(raw)
    return data.get("rows", [])
def vision_transcribe_rows(batch_imgs_b64: List[str]) -> List[Dict[str, str]]:
    """
    For each row image, return {tag, block_text}.
    Output list index must match input image index.
    """
    prompt = (
        "You are reading one or more IMAGE crops from a construction finish schedule.\n"
        "IMPORTANT: Each image shows exactly ONE schedule ROW.\n\n"
        "For EACH image i (starting at i=0 in the order provided):\n"
        "1) Read the row's TAG from the left-side marker/bubble.\n"
        "   Normalize to PREFIX-NUMBER with optional suffix (examples: AC-01, PT-06F, CT-10).\n"
        "2) Extract ONLY the description text that belongs to that SAME ROW.\n"
        "   Do NOT include text from any other row.\n"
        "3) Only output block_text=\"DELETED\" if the literal word DELETED appears in that row.\n"
        "   Otherwise never guess DELETED.\n"
        "4) If you cannot confidently read a tag, return tag=\"\" for that image.\n"
        "5) You MUST return exactly one JSON item for every image index i.\n\n"
        "Return STRICT JSON ONLY in this exact shape:\n"
        "{ \"items\": [ {\"i\": 0, \"tag\": \"AC-01\", \"block_text\": \"...\"}, {\"i\": 1, \"tag\": \"\", \"block_text\": \"\"}, ... ] }\n"
    )


    content = [{"type": "input_text", "text": prompt}]
    for img_b64 in batch_imgs_b64:
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"})

    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
    )

    data = safe_json_loads(resp.output_text)
    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue

        try:
            idx = int(it.get("i", -1))
        except Exception:
            idx = -1

        out.append({
            "i": idx,
            "tag": str(it.get("tag", "")).strip().upper(),
            "block_text": str(it.get("block_text", "")).strip()
        })
    return out

TAG_RE = re.compile(r"^([A-Z]{1,4})-(\d{1,3})([A-Z]?)$")

def normalize_tag(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().upper()
    # common OCR quirks: "AC 01", "AC01", "AC-01"
    s = re.sub(r"[\s_]+", "-", s)
    s = s.replace("--", "-")
    s = s.replace("–", "-").replace("—", "-")

    # if no dash but looks like letters+digits, insert dash
    m = re.match(r"^([A-Z]{1,4})-?(\d{1,3})([A-Z]?)$", s.replace("-", ""))
    if not m:
        # try simpler form like "AC-01F" etc already handled above
        m2 = re.match(r"^([A-Z]{1,4})-(\d{1,3})([A-Z]?)$", s)
        if not m2:
            return ""
        return f"{m2.group(1)}-{int(m2.group(2)):02d}{m2.group(3)}".rstrip()
    prefix, num, suf = m.group(1), m.group(2), m.group(3)
    return f"{prefix}-{int(num):02d}{suf}".rstrip()

def is_valid_tag(tag: str) -> bool:
    if not tag:
        return False
    if tag in {"DELETED", "DELETE", "VOID", "N/A", "NA"}:
        return False
    return bool(TAG_RE.match(tag))

def looks_like_finish_row_text(txt: str) -> bool:
    """
    Universal-ish sanity: most finish rows contain at least one of these tokens.
    Prevents headers/notes/addresses from being treated as row descriptions.
    """
    if not txt:
        return False
    t = txt.upper()
    return any(k in t for k in ["MFR", "MANUF", "PROD", "PRODUCT", "COLOR", "FINISH", "SIZE", "LOC", "LOCATION", "PATT", "PATTERN", "STYLE", "SERIES", "SKU", "COL:"])


def vision_transcribe_grid_rows(tag_imgs_b64: List[str], desc_imgs_b64: List[str]) -> List[Dict[str, str]]:
    """
    Each row has TWO images:
      - tag cell crop (left column)
      - description crop (rest of row)
    Returns list of {i, tag, block_text} where i aligns with the input order.
    """
    assert len(tag_imgs_b64) == len(desc_imgs_b64)

    prompt = (
    "You are reading multiple ROWS from a construction finish schedule.\n"
    "Each ROW i is provided as TWO images:\n"
    "  - Image A(i): the TAG cell (left column)\n"
    "  - Image B(i): the DESCRIPTION area for the SAME ROW\n\n"
    "For each row i:\n"
    "1) Read the finish TAG from Image A(i). It will look like letters + numbers (sometimes with a suffix), e.g. AC-01, PT-06F, CT-10.\n"
    "   - The TAG is NEVER the word 'DELETED'.\n"
    "   - If you do not see a real tag pattern in Image A(i), return tag=\"\".\n"
    "2) Read the description text from Image B(i). Keep ONLY what belongs to that row.\n"
    "3) Only return block_text=\"DELETED\" if the literal word DELETED appears in the DESCRIPTION for that same row.\n"
    "4) You MUST return exactly one item for every row index i.\n\n"
    "Return STRICT JSON ONLY:\n"
    "{ \"items\": [ {\"i\": 0, \"tag\": \"AC-01\", \"block_text\": \"...\"}, ... ] }\n"
)


    content = [{"type": "input_text", "text": prompt}]
    for i, (a, b) in enumerate(zip(tag_imgs_b64, desc_imgs_b64)):
        content.append({"type": "input_text", "text": f"ROW {i}: Image A then Image B"})
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{a}"})
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b}"})

    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "row_transcription",
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "i": {"type": "integer"},
                                    "tag": {"type": "string"},
                                    "block_text": {"type": "string"}
                                },
                                "required": ["i", "tag", "block_text"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

# With response_format json_schema, output_text should be strict JSON
data = json.loads(resp.output_text)
if not isinstance(data, dict) or "items" not in data:
    return []

    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("i", -1))
        except Exception:
            idx = -1
        
        raw_tag = str(it.get("tag", "")).strip()
        tag = normalize_tag(raw_tag)

        block_text = str(it.get("block_text", "")).strip()

        # If the model mistakenly returns DELETED as the tag, drop it
        if not is_valid_tag(tag):
            out.append({"i": idx, "tag": "", "block_text": ""})
            continue

        # Only allow DELETED when the DESCRIPTION actually indicates it
        if block_text.upper() == "DELETED":
            # accept deleted rows only for valid tags
            out.append({"i": idx, "tag": tag, "block_text": "DELETED"})
            continue

        # If row text doesn't resemble finish info, treat it as empty
        if not looks_like_finish_row_text(block_text):
            out.append({"i": idx, "tag": "", "block_text": ""})
            continue

        out.append({"i": idx, "tag": tag, "block_text": block_text})

    return out


def png_bytes_to_cv2(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode PNG to image")
    return img

def crop_png_bytes(png_bytes: bytes, bbox: List[int]) -> bytes:
    """
    Crop a rendered page PNG using pixel bbox [x0,y0,x1,y1] and return PNG bytes.
    """
    img = png_bytes_to_cv2(png_bytes)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w-1, int(x0)))
    x1 = max(0, min(w,   int(x1)))
    y0 = max(0, min(h-1, int(y0)))
    y1 = max(0, min(h,   int(y1)))
    if x1 <= x0 or y1 <= y0:
        return b""
    crop = img[y0:y1, x0:x1]
    ok, out = cv2.imencode(".png", crop)
    return out.tobytes() if ok else b""


def detect_horizontal_lines(img: np.ndarray) -> List[int]:
    """
    Returns y-positions of horizontal grid lines in pixel coords.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert to make lines bright
    inv = 255 - gray

    # Threshold
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology to extract horizontal lines
    h, w = bw.shape
    kernel_len = max(20, w // 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    horiz = cv2.erode(bw, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

    # Find contours -> y positions
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww > w * 0.15:  # ignore tiny line fragments
            ys.append(y)

    ys = sorted(ys)
    # Deduplicate near-duplicates
    dedup = []
    for y in ys:
        if not dedup or abs(y - dedup[-1]) > 4:
            dedup.append(y)
    return dedup
def detect_vertical_lines(img: np.ndarray) -> List[int]:
    """
    Return x positions (pixels) of detected vertical table grid lines.
    Works best on schedules with visible column lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = bw.shape

    # Long vertical kernel to extract column lines
    kernel_len = max(25, h // 25)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    vert = cv2.erode(bw, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        # filter out short junk lines
        if hh > h * 0.20:
            xs.append(x)

    xs = sorted(xs)
    # de-duplicate close x positions
    dedup = []
    for x in xs:
        if not dedup or abs(x - dedup[-1]) > 8:
            dedup.append(x)

    # also add right-most boundary if missing (makes cropping safer)
    if dedup and (w - dedup[-1] > 20):
        dedup.append(w - 1)

    return dedup

def detect_bubbles(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect circular bubbles. Returns list of {x,y,r} in pixel coords.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # HoughCircles is sensitive; we’ll tune parameters safely.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,        # min distance between bubble centers
        param1=120,        # edge detection threshold
        param2=30,         # circle center detection threshold
        minRadius=10,
        maxRadius=40
    )

    out = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            out.append({"x": int(x), "y": int(y), "r": int(r)})
    return out
def build_row_boxes(img: np.ndarray, bubbles: List[Dict[str, Any]], hlines: List[int]) -> List[List[int]]:
    """
    For each bubble, find nearest horizontal lines above/below to form a row box:
    [x0,y0,x1,y1] in pixels.
    """
    h, w = img.shape[:2]
    row_boxes = []

    if len(hlines) < 4 or len(bubbles) < 5:
        return row_boxes

    # Sort bubbles top-to-bottom
    bubbles_sorted = sorted(bubbles, key=lambda b: b["y"])

    # Estimate left boundary based on bubble centers (bubble column)
    bubble_xs = [b["x"] for b in bubbles_sorted]
    left_col_x = int(np.percentile(bubble_xs, 15))  # left-ish bubble column

    # Row right boundary: we don’t know exact table border yet; use 95% width for now
    right_x = int(w * 0.95)

    for b in bubbles_sorted:
        y = b["y"]

        # find nearest line above and below
        above = [ly for ly in hlines if ly < y]
        below = [ly for ly in hlines if ly > y]
        if not above or not below:
            continue

        y0 = above[-1]
        y1 = below[0]

        # sanity: ignore huge spans (this is what caused your “massive” blocks)
        if (y1 - y0) > 180:  # tuned; we can adjust based on your sheet DPI
            continue
        if (y1 - y0) < 18:
            continue

        x0 = max(0, left_col_x - 5)
        row_boxes.append([x0, y0, right_x, y1])

    # Deduplicate very similar boxes
    row_boxes = sorted(row_boxes, key=lambda bb: (bb[1], bb[0]))
    dedup = []
    for bb in row_boxes:
        if not dedup or abs(bb[1] - dedup[-1][1]) > 6:
            dedup.append(bb)
    return dedup


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

def detect_vertical_lines(img: np.ndarray) -> List[int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = bw.shape
    kernel_len = max(20, h // 30)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    vert = cv2.erode(bw, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh > h * 0.15:
            xs.append(x)

    xs = sorted(xs)
    dedup = []
    for x in xs:
        if not dedup or abs(x - dedup[-1]) > 6:
            dedup.append(x)
    return dedup


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
@app.get("/extract-finish-schedule-vision")
def extract_finish_schedule_vision(
    filename: str = Query(...),
    page_number: int = Query(..., ge=0),
):
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        doc = fitz.open(pdf_path)
        if page_number >= len(doc):
            raise HTTPException(status_code=400, detail="page_number out of range")

        # 1) Render page image
        page_png, scale = render_page_png(doc, page_number, max_width_px=2200)
        page_b64 = b64_png(page_png)

        rows = []  # always define to avoid UnboundLocalError
        row_tag_cells = []
        row_desc_cells = []
        row_bboxes = []

        # 2) Detect grid lines (this is the core of "perfect")
        page_img = png_bytes_to_cv2(page_png)
        hlines = detect_horizontal_lines(page_img)
        vlines = detect_vertical_lines(page_img)

        h, w = page_img.shape[:2]
        print(f"[grid] hlines={len(hlines)} vlines={len(vlines)}")

        # If grid detection is weak, fallback to old method (rare case)
        if len(hlines) < 6 or len(vlines) < 4:
            bubbles = detect_bubbles(page_img)
            row_boxes = build_row_boxes(page_img, bubbles, hlines)
            print(f"[fallback_rows] bubbles={len(bubbles)} row_boxes={len(row_boxes)}")

            rows = []
            for bb in row_boxes:
                x0, y0, x1, y1 = bb
                rows.append({"row_bbox_px": [x0, y0, x1, y1]})

            # For fallback, we will treat each row bbox as the whole row image
            row_imgs_b64 = []
            row_bboxes = []
            for r in rows:
                bbox_px = r.get("row_bbox_px")
                row_png = crop_png_bytes(page_png, bbox_px)
                if not row_png:
                    continue
                row_imgs_b64.append(b64_png(row_png))
                row_bboxes.append(bbox_px)

            # We'll transcribe from whole-row crops in fallback mode
            mode = "fallback"

        else:
            # GRID MODE: rows = each gap between adjacent horizontal lines
            # Choose TAG column as the narrow column near the left (most schedules)
            # Candidate columns are between vlines[j]..vlines[j+1]
            col_ranges = []
            for j in range(len(vlines) - 1):
                x0 = vlines[j]
                x1 = vlines[j + 1]
                col_ranges.append((j, x0, x1, x1 - x0))

            # Pick the earliest narrow column as the tag column
            # (narrow and near left side is almost always the tag)
            col_ranges_sorted = sorted(col_ranges, key=lambda t: (t[1], t[3]))
            tag_col_j, tag_x0, tag_x1, _ = col_ranges_sorted[0]

            # Description region starts at the next vertical line after tag column
            desc_x0 = tag_x1
            desc_x1 = w - 1

            print(f"[grid_cols] tag_col={tag_col_j} tag_x0={tag_x0} tag_x1={tag_x1} desc_x0={desc_x0}")

            # Build row crops: (tag cell crop, description region crop) per row
            row_bboxes = []  # store [x0,y0,x1,y1] of full row
            row_tag_cells = []
            row_desc_cells = []

            # Use each band between horizontal lines as a row
            for i in range(len(hlines) - 1):
                y0 = hlines[i]
                y1 = hlines[i + 1]
                if y1 - y0 < 10:
                    continue

                # full row bbox (for returning y_center etc.)
                full_bbox = [0, y0, w - 1, y1]

                # tag cell bbox and desc bbox
                tag_bbox = [tag_x0, y0, tag_x1, y1]
                desc_bbox = [desc_x0, y0, desc_x1, y1]

                tag_png = crop_png_bytes(page_png, tag_bbox)
                desc_png = crop_png_bytes(page_png, desc_bbox)

                if not tag_png or not desc_png:
                    continue

                # Optional: downscale desc to reduce memory
                desc_img = png_bytes_to_cv2(desc_png)
                dh, dw = desc_img.shape[:2]
                max_dw = 1400
                if dw > max_dw:
                    s = max_dw / float(dw)
                    desc_img = cv2.resize(desc_img, (int(dw * s), int(dh * s)), interpolation=cv2.INTER_AREA)
                    ok, out = cv2.imencode(".png", desc_img)
                    if ok:
                        desc_png = out.tobytes()

                row_tag_cells.append(b64_png(tag_png))
                row_desc_cells.append(b64_png(desc_png))
                row_bboxes.append(full_bbox)

            mode = "grid"
            print(f"[grid_rows] rows={len(row_bboxes)}")

        # 3) Transcribe rows
        tag_to_text: Dict[str, str] = {}
        tag_to_bbox: Dict[str, List[int]] = {}

        if mode == "grid":
            # Use paired (tag cell + desc cell) reading
            BATCH = 4  # keep small to avoid memory spikes
            n = len(row_bboxes)

            for i0 in range(0, n, BATCH):
                batch_tag_imgs = row_tag_cells[i0:i0 + BATCH]
                batch_desc_imgs = row_desc_cells[i0:i0 + BATCH]

                items = vision_transcribe_grid_rows(batch_tag_imgs, batch_desc_imgs)

                for it in items:
                    idx = it.get("i", -1)
                    tag = (it.get("tag") or "").strip().upper()
                    txt = (it.get("block_text") or "").strip()

                    if not isinstance(idx, int):
                        continue
                    real_idx = i0 + idx
                    if real_idx < 0 or real_idx >= n:
                        continue
                    if not tag:
                        continue

                    if tag not in tag_to_text or len(txt) > len(tag_to_text[tag]):
                        tag_to_text[tag] = txt
                        tag_to_bbox[tag] = row_bboxes[real_idx]

        else:
            # fallback: single row image reading
            # IMPORTANT: in fallback, row_imgs_b64 must exist (you build it in fallback branch)
            BATCH = 6
            n = len(row_imgs_b64)

            for i0 in range(0, n, BATCH):
                batch_imgs = row_imgs_b64[i0:i0 + BATCH]
                items = vision_transcribe_rows(batch_imgs)

                for it in items:
                    idx = it.get("i", -1)
                    tag = (it.get("tag") or "").strip().upper()
                    txt = (it.get("block_text") or "").strip()

                    if not isinstance(idx, int):
                        continue
                    real_idx = i0 + idx
                    if real_idx < 0 or real_idx >= len(row_bboxes):
                        continue
                    if not tag:
                        continue

                    if tag not in tag_to_text or len(txt) > len(tag_to_text[tag]):
                        tag_to_text[tag] = txt
                        tag_to_bbox[tag] = row_bboxes[real_idx]


  


        # 4) Batch transcribe
        tag_to_text: Dict[str, str] = {}
        tag_to_bbox: Dict[str, List[int]] = {}

        BATCH = 4  # two images per row; keep batch small
        n = len(row_bboxes)

        for i0 in range(0, n, BATCH):
            batch_tag_imgs = row_tag_cells[i0:i0 + BATCH]
            batch_desc_imgs = row_desc_cells[i0:i0 + BATCH]

            items = vision_transcribe_grid_rows(batch_tag_imgs, batch_desc_imgs)

            for it in items:
                idx = it.get("i", -1)
                tag = (it.get("tag") or "").strip().upper()
                txt = (it.get("block_text") or "").strip()

                if not isinstance(idx, int):
                    continue
                real_idx = i0 + idx
                if real_idx < 0 or real_idx >= n:
                    continue
                if not tag:
                    continue

                # keep best (prefer longer non-empty)
                if tag not in tag_to_text or len(txt) > len(tag_to_text[tag]):
                    tag_to_text[tag] = txt
                    tag_to_bbox[tag] = row_bboxes[real_idx]



        # 5) Build response
        blocks = []
        for tag, txt in tag_to_text.items():
            bbox_px = tag_to_bbox.get(tag, [0, 0, 0, 0])
            y_center = (bbox_px[1] + bbox_px[3]) / 2.0
            blocks.append({
                "tag": tag,
                "y_center": y_center,
                "region_top": bbox_px[1],
                "region_bottom": bbox_px[3],
                "block_no": 0,
                "block_text": txt
            })

        blocks.sort(key=lambda b: (b["y_center"], b["tag"]))



        return JSONResponse({
            "page": page_number,
            "num_blocks": len(blocks),
            "blocks": blocks
        })

    except Exception as e:
        print("❌ extract_finish_schedule_vision error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
