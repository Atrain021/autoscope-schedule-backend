# main.py - AutoScope Schedule Backend (Production Ready) 

import os
import re
import gc
import json
import uuid
import base64
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel, Field


import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

import re
import json
import base64
import pdfplumber

# ----------------------------
# DISCIPLINE + DRAWING TAXONOMY (Phase 1)
# ----------------------------

DISCIPLINES = [
    "ARCHITECTURAL",
    "INTERIOR_DESIGN",
    "STRUCTURAL",
    "CIVIL",
    "MECHANICAL",
    "ELECTRICAL",
    "PLUMBING",
    "FIRE_PROTECTION",
    "FIRE_ALARM_LOW_VOLTAGE",
    "LANDSCAPE",
    "UNKNOWN",
]

# Shared drawing types (appear in many disciplines)
COMMON_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX",
    "GENERAL_NOTES",
    "LEGENDS_SYMBOLS_ABBREVIATIONS",
    "CODE_LIFE_SAFETY",
    "DETAILS",
    "SECTIONS",
    "ELEVATIONS",
    "SCHEDULES",
    "DIAGRAMS",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Architectural / Interior Design types (your existing list, refined)
ARCH_ID_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "CODE_LIFE_SAFETY",
    "SITE_PLAN",
    "FLOOR_PLAN",
    "ENLARGED_PLAN",
    "REFLECTED_CEILING_PLAN",
    "ROOF_PLAN",
    "LIFE_SAFETY_EGRESS_PLAN",
    "ELEVATIONS",
    "SECTIONS",
    "DETAILS",
    "WALL_TYPES_ASSEMBLIES",
    "DOOR_WINDOW_SCHEDULES",
    "ROOM_FINISH_SCHEDULES",
    "FINISH_LEGENDS_TAG_GLOSSARY",
    "OTHER_SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Structural types
STRUCTURAL_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "GENERAL_STRUCTURAL_NOTES",
    "FOUNDATION_PLAN",
    "FRAMING_PLAN",
    "ROOF_FRAMING_PLAN",
    "SLAB_ON_GRADE_PLAN",
    "COLUMN_SCHEDULE",
    "BEAM_SCHEDULE",
    "STEEL_CONNECTION_DETAILS",
    "CONCRETE_DETAILS",
    "REBAR_DETAILS",
    "SECTIONS",
    "DETAILS",
    "GENERAL_DETAILS",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Civil types
CIVIL_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "EXISTING_CONDITIONS_PLAN",
    "DEMOLITION_PLAN",
    "SITE_PLAN",
    "GRADING_PLAN",
    "DRAINAGE_PLAN",
    "UTILITY_PLAN",
    "STORMWATER_MANAGEMENT",
    "EROSION_SEDIMENT_CONTROL",
    "SITE_DETAILS",
    "PROFILES_SECTIONS",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Mechanical types
MECHANICAL_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "MECHANICAL_NOTES",
    "HVAC_FLOOR_PLAN",
    "EQUIPMENT_PLAN",
    "ROOF_MECHANICAL_PLAN",
    "DUCTWORK_PLAN",
    "PIPING_PLAN",
    "SCHEMATICS_DIAGRAMS",
    "CONTROLS_DIAGRAMS",
    "DETAILS",
    "SECTIONS",
    "SCHEDULES",
    "EQUIPMENT_SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Electrical types
ELECTRICAL_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "ELECTRICAL_NOTES",
    "POWER_PLAN",
    "LIGHTING_PLAN",
    "LIGHTING_CONTROL_PLAN",
    "ONE_LINE_DIAGRAM",
    "RISER_DIAGRAM",
    "PANEL_SCHEDULES",
    "DETAILS",
    "SECTIONS",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Plumbing types
PLUMBING_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "PLUMBING_NOTES",
    "PLUMBING_FLOOR_PLAN",
    "DOMESTIC_WATER_PLAN",
    "SANITARY_PLAN",
    "STORM_PLAN",
    "GAS_PLAN",
    "RISER_DIAGRAM",
    "ISOMETRICS",
    "DETAILS",
    "SECTIONS",
    "SCHEDULES",
    "FIXTURE_SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Fire Protection (sprinkler) types
FIRE_PROTECTION_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "FIRE_PROTECTION_NOTES",
    "SPRINKLER_FLOOR_PLAN",
    "STANDPIPE_PLAN",
    "RISER_DIAGRAM",
    "DETAILS",
    "SECTIONS",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Fire Alarm / Low Voltage types (optional discipline bucket)
FIRE_ALARM_LOW_VOLTAGE_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "LOW_VOLTAGE_NOTES",
    "FIRE_ALARM_PLAN",
    "SECURITY_PLAN",
    "DATA_TELECOM_PLAN",
    "RISER_DIAGRAM",
    "SINGLE_LINE_DIAGRAM",
    "DEVICE_SCHEDULES",
    "DETAILS",
    "SECTIONS",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

# Landscape types (optional)
LANDSCAPE_SHEET_TYPES = [
    "COVER_SHEET",
    "SHEET_INDEX_GENERAL_NOTES",
    "PLANTING_PLAN",
    "IRRIGATION_PLAN",
    "LANDSCAPE_DETAILS",
    "GRADING_PLAN",
    "SCHEDULES",
    "SPECIFICATIONS_NOTES",
    "OTHER",
]

ALLOWED_TYPES_BY_DISCIPLINE = {
    "ARCHITECTURAL": set(ARCH_ID_SHEET_TYPES),
    "INTERIOR_DESIGN": set(ARCH_ID_SHEET_TYPES),
    "STRUCTURAL": set(STRUCTURAL_SHEET_TYPES),
    "CIVIL": set(CIVIL_SHEET_TYPES),
    "MECHANICAL": set(MECHANICAL_SHEET_TYPES),
    "ELECTRICAL": set(ELECTRICAL_SHEET_TYPES),
    "PLUMBING": set(PLUMBING_SHEET_TYPES),
    "FIRE_PROTECTION": set(FIRE_PROTECTION_SHEET_TYPES),
    "FIRE_ALARM_LOW_VOLTAGE": set(FIRE_ALARM_LOW_VOLTAGE_SHEET_TYPES),
    "LANDSCAPE": set(LANDSCAPE_SHEET_TYPES),
    "UNKNOWN": set(COMMON_SHEET_TYPES),
}


def _clean_json_text(text: str) -> str:
    """Remove ```json fences if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r'^```[a-zA-Z]*\n', '', t)
        t = re.sub(r'\n```$', '', t)
        t = t.strip()
    return t


def normalize_sheet_type(discipline: str, value: str) -> str:
    """
    Normalize raw model output to a valid type for the discipline.
    """
    v = (value or "").strip().upper()
    v = v.replace(" ", "_").replace("-", "_")

    # Cross-discipline aliases (common)
    aliases = {
        "COVER": "COVER_SHEET",
        "COVER_SHEET": "COVER_SHEET",
        "SHEET_INDEX": "SHEET_INDEX_GENERAL_NOTES",
        "GENERAL_NOTES": "SHEET_INDEX_GENERAL_NOTES",
        "NOTES": "SPECIFICATIONS_NOTES",
        "DETAIL": "DETAILS",
        "DETAILS": "DETAILS",
        "SECTION": "SECTIONS",
        "SECTIONS": "SECTIONS",
        "ELEVATION": "ELEVATIONS",
        "ELEVATIONS": "ELEVATIONS",
        "SCHEDULE": "SCHEDULES",
        "SCHEDULES": "SCHEDULES",
        "DIAGRAM": "DIAGRAMS",
        "DIAGRAMS": "DIAGRAMS",
    }

    # Arch/ID-specific aliases
    if discipline in ["ARCHITECTURAL", "INTERIOR_DESIGN"]:
        aliases.update({
            "FINISH_SCHEDULE": "ROOM_FINISH_SCHEDULES",
            "FINISH_SCHEDULES": "ROOM_FINISH_SCHEDULES",
            "ROOM_FINISH_SCHEDULE": "ROOM_FINISH_SCHEDULES",
            "RFS": "ROOM_FINISH_SCHEDULES",
            "DOOR_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
            "WINDOW_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
            "DOOR_WINDOW_SCHEDULE": "DOOR_WINDOW_SCHEDULES",
            "PLAN": "FLOOR_PLAN",
            "WALL_TYPES": "WALL_TYPES_ASSEMBLIES",
            "ASSEMBLY_TYPES": "WALL_TYPES_ASSEMBLIES",
            "FINISH_LEGEND": "FINISH_LEGENDS_TAG_GLOSSARY",
            "TAG_GLOSSARY": "FINISH_LEGENDS_TAG_GLOSSARY",
        })

    if v in aliases:
        v = aliases[v]

    allowed = ALLOWED_TYPES_BY_DISCIPLINE.get(discipline, ALLOWED_TYPES_BY_DISCIPLINE["UNKNOWN"])
    return v if v in allowed else "OTHER"

def get_type_list_for_discipline(discipline: str) -> List[str]:
    """
    Returns the allowed type list (as a stable, sorted list) for the given discipline.
    """
    allowed = ALLOWED_TYPES_BY_DISCIPLINE.get(discipline) or ALLOWED_TYPES_BY_DISCIPLINE["UNKNOWN"]
    return sorted(list(allowed))


def _extract_sheet_id_title(image_bytes: bytes) -> Dict[str, str]:
    """
    Pass 1: Extract sheet_identifier and sheet_title only.
    Keeps this focused so the model doesn’t hallucinate.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """You are looking at ONE page from a construction drawing PDF.

Extract ONLY these fields if visible:
- sheet_identifier (examples: A-101, I-201, S-101, M-301, E-601, P-201, C-101, FP-101, FA-101, L-101)
- sheet_title (examples: "LEVEL 1 FLOOR PLAN", "PANEL SCHEDULES", etc.)

Return ONLY valid JSON:
{
  "sheet_identifier": "A-101",
  "sheet_title": "LEVEL 1 FLOOR PLAN"
}

Rules:
- Do NOT output markdown.
- If not clearly visible, return "" for that field.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
            ],
        }],
        max_tokens=250,
        temperature=0.0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    cleaned = _clean_json_text(raw)
    data = json.loads(cleaned)

    return {
        "sheet_identifier": (data.get("sheet_identifier") or "").strip(),
        "sheet_title": (data.get("sheet_title") or "").strip(),
    }


def _classify_type_for_discipline(image_bytes: bytes, discipline: str) -> str:
    """
    Pass 2: Classify the page using the discipline-specific type list.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    type_list = ", ".join(get_type_list_for_discipline(discipline))

    prompt = f"""You are looking at ONE page from a construction drawing PDF.

Discipline has been determined as: {discipline}

Classify this page into exactly ONE of these types:
{type_list}

Return ONLY valid JSON:
{{ "classification": "TYPE_HERE" }}

Rules:
- Do NOT output markdown.
- Output must be one of the listed types exactly.
- Choose the most specific type available.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
            ],
        }],
        max_tokens=150,
        temperature=0.0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    cleaned = _clean_json_text(raw)
    data = json.loads(cleaned)

    return (data.get("classification") or "").strip()


def override_by_title(discipline: str, sheet_title: str, sheet_id: str, base_type: str) -> str:
    """
    Deterministic overrides based on title/id.
    Discipline-aware so 'DETAILS' doesn't drown out everything.
    """
    title = (sheet_title or "").strip().upper()
    sid = (sheet_id or "").strip().upper()

    if not title and not sid:
        return base_type

    # Universal high-confidence
    if "COVER" in title:
        return "COVER_SHEET"
    if any(k in title for k in ["SHEET INDEX", "GENERAL NOTES", "ABBREVIATIONS", "LEGEND", "SYMBOLS"]):
        return "SHEET_INDEX_GENERAL_NOTES"

    # ARCH/ID overrides (your existing logic, kept)
    if discipline in ["ARCHITECTURAL", "INTERIOR_DESIGN"]:
        if any(k in title for k in ["EGRESS", "EXIT", "LIFE SAFETY", "LIFE-SAFETY", "EVACUATION"]):
            return "LIFE_SAFETY_EGRESS_PLAN"

        if any(k in title for k in ["CODE", "OCCUPANCY", "UNIT MIX", "FIRE RATING", "ACCESSIBILITY", "ADA", "IBC", "NFPA"]):
            return "CODE_LIFE_SAFETY"

        if any(k in title for k in ["FINISH SCHEDULE", "ROOM FINISH", "FINISH LEGEND", "FINISH PLAN LEGEND"]):
            return "ROOM_FINISH_SCHEDULES"

        if any(k in title for k in ["DOOR SCHEDULE", "WINDOW SCHEDULE", "STOREFRONT SCHEDULE", "FRAME SCHEDULE"]):
            return "DOOR_WINDOW_SCHEDULES"

        if any(k in title for k in ["REFLECTED CEILING", "RCP", "CEILING PLAN"]):
            return "REFLECTED_CEILING_PLAN"

        if any(k in title for k in ["ROOF PLAN", "PENTHOUSE ROOF", "ROOF"]):
            if "DETAIL" not in title and "SECTION" not in title:
                return "ROOF_PLAN"

        if any(k in title for k in ["SITE PLAN", "GRADING", "UTILITY PLAN", "EROSION", "SEDIMENT"]):
            return "SITE_PLAN"

        if "ELEVATION" in title:
            return "ELEVATIONS"
        if "SECTION" in title:
            return "SECTIONS"
        if "DETAIL" in title:
            return "DETAILS"
        if any(k in title for k in ["WALL TYPE", "PARTITION TYPE", "ASSEMBLY", "TYPICAL WALL"]):
            return "WALL_TYPES_ASSEMBLIES"
        if "ENLARGED" in title or any(k in title for k in ["TOILET ROOM PLAN", "KITCHEN PLAN", "BATHROOM PLAN"]):
            return "ENLARGED_PLAN"

        return base_type

    # STRUCTURAL overrides
    if discipline == "STRUCTURAL":
        if any(k in title for k in ["FOUNDATION", "FOOTING", "PILE", "CAISSON"]):
            return "FOUNDATION_PLAN"
        if any(k in title for k in ["FRAMING", "FRAMING PLAN"]):
            return "FRAMING_PLAN"
        if any(k in title for k in ["ROOF FRAMING"]):
            return "ROOF_FRAMING_PLAN"
        if any(k in title for k in ["SLAB", "SOG", "SLAB ON GRADE"]):
            return "SLAB_ON_GRADE_PLAN"
        if any(k in title for k in ["CONNECTION", "STEEL CONNECTION"]):
            return "STEEL_CONNECTION_DETAILS"
        if any(k in title for k in ["REBAR", "REINFORCING"]):
            return "REBAR_DETAILS"
        if any(k in title for k in ["CONCRETE DETAIL", "CONCRETE DETAILS"]):
            return "CONCRETE_DETAILS"
        if any(k in title for k in ["SCHEDULE"]):
            return "SCHEDULES"
        if "DETAIL" in title:
            return "DETAILS"
        if "SECTION" in title:
            return "SECTIONS"
        return base_type

    # MECHANICAL overrides
    if discipline == "MECHANICAL":
        if any(k in title for k in ["HVAC", "MECHANICAL PLAN", "AIRFLOW"]):
            return "HVAC_FLOOR_PLAN"
        if any(k in title for k in ["DUCT", "DUCTWORK"]):
            return "DUCTWORK_PLAN"
        if any(k in title for k in ["PIPING", "HYDRONIC", "CHW", "HHW", "REFRIGERANT"]):
            return "PIPING_PLAN"
        if any(k in title for k in ["DIAGRAM", "SCHEMATIC", "RISER"]):
            return "SCHEMATICS_DIAGRAMS"
        if "SCHEDULE" in title:
            return "EQUIPMENT_SCHEDULES"
        if "DETAIL" in title:
            return "DETAILS"
        if "SECTION" in title:
            return "SECTIONS"
        return base_type

    # ELECTRICAL overrides
    if discipline == "ELECTRICAL":
        if any(k in title for k in ["POWER"]):
            return "POWER_PLAN"
        if any(k in title for k in ["LIGHTING"]):
            return "LIGHTING_PLAN"
        if any(k in title for k in ["ONE-LINE", "ONE LINE", "SINGLE LINE"]):
            return "ONE_LINE_DIAGRAM"
        if any(k in title for k in ["RISER"]):
            return "RISER_DIAGRAM"
        if any(k in title for k in ["PANEL SCHEDULE"]):
            return "PANEL_SCHEDULES"
        if "DETAIL" in title:
            return "DETAILS"
        if "SECTION" in title:
            return "SECTIONS"
        if "SCHEDULE" in title:
            return "SCHEDULES"
        return base_type

    # PLUMBING overrides
    if discipline == "PLUMBING":
        if any(k in title for k in ["DOMESTIC", "WATER"]):
            return "DOMESTIC_WATER_PLAN"
        if any(k in title for k in ["SANITARY"]):
            return "SANITARY_PLAN"
        if any(k in title for k in ["STORM"]):
            return "STORM_PLAN"
        if any(k in title for k in ["GAS"]):
            return "GAS_PLAN"
        if any(k in title for k in ["RISER"]):
            return "RISER_DIAGRAM"
        if "SCHEDULE" in title:
            return "FIXTURE_SCHEDULES"
        if "DETAIL" in title:
            return "DETAILS"
        if "SECTION" in title:
            return "SECTIONS"
        return base_type

    # CIVIL overrides
    if discipline == "CIVIL":
        if any(k in title for k in ["EXISTING"]):
            return "EXISTING_CONDITIONS_PLAN"
        if any(k in title for k in ["DEMOLITION", "DEMO"]):
            return "DEMOLITION_PLAN"
        if any(k in title for k in ["GRADING"]):
            return "GRADING_PLAN"
        if any(k in title for k in ["DRAINAGE"]):
            return "DRAINAGE_PLAN"
        if any(k in title for k in ["UTILITY"]):
            return "UTILITY_PLAN"
        if any(k in title for k in ["EROSION", "SEDIMENT"]):
            return "EROSION_SEDIMENT_CONTROL"
        if "DETAIL" in title:
            return "SITE_DETAILS"
        if "PROFILE" in title or "SECTION" in title:
            return "PROFILES_SECTIONS"
        if "SCHEDULE" in title:
            return "SCHEDULES"
        return base_type

    # FIRE PROTECTION overrides
    if discipline == "FIRE_PROTECTION":
        if any(k in title for k in ["SPRINKLER", "FIRE PROTECTION"]):
            return "SPRINKLER_FLOOR_PLAN"
        if any(k in title for k in ["STANDPIPE"]):
            return "STANDPIPE_PLAN"
        if any(k in title for k in ["RISER"]):
            return "RISER_DIAGRAM"
        if "DETAIL" in title:
            return "DETAILS"
        if "SECTION" in title:
            return "SECTIONS"
        if "SCHEDULE" in title:
            return "SCHEDULES"
        return base_type

    return base_type


import re


def infer_discipline_from_sheet_id(sheet_id: str | None) -> str:
    """
    Robust discipline inference from sheet identifier.
    Examples:
      A-101 -> ARCHITECTURAL
      I-201 / ID-201 -> INTERIOR_DESIGN
      S-101 -> STRUCTURAL
      C-101 -> CIVIL
      M-101 -> MECHANICAL
      E-101 -> ELECTRICAL
      P-101 -> PLUMBING
      FP-101 -> FIRE_PROTECTION
      FA-101 -> FIRE_ALARM_LOW_VOLTAGE
      L-101 -> LANDSCAPE
    """
    raw = (sheet_id or "").strip().upper()
    if not raw:
        return "UNKNOWN"

    # normalize dash variants
    raw = raw.translate(str.maketrans({
        "–": "-", "—": "-", "-": "-", "−": "-", "﹣": "-", "－": "-"
    }))

    # Match either "A-101" OR "A101" OR "FP-101" OR "FP101"
    m = re.search(r"\b([A-Z]{1,3})\s*-\s*(\d+)\b|\b([A-Z]{1,3})(\d+)\b", raw)

    if m:
        # group(1) is prefix when dash-form matches; group(3) is prefix when no-dash matches
        prefix = (m.group(1) or m.group(3) or "").strip()
    else:
        # fallback: just take first 1–3 letters at start
        m2 = re.match(r"^([A-Z]{1,3})", raw)
        prefix = m2.group(1) if m2 else ""


    if prefix == "A":
        return "ARCHITECTURAL"
    if prefix in ["I", "ID"]:
        return "INTERIOR_DESIGN"
    if prefix == "S":
        return "STRUCTURAL"
    if prefix == "C":
        return "CIVIL"
    if prefix == "M":
        return "MECHANICAL"
    if prefix == "E":
        return "ELECTRICAL"
    if prefix == "P":
        return "PLUMBING"
    if prefix == "FP":
        return "FIRE_PROTECTION"
    if prefix in ["FA", "LV", "T", "ELV"]:
        return "FIRE_ALARM_LOW_VOLTAGE"
    if prefix in ["L", "LA"]:
        return "LANDSCAPE"

    return "UNKNOWN"

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

class PageIndexPage(BaseModel):
    page_number: int
    sheet_identifier: Optional[str] = None
    sheet_title: Optional[str] = None

    # ✅ use this going forward (matches /classify-pdf output)
    drawing_type: Optional[str] = None

    # (optional) keep for backward compatibility if Base44 ever stored "classification"
    classification: Optional[str] = None

    discipline: Optional[str] = None


class PageIndexPayload(BaseModel):
    filename: str
    pdf_url: Optional[str] = None
    taxonomy_version: str = "v1-coarse"
    pages: List[PageIndexPage] = Field(default_factory=list)


def _page_index_qc_flags(pages: List[PageIndexPage]) -> Dict[str, Any]:
    # Basic QC flags that help you quickly spot issues.
    missing_sheet_id_pages = [p.page_number for p in pages if not (p.sheet_identifier and p.sheet_identifier.strip())]
    low_info_title_pages = [p.page_number for p in pages if not (p.sheet_title and p.sheet_title.strip())]

    # Duplicate sheet IDs (common when OCR/classifier mistakes happen)
    sheet_map: Dict[str, List[int]] = {}
    for p in pages:
        sid = (p.sheet_identifier or "").strip()
        if not sid:
            continue
        sheet_map.setdefault(sid, []).append(p.page_number)
    duplicates = {sid: nums for sid, nums in sheet_map.items() if len(nums) > 1}

    # Classification counts
    counts: Dict[str, int] = {}
    for p in pages:
        c = (p.drawing_type or p.classification or "UNKNOWN").strip()
        counts[c] = counts.get(c, 0) + 1

    return {
        "page_count": len(pages),
        "classification_counts": counts,
        "missing_sheet_identifier_pages": missing_sheet_id_pages[:50],  # cap so response doesn't blow up
        "missing_sheet_title_pages": low_info_title_pages[:50],
        "duplicate_sheet_identifiers": duplicates,  # may be large but usually small
    }

@app.post("/page-index/qc-summary")
def page_index_qc_summary(payload: PageIndexPayload) -> Dict[str, Any]:
    flags = _page_index_qc_flags(payload.pages)

    # A simple "success criteria" you can use in UI:
    # - no missing sheet IDs
    # - page_count matches expected
    ok = (len(flags["missing_sheet_identifier_pages"]) == 0)

    return {
        "ok": ok,
        "filename": payload.filename,
        "taxonomy_version": payload.taxonomy_version,
        "stats": flags,
    }


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

class ExtractNotesV1Request(BaseModel):
    filename: str
    pages: List[Dict[str, Any]]  # Base44 stored PageIndex.pagesJson parsed into a list

# -----------------------------
# Phase 2 Stream B (Notes/Keynotes) helpers
# -----------------------------

NOTES_DRAWING_TYPES = {
    # ✅ Core notes buckets (appear in multiple disciplines)
    "SHEET_INDEX_GENERAL_NOTES",
    "SPECIFICATIONS_NOTES",
    "CODE_LIFE_SAFETY",
    "LEGENDS_SYMBOLS_ABBREVIATIONS",

    # ✅ Typical “notes-like” pages by discipline (you already defined these)
    "GENERAL_STRUCTURAL_NOTES",
    "MECHANICAL_NOTES",
    "ELECTRICAL_NOTES",
    "PLUMBING_NOTES",
    "FIRE_PROTECTION_NOTES",
    "LOW_VOLTAGE_NOTES",

    # ✅ Often contains real scope drivers / narrative requirements
    # (Not always “notes”, but in practice behaves like notes/requirements)
    "WALL_TYPES_ASSEMBLIES",
}


def _is_notes_page(p: Dict[str, Any]) -> bool:
    dt = (p.get("drawing_type") or p.get("classification") or "").strip().upper()

    # Exact matches (high confidence)
    if dt in NOTES_DRAWING_TYPES:
        return True

    # Heuristic: anything with NOTES in the type name counts as notes-like
    # (covers future taxonomy additions without updating this list)
    if "NOTES" in dt:
        return True

    # Common legend pages across disciplines
    if "LEGEND" in dt or "SYMBOL" in dt or "ABBREVIATION" in dt:
        return True

    return False

def parse_numbered_notes_robust(text: str) -> List[Dict[str, Any]]:
    """
    More robust numbered notes parser that handles edge cases.
    
    Improvements:
    - Relaxed regex (no required space after delimiter)
    - Better handling of wrapped lines
    - Filters out false positives (dates, measurements, etc.)
    """
    if not text:
        return []
    
    lines = text.split('\n')
    notes = []
    
    # More flexible pattern - delimiter can be ., ), or :
    # No required space after delimiter
    note_start_pattern = re.compile(r'^\s*(\d{1,3})\s*[\.\)\:]\s*(.*)', re.IGNORECASE)
    
    current_note_id = None
    current_text_parts = []
    
    for line in lines:
        line = line.rstrip()
        
        # Skip obviously non-note lines
        if not line.strip():
            continue
        
        # Check if this line starts a new note
        match = note_start_pattern.match(line)
        
        if match:
            note_num = match.group(1)
            note_text = match.group(2).strip()
            
            # Filter false positives:
            # - Skip if "note" is actually a measurement like "1. 5 inches"
            # - Skip if "note" is actually a date or time
            # - Skip if number is too high (> 200) - unlikely to be a note number
            if int(note_num) > 200:
                continue
            
            if note_text and (
                re.match(r'^\d+\s*(inch|in|ft|feet|mm|cm|meter)', note_text, re.IGNORECASE) or
                re.match(r'^\d+[:/]\d+', note_text)  # Date or time pattern
            ):
                continue
            
            # Save previous note if exists
            if current_note_id is not None and current_text_parts:
                combined_text = ' '.join(current_text_parts)
                combined_text = ' '.join(combined_text.split())  # Normalize whitespace
                
                if len(combined_text) > 10:  # Filter very short "notes" (likely false positives)
                    notes.append({
                        'note_id': current_note_id,
                        'text': combined_text
                    })
            
            # Start new note
            current_note_id = note_num
            current_text_parts = [note_text] if note_text else []
            
        elif current_note_id is not None:
            # Continuation line
            stripped = line.strip()
            
            # Skip lines that look like they're from a different section
            if re.match(r'^[A-Z\s]{20,}$', stripped):  # All caps header
                continue
            
            if stripped:
                current_text_parts.append(stripped)
    
    # Don't forget the last note
    if current_note_id is not None and current_text_parts:
        combined_text = ' '.join(current_text_parts)
        combined_text = ' '.join(combined_text.split())
        
        if len(combined_text) > 10:
            notes.append({
                'note_id': current_note_id,
                'text': combined_text
            })
    
    return notes


def _dedupe_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicates caused by region overlap or repeated parsing.
    Dedupes by (note_id + text).
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for n in notes:
        note_id = n.get("note_id")
        txt = (n.get("text") or "").strip()
        key = (note_id, txt)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


import statistics
from typing import List, Dict, Any, Tuple

def detect_columns_conservative(words: List[Dict], page_width: float) -> List[Tuple[float, float]]:
    """
    Conservative column detection that only splits on VERY clear gutters.
    
    Strategy:
    1. Look for vertical strips with NO words (true gutters)
    2. Gutter must be at least 50pts wide (very conservative)
    3. Gutter must span most of the page height
    4. Only split if gutter is near page center (± 20%)
    """
    if not words or len(words) < 20:  # Too few words = probably single column
        return [(0, page_width)]
    
    # Get all word bounding boxes
    word_spans = [(w['x0'], w['x1'], w['top'], w['bottom']) for w in words]
    
    # Define center zone (40-60% of page width)
    center_min = page_width * 0.4
    center_max = page_width * 0.6
    
    # Find gaps in the center zone
    MIN_GUTTER_WIDTH = 50  # Very conservative - half inch at 100 DPI
    
    # Sort words by x position
    word_spans.sort(key=lambda w: w[0])
    
    # Look for gaps between consecutive words
    potential_gutters = []
    for i in range(len(word_spans) - 1):
        gap_start = word_spans[i][1]  # Right edge of current word
        gap_end = word_spans[i + 1][0]  # Left edge of next word
        gap_width = gap_end - gap_start
        gap_center = (gap_start + gap_end) / 2
        
        # Only consider gaps in the center zone that are wide enough
        if (center_min <= gap_center <= center_max and 
            gap_width >= MIN_GUTTER_WIDTH):
            potential_gutters.append((gap_start, gap_end, gap_width))
    
    if not potential_gutters:
        return [(0, page_width)]  # No clear gutter found
    
    # Use the widest gutter in the center zone
    potential_gutters.sort(key=lambda g: g[2], reverse=True)
    best_gutter = potential_gutters[0]
    
    split_point = (best_gutter[0] + best_gutter[1]) / 2
    
    return [
        (0, best_gutter[0]),           # Left column
        (best_gutter[1], page_width)   # Right column
    ]



def extract_text_by_region(pdf_page, bbox: Tuple[float, float, float, float]) -> str:
    """
    Extract text from a region WITHOUT layout mode to avoid line break issues.
    
    Instead, we'll reconstruct lines based on y-position clustering.
    """
    x0, top, x1, bottom = bbox
    
    try:
        cropped = pdf_page.crop((x0, top, x1, bottom))
        
        # Extract words with their positions
        words = cropped.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False
        )
        
        if not words:
            return ""
        
        # Group words into lines based on y-position
        lines_dict = {}
        for word in words:
            # Round y position to nearest 2 points to group words on same line
            y_key = round(word['top'] / 2) * 2
            
            if y_key not in lines_dict:
                lines_dict[y_key] = []
            
            lines_dict[y_key].append(word)
        
        # Sort lines by y position, then sort words in each line by x position
        sorted_lines = []
        for y_pos in sorted(lines_dict.keys()):
            line_words = sorted(lines_dict[y_pos], key=lambda w: w['x0'])
            line_text = ' '.join(w['text'] for w in line_words)
            sorted_lines.append(line_text)
        
        return '\n'.join(sorted_lines)
        
    except Exception as e:
        print(f"Region extraction error: {e}")
        return ""



def get_content_bbox_adaptive(pdf_page) -> Tuple[float, float, float, float]:
    """
    Adaptive bbox that detects title block based on actual content.
    
    Strategy: Find the lowest text cluster - that's likely the title block.
    """
    w = float(pdf_page.width)
    h = float(pdf_page.height)
    
    # Get all words to find title block
    words = pdf_page.extract_words()
    
    if not words:
        # Fallback to conservative fixed bbox
        return (w * 0.05, h * 0.05, w * 0.95, h * 0.92)
    
    # Find y-position of bottom 10% of words
    word_bottoms = [word['bottom'] for word in words]
    word_bottoms.sort()
    
    # Take the 90th percentile bottom position as the cutoff
    cutoff_index = int(len(word_bottoms) * 0.9)
    bottom_cutoff = word_bottoms[cutoff_index] if cutoff_index < len(word_bottoms) else h * 0.92
    
    # Add small margins
    return (
        w * 0.03,        # 3% left margin
        h * 0.03,        # 3% top margin
        w * 0.97,        # 3% right margin
        bottom_cutoff    # Adaptive bottom
    )



def extract_notes_from_page_improved(pdf_page) -> List[Dict[str, Any]]:
    """
    Main extraction function with all improvements.
    
    Key changes:
    1. More conservative column detection
    2. Better text extraction (no layout mode issues)
    3. Robust note parsing with false positive filtering
    4. Adaptive title block detection
    """
    # Get adaptive content bbox
    bbox = get_content_bbox_adaptive(pdf_page)
    x0, top, x1, bottom = bbox
    
    # Crop to content area
    content_page = pdf_page.crop(bbox)
    
    # Extract words for column detection
    words = content_page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False
    )
    
    if not words:
        return []
    
    # Detect columns conservatively
    content_width = x1 - x0
    column_bounds = detect_columns_conservative(words, content_width)
    
    print(f"Detected {len(column_bounds)} column(s): {column_bounds}")
    
    # Extract text from each column
    all_notes = []
    
    for col_idx, (col_x0, col_x1) in enumerate(column_bounds):
        # Adjust column bounds back to full page coordinates
        col_bbox = (x0 + col_x0, top, x0 + col_x1, bottom)
        
        col_text = extract_text_by_region(pdf_page, col_bbox)
        
        if col_text:
            print(f"Column {col_idx + 1} extracted {len(col_text)} chars")
            col_notes = parse_numbered_notes_robust(col_text)
            print(f"Column {col_idx + 1} found {len(col_notes)} notes")
            all_notes.extend(col_notes)
    
    # Deduplicate by (note_id, text_prefix) to handle overlap between columns
    seen = set()
    unique_notes = []
    
    for note in all_notes:
        # Use first 50 chars for dedup key (in case of minor OCR differences)
        key = (note['note_id'], note['text'][:50])
        if key not in seen:
            seen.add(key)
            unique_notes.append(note)
    
    # Sort by note_id numerically
    unique_notes.sort(key=lambda n: int(n['note_id']))
    
    return unique_notes

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

@app.post("/extract-notes-v1")
async def extract_notes_v1(request: ExtractNotesV1Request):
    """
    Phase 2 Stream B v1: Extract numbered notes from general notes pages.
    """
    try:
        filepath = UPLOAD_DIR / request.filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {request.filename}")

        target_pages = [p for p in request.pages if _is_notes_page(p)]
        page_numbers = sorted({
            int(p.get("page_number"))
            for p in target_pages
            if p.get("page_number") is not None
        })

        if not page_numbers:
            return JSONResponse({
                "ok": True,
                "filename": request.filename,
                "page_numbers_scanned": [],
                "note_items": [],
                "summary": {"notes_found": 0, "pages_scanned": 0},
                "message": "No notes pages found (by drawing_type)."
            })

        note_items = []

        with pdfplumber.open(str(filepath)) as pdf:
            for page_num in page_numbers:
                page_idx = page_num - 1  # Convert to 0-based
                
                if page_idx < 0 or page_idx >= len(pdf.pages):
                    continue

                page = pdf.pages[page_idx]
                notes = extract_notes_from_page_improved(page)
                
                for note in notes:
                    note_items.append({
                        'source_page_number': page_num,
                        'note_id': note['note_id'],
                        'text': note['text'],
                        'parse_method': 'improved_multicolumn_extraction'
                    })

        return JSONResponse({
            "ok": True,
            "filename": request.filename,
            "page_numbers_scanned": page_numbers,
            "note_items": note_items,
            "summary": {
                "notes_found": len(note_items),
                "pages_scanned": len(page_numbers)
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extract-notes-v1 failed: {str(e)}")

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
    Discipline-aware page classifier:
    1) Extract sheet id/title
    2) Infer discipline from sheet id
    3) Classify type using discipline-specific allowed list
    4) Normalize + override deterministically
    """
    meta = _extract_sheet_id_title(image_bytes)
    sheet_id = meta["sheet_identifier"]
    sheet_title = meta["sheet_title"]

    discipline = infer_discipline_from_sheet_id(sheet_id)

    model_type = _classify_type_for_discipline(image_bytes, discipline)
    base_type = normalize_sheet_type(discipline, model_type)
    final_type = override_by_title(discipline, sheet_title, sheet_id, base_type)

    return {
        "discipline": discipline,
        "drawing_type": final_type,
        "sheet_identifier": sheet_id,
        "sheet_title": sheet_title,
        "model_type": model_type,  # debug
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

        start = max(1, int(request.start_page))

        # ✅ NEW: if Base44 asks beyond the last page, return empty list (don't error)
        if start > total_pages:
            doc.close()
            return JSONResponse({
                "filename": request.filename,
                "total_pages": total_pages,
                "pages": []
            })

        end = int(request.end_page) if request.end_page else total_pages
        end = min(end, total_pages)

        if start > end:
            doc.close()
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
            print("DEBUG classify:", page_number, page_info)
            sheet_id = page_info.get("sheet_identifier")
            sheet_title = page_info.get("sheet_title")

            discipline = page_info.get("discipline") or infer_discipline_from_sheet_id(sheet_id)
            drawing_type = page_info.get("drawing_type") or "OTHER"


            pages_out.append({
                "page_number": page_number,
                "sheet_identifier": sheet_id,
                "sheet_title": sheet_title,
                "discipline": discipline,
                "drawing_type": drawing_type,

                # optional debug fields:
                "model_type": page_info.get("model_type", ""),
            })



        doc.close()

        return JSONResponse({
            "filename": request.filename,
            "total_pages": total_pages,
            "pages": pages_out,
            "source": "autoscope-python-backend"
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
