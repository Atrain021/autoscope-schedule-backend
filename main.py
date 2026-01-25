# main.py - AutoScope Schedule Backend (Production Ready) 

import os
import re
import gc
import json
import uuid
import base64
from typing import Dict, List, Optional, Any
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

def _parse_numbered_notes(text: str) -> List[Dict[str, Any]]:
    """
    Parse notes like:
      1. text...
      2) text...
      3: text...
    Joins wrapped lines into the note.
    No nonlocal usage (safer).
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    notes: List[Dict[str, Any]] = []

    current_key: Optional[str] = None
    current_parts: List[str] = []

    key_re = re.compile(r"^\s*(\d{1,4})\s*[\)\.\:]\s+(.*)$")

    for ln in lines:
        m = key_re.match(ln)
        if m:
            if current_key and current_parts:
                notes.append({
                    "note_id": current_key,
                    "text": " ".join(" ".join(current_parts).split())
                })
            current_key = m.group(1)
            current_parts = [m.group(2)]
        else:
            if current_key and ln.strip():
                current_parts.append(ln.strip())

    if current_key and current_parts:
        notes.append({
            "note_id": current_key,
            "text": " ".join(" ".join(current_parts).split())
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

def detect_columns_by_density(words: List[Dict[str, Any]], page_width: float, 
                               bin_width: float = 10.0) -> List[Tuple[float, float]]:
    """
    Detect column boundaries using horizontal density analysis.
    
    Strategy:
    1. Create histogram of word density across page width
    2. Find valleys (low-density regions) = gutters between columns
    3. Return column boundaries as (x_start, x_end) tuples
    
    Returns: List of (x_min, x_max) tuples for each column, sorted left-to-right
    """
    if not words:
        return [(0, page_width)]
    
    # Build density histogram
    num_bins = int(page_width / bin_width) + 1
    density = [0] * num_bins
    
    for word in words:
        # Count word presence in bins it overlaps
        x_start = word["x0"]
        x_end = word["x1"]
        
        bin_start = int(x_start / bin_width)
        bin_end = int(x_end / bin_width)
        
        for b in range(max(0, bin_start), min(num_bins, bin_end + 1)):
            density[b] += 1
    
    # Find valleys (potential gutters)
    # A valley is a region with significantly lower density than neighbors
    threshold = statistics.mean(density) * 0.2  # valleys should be <20% of average
    
    valleys = []
    in_valley = False
    valley_start = 0
    
    for i, d in enumerate(density):
        if d < threshold and not in_valley:
            valley_start = i
            in_valley = True
        elif d >= threshold and in_valley:
            valley_end = i
            valleys.append((valley_start * bin_width, valley_end * bin_width))
            in_valley = False
    
    # Convert valleys to column boundaries
    if not valleys:
        # No gutters found = single column
        return [(0, page_width)]
    
    columns = []
    current_x = 0
    
    for valley_start, valley_end in valleys:
        # Column ends where gutter begins
        if valley_start - current_x > 50:  # minimum column width
            columns.append((current_x, valley_start))
        current_x = valley_end
    
    # Last column (after final gutter)
    if page_width - current_x > 50:
        columns.append((current_x, page_width))
    
    return columns if columns else [(0, page_width)]


def assign_words_to_columns(words: List[Dict[str, Any]], 
                            column_bounds: List[Tuple[float, float]]) -> List[List[Dict[str, Any]]]:
    """
    Assign each word to exactly one column based on its x-center position.
    
    Returns: List of word lists, one per column
    """
    columns = [[] for _ in column_bounds]
    
    for word in words:
        x_center = (word["x0"] + word["x1"]) / 2.0
        
        # Find which column this word belongs to
        for i, (x_min, x_max) in enumerate(column_bounds):
            if x_min <= x_center <= x_max:
                columns[i].append(word)
                break
    
    return columns


def words_to_lines_strict(words: List[Dict[str, Any]], y_tolerance: float = 3.0) -> List[str]:
    """
    Convert words to lines using STRICT y-position grouping.
    
    Key: Only group words with nearly identical y-positions (same line).
    """
    if not words:
        return []
    
    # Sort by y-position first (top to bottom), then x-position (left to right)
    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    
    lines = []
    current_line_words = []
    current_y = None
    
    for word in sorted_words:
        word_y = round(word["top"], 1)
        
        if current_y is None:
            current_y = word_y
            current_line_words = [word["text"]]
        elif abs(word_y - current_y) <= y_tolerance:
            # Same line
            current_line_words.append(word["text"])
        else:
            # New line
            if current_line_words:
                lines.append(" ".join(current_line_words))
            current_line_words = [word["text"]]
            current_y = word_y
    
    # Don't forget last line
    if current_line_words:
        lines.append(" ".join(current_line_words))
    
    return lines


def extract_text_multicolumn(pdf_page) -> str:
    """
    Extract text from PDF page respecting column layout.
    
    Algorithm:
    1. Extract all words with coordinates
    2. Detect column boundaries using density analysis
    3. Assign words to columns by x-center
    4. Within each column, build lines top-to-bottom
    5. Concatenate columns left-to-right
    """
    words = pdf_page.extract_words(
        x_tolerance=2,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,  # CRITICAL: don't let pdfplumber guess reading order
    )
    
    if not words:
        return ""
    
    page_width = float(pdf_page.width)
    
    # Step 1: Detect columns
    column_bounds = detect_columns_by_density(words, page_width)
    
    # Step 2: Assign words to columns
    column_words = assign_words_to_columns(words, column_bounds)
    
    # Step 3: Build text for each column
    column_texts = []
    for col_words in column_words:
        if not col_words:
            continue
        lines = words_to_lines_strict(col_words)
        col_text = "\n".join(lines)
        if col_text.strip():
            column_texts.append(col_text)
    
    # Step 4: Join columns left-to-right
    return "\n\n".join(column_texts)


# Integration with your existing code:
def _extract_text_reading_order_auto_columns_FIXED(pdf_page) -> str:
    """
    Drop-in replacement for your existing function.
    """
    return extract_text_multicolumn(pdf_page)

def _extract_two_column_texts_by_crop(pdf_page) -> List[str]:
    """
    Deterministic 2-column reader:
    - crops the page into left/right halves (with a gutter)
    - extracts words from each crop
    - reconstructs lines top->bottom inside each crop
    Returns [left_text, right_text] (empty strings removed).
    """
    w = float(getattr(pdf_page, "width", 0) or 0)
    h = float(getattr(pdf_page, "height", 0) or 0)
    if w <= 0 or h <= 0:
        t = pdf_page.extract_text() or ""
        return [t] if t else []

    gutter = max(24.0, w * 0.04) # 2% width or 12pt
    mid = w / 2.0

    # IMPORTANT: do NOT crop top/bottom here.
    # The caller (extract-notes-v1) already crops the page using bbox.
    left = pdf_page.crop((0, 0, mid - gutter, h))
    right = pdf_page.crop((mid + gutter, 0, w, h))

    def words_to_text(p) -> str:
        words = p.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False,
        )
        if not words:
            return ""
        lines = words_to_lines_strict(words)
        return "\n".join(lines).strip()

    left_text = words_to_text(left)
    right_text = words_to_text(right)

    out: List[str] = []
    if left_text:
        out.append(left_text)
    if right_text:
        out.append(right_text)
    return out


def _looks_like_numbered_notes_page(text: str) -> bool:
    t = (text or "").upper()
    if "GENERAL NOTES" in t or "NOTES" in t:
        return True
    # If it has multiple numbered note starters, it’s probably a notes list
    starters = re.findall(r"(?m)^\s*\d{1,3}\s*[\)\.\:]\s+", text or "")
    return len(starters) >= 5

def _get_notes_content_bbox(pdf_page):
    """
    Return a bounding box (x0, top, x1, bottom) that excludes
    common junk areas like title blocks/footers and side keyplans.
    Coordinates are in PDF points. pdfplumber uses (x0, top, x1, bottom).
    """
    w = float(pdf_page.width)
    h = float(pdf_page.height)

    # Conservative defaults that work on most sheets:
    # - remove bottom 12% (title block / footer)
    # - remove right 12% (keyplan/index side strip) ONLY if you want
    x0 = 0
    top = 0
    x1 = w
    bottom = h * 0.88  # keep top 88%

    # OPTIONAL: if your notes sheets often have a keyplan strip on the far right,
    # uncomment this:
    # x1 = w * 0.88

    return (x0, top, x1, bottom)

def _histogram_density(words: List[Dict[str, Any]], axis: str, bins: int, min_v: float, max_v: float) -> List[int]:
    """
    Build a simple count histogram for word positions along an axis ('x' uses x0/x1 mid, 'y' uses top/bottom mid).
    """
    hist = [0] * bins
    if not words:
        return hist

    span = max(max_v - min_v, 1e-6)
    for w in words:
        if axis == "x":
            v = (float(w["x0"]) + float(w["x1"])) / 2.0
        else:
            v = (float(w["top"]) + float(w["bottom"])) / 2.0

        idx = int(((v - min_v) / span) * bins)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1

    return hist


def _find_best_valley_cut(hist: List[int], cut_range: tuple[float, float] = (0.30, 0.70)) -> Optional[int]:
    """
    Choose a cut index in the histogram where density is low (a valley),
    but only within the central portion (default 30%..70%) to avoid margins.
    Returns bin index or None.
    """
    if not hist or len(hist) < 10:
        return None

    n = len(hist)
    lo = int(n * cut_range[0])
    hi = int(n * cut_range[1])
    if hi <= lo + 2:
        return None

    window = hist[lo:hi]
    if not window:
        return None

    # Pick the minimum density bin. If many, prefer the one closest to center.
    min_val = min(window)
    candidates = [i + lo for i, v in enumerate(window) if v == min_val]
    if not candidates:
        return None

    center = n / 2.0
    candidates.sort(key=lambda i: abs(i - center))
    return candidates[0]


def _split_bbox_vertical(bbox: tuple[float, float, float, float], x_cut: float, gap: float = 6.0):
    """
    Split bbox into left/right at x_cut. gap makes a small gutter.
    """
    x0, top, x1, bottom = bbox
    left = (x0, top, max(x0, x_cut - gap), bottom)
    right = (min(x1, x_cut + gap), top, x1, bottom)
    return left, right


def _split_bbox_horizontal(bbox: tuple[float, float, float, float], y_cut: float, gap: float = 6.0):
    """
    Split bbox into top/bottom at y_cut. gap makes a small gutter.
    """
    x0, top, x1, bottom = bbox
    upper = (x0, top, x1, max(top, y_cut - gap))
    lower = (x0, min(bottom, y_cut + gap), x1, bottom)
    return upper, lower


def _notes_region_score(text: str) -> float:
    """
    Heuristic score: higher means more likely to be a notes block.
    Boosts numbered-list starts; penalizes sheet-index style lists.
    """
    t = (text or "")
    tu = t.upper()

    # Count numbered note starters at line starts: "1.", "2)", "3:"
    starters = re.findall(r"(?m)^\s*\d{1,3}\s*[\)\.\:]\s+", t)
    starter_count = len(starters)

    # Penalize sheet index / drawing list vibe: lots of tokens like A101, E-601, M301
    sheet_ids = re.findall(r"\b[A-Z]{1,3}\s*[-]?\s*\d{2,4}\b", tu)
    sheet_id_count = len(sheet_ids)

    score = 0.0
    score += starter_count * 3.0

    if "GENERAL NOTES" in tu:
        score += 10.0
    elif "NOTES" in tu:
        score += 4.0

    # Penalize if it looks like a sheet index region (lots of sheet ids, few numbered starters)
    if sheet_id_count >= 12 and starter_count <= 3:
        score -= 12.0
    else:
        score -= min(sheet_id_count, 30) * 0.2

    # If it's extremely short, likely not useful
    if len(t.strip()) < 120:
        score -= 3.0

    return score


def regionize_page_v1(pdf_page, base_bbox: Optional[tuple[float, float, float, float]] = None,
                      max_regions: int = 6, min_words_per_region: int = 35) -> List[tuple[float, float, float, float]]:
    """
    Regionizer v1:
    - Start from base_bbox (or full page).
    - Extract words in that bbox.
    - Find a strong vertical or horizontal whitespace valley near the center.
    - Recursively split up to max_regions, ensuring each region has enough words.

    Returns list of bboxes in ORIGINAL page coordinates.
    """
    # Start bbox
    if base_bbox is None:
        base_bbox = (0.0, 0.0, float(pdf_page.width), float(pdf_page.height))

    regions = [base_bbox]

    def word_count_in_bbox(b: tuple[float, float, float, float]) -> int:
        try:
            cropped = pdf_page.crop(b)
            ws = cropped.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=False)
            return len(ws or [])
        except Exception:
            return 0

    # Simple best-first splitting
    while True:
        if len(regions) >= max_regions:
            break

        # Pick region with the most words to attempt splitting
        regions_with_counts = [(b, word_count_in_bbox(b)) for b in regions]
        regions_with_counts.sort(key=lambda x: x[1], reverse=True)

        # If biggest region is too small, stop
        b0, c0 = regions_with_counts[0]
        if c0 < (min_words_per_region * 2):
            break

        # Try split this region
        cropped0 = pdf_page.crop(b0)
        words = cropped0.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=False) or []
        if len(words) < (min_words_per_region * 2):
            break

        # Compute histograms in cropped coordinate space
        x0, top0, x1, bot0 = b0
        w_span = max(x1 - x0, 1e-6)
        h_span = max(bot0 - top0, 1e-6)

        # Use midpoints along axes
        x_hist = _histogram_density(words, axis="x", bins=80, min_v=x0, max_v=x1)
        y_hist = _histogram_density(words, axis="y", bins=80, min_v=top0, max_v=bot0)

        x_cut_bin = _find_best_valley_cut(x_hist, (0.30, 0.70))
        y_cut_bin = _find_best_valley_cut(y_hist, (0.30, 0.70))

        # Turn bin into coordinate
        x_cut = None
        y_cut = None
        if x_cut_bin is not None:
            x_cut = x0 + (x_cut_bin / 80.0) * w_span
        if y_cut_bin is not None:
            y_cut = top0 + (y_cut_bin / 80.0) * h_span

        # Decide which split is "better" by checking word balance
        best_split = None  # ("v" or "h", bA, bB)
        best_balance = 0.0

        if x_cut is not None:
            left, right = _split_bbox_vertical(b0, x_cut, gap=8.0)
            cL = word_count_in_bbox(left)
            cR = word_count_in_bbox(right)
            if min(cL, cR) >= min_words_per_region:
                balance = min(cL, cR) / max(cL, cR)
                if balance > best_balance:
                    best_balance = balance
                    best_split = ("v", left, right)

        if y_cut is not None:
            upper, lower = _split_bbox_horizontal(b0, y_cut, gap=8.0)
            cU = word_count_in_bbox(upper)
            cD = word_count_in_bbox(lower)
            if min(cU, cD) >= min_words_per_region:
                balance = min(cU, cD) / max(cU, cD)
                if balance > best_balance:
                    best_balance = balance
                    best_split = ("h", upper, lower)

        # If no split met minimum criteria, stop splitting
        if not best_split:
            break

        # Apply split: replace b0 in regions with the two new bboxes
        _, bA, bB = best_split
        new_regions = []
        for b in regions:
            if b == b0:
                new_regions.extend([bA, bB])
            else:
                new_regions.append(b)
        regions = new_regions

    return regions


def _extract_best_notes_text_from_page(pdf_page, base_bbox: tuple[float, float, float, float]) -> tuple[str, List[tuple]]:
    """
    Uses regionizer + scoring to select best region(s) for notes extraction.
    Returns (combined_text, chosen_region_bboxes).
    """
    region_bboxes = regionize_page_v1(pdf_page, base_bbox=base_bbox, max_regions=6, min_words_per_region=35)

    scored_regions = []
    for rb in region_bboxes:
        try:
            cropped = pdf_page.crop(rb)
            text = extract_text_multicolumn(cropped) or ""
            score = _notes_region_score(text)
            scored_regions.append((score, rb, text))
        except Exception:
            continue

    if not scored_regions:
        return "", []

    scored_regions.sort(key=lambda x: x[0], reverse=True)

    # Keep the best region, and optionally a 2nd if it’s also clearly notes-like
    best = scored_regions[0]
    chosen = [best]

    if len(scored_regions) > 1:
        second = scored_regions[1]
        # Only include 2nd region if it has real numbered note density too
        if second[0] >= best[0] * 0.65 and second[0] >= 6.0:
            chosen.append(second)

    combined = "\n\n".join([c[2] for c in chosen]).strip()
    return combined, [c[1] for c in chosen]


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
    Phase 2 Stream B v1:
    - Select pages by drawing_type (General Notes / Code / Legends / Wall Types)
    - Extract text with pdfplumber
    - Parse numbered notes where possible
    - Fallback to a text snippet if not numbered
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

        note_items: List[Dict[str, Any]] = []

        with pdfplumber.open(str(filepath)) as pdf:
            total_pages = len(pdf.pages)

            for pn in page_numbers:
                idx = pn - 1  # ✅ your system is 1-based page_number
                if idx < 0 or idx >= total_pages:
                    continue

                page = pdf.pages[idx]
                bbox = _get_notes_content_bbox(page)

                # 1) ✅ Regionize + pick best notes-like region(s)
                text, chosen_regions = _extract_best_notes_text_from_page(page, base_bbox=bbox)

                # 2) Fallback: try the bbox only (still avoids title block)
                if not text.strip():
                    cropped = page.crop(bbox)
                    text = _extract_text_reading_order_auto_columns(cropped) or ""

                # 3) Fallback: try full-page reading order
                if not text.strip():
                    text = extract_text_multicolumn(page) or ""

                # 4) Last resort: raw pdfplumber text
                if not text.strip():
                    # For notes sheets: use column-aware reading order to avoid left/right interleaving
                    text = _extract_text_reading_order_auto_columns(page) or (page.extract_text() or "")


                # ✅ NEW: parse per column (prevents left/right interleaving)
                page_notes: List[Dict[str, Any]] = []

                # Prefer column extraction from the chosen regions if we have them,
                # otherwise use the bbox-cropped page.
                # Always parse from ONE stable crop: the bbox.
                # (This prevents double-parsing and region overlap duplicates.)
                cropped_bbox = page.crop(bbox)

                # 1) Try your general N-column method first (works for 1,2,3,4,5 columns)
                words = cropped_bbox.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=False)
                page_width = float(cropped_bbox.width)
                column_bounds = detect_columns_by_density(words, page_width)
                column_words = assign_words_to_columns(words, column_bounds)
                col_texts = []
                for col_words in column_words:
                    if col_words:
                        lines = words_to_lines_strict(col_words)
                        col_text = "\n".join(lines)
                        if col_text.strip():
                            col_texts.append(col_text)

                # 2) If it looks like a 2-column page, compare against the hard 2-column crop method
                if len(col_texts) == 2:
                    def starters_count(t: str) -> int:
                        return len(re.findall(r"(?m)^\s*\d{1,4}\s*[\)\.\:]\s+", t or ""))

                    clustered_score = starters_count(col_texts[0]) + starters_count(col_texts[1])

                    cropped_2col = _extract_two_column_texts_by_crop(cropped_bbox)
                    crop_score = sum(starters_count(t) for t in cropped_2col)

                    # pick whichever gives more numbered-note starters
                    if crop_score > clustered_score:
                        col_texts = cropped_2col

                # 3) Fallback: if nothing came back, use reading-order text
                if not col_texts:
                    fallback_text = extract_text_multicolumn(cropped_bbox) or (cropped_bbox.extract_text() or "")
                    col_texts = [fallback_text] if fallback_text else []

                # 4) Parse notes per column
                for col_text in col_texts:
                    page_notes.extend(_parse_numbered_notes(col_text))

                # 5) Deduplicate (fixes your “same notes twice” problem)
                page_notes = _dedupe_notes(page_notes)




                if page_notes:
                    for n in page_notes:
                        note_items.append({
                            "source_page_number": pn,
                            "note_id": n["note_id"],
                            "text": n["text"],
                            "parse_method": "numbered_notes_two_column_crop"
                        })
                else:
                    # fallback snippet (only if it really looks notes-like)
                    cleaned = " ".join(text.strip().split())
                    if cleaned and _looks_like_numbered_notes_page(cleaned) and _notes_region_score(text) >= 6.0:
                        note_items.append({
                            "source_page_number": pn,
                            "note_id": None,
                            "text": cleaned[:2000],
                            "parse_method": "fulltext_snippet_column_aware"
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
