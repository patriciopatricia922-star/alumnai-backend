from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from typing import Optional
import os
import re
import sys
import logging
import requests
import subprocess
from dotenv import load_dotenv

# ─── CONFIGURATION & INITIALIZATION ──────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alumni_id_parser")

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ai_engine import init_ai, sentiment_analyzer, feedback_summarizer, insights_generator

    init_ai()
    AI_AVAILABLE = True
    print("✅ AI engine loaded and initialized")
except ImportError as e:
    print(f"⚠️ AI engine import error: {e}")
    AI_AVAILABLE = False

    def init_ai():
        return False

    sentiment_analyzer = None
    feedback_summarizer = None
    insights_generator = None

app = FastAPI(title="Alumni AI System", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://localhost:5174",
#         "http://localhost:3000",
#         "http://127.0.0.1:5173",
#         "http://127.0.0.1:5174",
#     ],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5174"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
OCR_API_KEY = os.getenv("OCR_API_KEY", "YOUR_API_KEY_HERE")


# =============================================================================
# CANONICAL PROGRAM LIST
# =============================================================================
# This is the single source of truth for approved program abbreviations.
# Every entry in prog_patterns must have a corresponding alias entry in
# _ALIAS_TABLE (train_model.py) so that OCR output and manual survey entries
# both resolve to the same canonical label.
#
# Order matters: list longest / most-specific patterns first so that
# "BSBA-MktgMgt" is matched before the shorter "BSBA" substring.
# =============================================================================
CANONICAL_PROGRAMS = [
    # SECA
    "BSArch",
    "BSIT-MWA",
    "BSCE",
    "BSCpE",
    "BSCS-ML",
    # SASE
    "BSPSY",
    "ABComm",
    "BPEd",
    # SBMA (longest/most-specific first)
    "BSBA-MktgMgt",
    "BSBA-FinMgt",
    "BSBA-HRM",
    "BSHM",
    "BSMA",
    "BSTM",
    "BSAccountancy",
    # SHS Strands
    "SHS-STEM",
    "SHS-ABM",
    "SHS-HUMSS",
]


# ─── NAME PARSING HELPERS (Program-anchored, line-break independent) ────────
# These helpers replace the old "line 1 = first name, line 2 = middle+last"
# assumption. Instead, the detected program acts as a structural anchor:
# everything before it is candidate name content, which is normalized into
# a single token stream before any semantic splitting happens. This makes
# the result insensitive to how OCR.Space happened to break the lines.

SUFFIX_RX = re.compile(r'\b(JR|SR|II|III|IV)\.?', re.IGNORECASE)
COMPOUND_PREFIXES = ["DE", "DEL", "DELA", "DE LOS", "DE LAS", "SAN", "SANTA"]


def strip_suffix(text):
    """Remove a trailing generational suffix (Jr., Sr., II, III, IV) from text."""
    match = SUFFIX_RX.search(text)
    if match:
        return SUFFIX_RX.sub('', text).strip(), match.group(0).strip().rstrip('.')
    return text, ""


def extract_last_name(tokens):
    """
    Detect a compound surname prefix at the END of a token list and split
    the list into (remaining_tokens, last_name_tokens).

    Two-word prefixes (e.g. "DE LOS", "DE LAS") are checked before
    single-word ones (e.g. "DEL", "SAN", "SANTA") so the more specific
    match wins. Falls back to treating the final token as the surname.
    """
    if not tokens:
        return [], []

    if len(tokens) >= 3:
        two_word = f"{tokens[-3]} {tokens[-2]}".upper()
        if two_word in COMPOUND_PREFIXES:
            return tokens[:-3], tokens[-3:]

    if len(tokens) >= 2:
        one_word = tokens[-2].upper()
        if one_word in COMPOUND_PREFIXES:
            return tokens[:-2], tokens[-2:]

    return tokens[:-1], tokens[-1:]


def split_first_middle(tokens):
    """
    Split the tokens that remain after the surname has been removed into
    (first_name, middle_name).

    - 0 tokens  -> ("", None)
    - 1 token   -> (token, None)                       e.g. "Juan Cruz"
    - 2 tokens  -> (token0, token1)                     e.g. "Juan Santos Cruz"
    - 3+ tokens -> (token0, "token1 token2 ...")         e.g. "Juan Miguel Antonio Cruz"

    NOTE: A single given name followed by one or more middle names is the
    more common Philippine ID convention, so that is the default for 3+
    tokens. Genuinely compound *first* names (e.g. "Jeremey Michael") that
    are OCR'd on their own line are still recovered correctly — see the
    line-hint logic in parse_full_name, which overrides this default when
    the original OCR line grouping unambiguously indicates a multi-word
    first name. When no such hint exists, this is an inherent ambiguity in
    Filipino naming (compound first name vs. multiple middle names) that
    cannot be resolved from tokens alone.
    """
    if not tokens:
        return "", None
    if len(tokens) == 1:
        return tokens[0], None
    if len(tokens) == 2:
        return tokens[0], tokens[1]
    return tokens[0], " ".join(tokens[1:])


def parse_full_name(tokens, first_line_token_count=None):
    """
    Turn a flat list of name tokens into (first_name, middle_name, last_name).

    `first_line_token_count`, when provided, is the number of tokens that
    made up the first candidate OCR line. If that count is smaller than the
    number of tokens remaining once the surname is removed, it is used as a
    hint that the first line represents a (possibly multi-word) first name,
    letting us recover compound first names such as "Jeremey Michael"
    without depending on token count alone.
    """
    tokens = list(tokens)
    suffix = ""

    if tokens:
        cleaned, suf = strip_suffix(tokens[-1])
        if suf:
            suffix = suf
            if cleaned:
                tokens[-1] = cleaned
            else:
                tokens.pop()

    remainder, last_tokens = extract_last_name(tokens)

    first_name, middle_name = None, None
    if first_line_token_count and 0 < first_line_token_count < len(remainder):
        first_name = " ".join(remainder[:first_line_token_count])
        rest = remainder[first_line_token_count:]
        middle_name = " ".join(rest) if rest else None

    if first_name is None:
        first_name, middle_name = split_first_middle(remainder)

    last_name = " ".join(last_tokens)
    if suffix:
        last_name = f"{last_name} {suffix}".strip()

    return first_name, middle_name, last_name


# ─── OCR PROCESSING UTILITIES ────────────────────────────────────────────────
def parse_id_content(raw_text):
    """Parse OCR text from NU Alumni ID cards with robust university and compound name validation"""
    upper_text = raw_text.upper().replace('\n', ' ')

    # ── STEP 1: UNIVERSITY VALIDATION ────────────────────────────────────────
    has_national = "NATIONAL" in upper_text or "NATL" in upper_text
    has_university = "UNIVERSITY" in upper_text or "UNIV" in upper_text
    is_alumni = "ALUMNI" in upper_text

    if not (has_national and has_university):
        return {"verified": False, "reason": "This ID does not appear to be a National University ID."}

    if not is_alumni:
        return {"verified": False, "reason": "This ID does not appear to be an Alumni ID."}

    # ── STEP 2: BRANCH VALIDATION (DASMARIÑAS) ───────────────────────────────
    dasma_markers = ["DASMARIÑAS", "DASMARINAS", "NU-D", "NUD", "DASMA"]
    is_dasmarinas = any(x in upper_text for x in dasma_markers) or re.search(r'NU\s+D\b', upper_text)

    if not is_dasmarinas:
        other_branches = [("MANILA", "Manila"), ("FAIRVIEW", "Fairview"), ("MOA", "MOA"), ("LIPA", "Lipa")]
        for keyword, label in other_branches:
            if keyword in upper_text:
                return {"verified": False, "reason": f"This is an NU {label} ID. Only NU Dasmariñas is accepted."}
        return {"verified": False, "reason": "Could not verify as an NU Dasmariñas ID."}

    # ── STEP 3: DATA EXTRACTION ──────────────────────────────────────────────
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    # Match OCR text against canonical programs.
    # Uses CANONICAL_PROGRAMS (ordered longest-first) so that the most
    # specific match wins. The canonical abbreviation itself (not the raw
    # OCR line) is stored so downstream code always receives a clean label.
    # We also record which line the program was found on: this line acts
    # as the structural anchor for name extraction below.
    program = ""
    program_line_idx = None
    for idx, line in enumerate(lines):
        line_upper = line.upper()
        # Collapse any spaces inside the line for matching against
        # space-collapsed canonical keys (e.g. "BS ARCH" -> "BSARCH")
        line_collapsed = re.sub(r'[\s\-]+', '', line_upper)
        for p in CANONICAL_PROGRAMS:
            p_upper = p.upper()
            p_collapsed = re.sub(r'[\s\-]+', '', p_upper)
            if p_upper in line_upper or p_collapsed in line_collapsed:
                program = p   # always store the canonical abbreviation
                program_line_idx = idx
                break
        if program:
            break

    batch_year = ""
    year_match = re.search(r'(?:Class\s*)?(\b20\d{2}\b)', raw_text, re.IGNORECASE)
    if year_match:
        batch_year = year_match.group(1)

    def is_name_line(line):
        u = line.upper()
        blacklist = ["NATIONAL", "UNIVERSITY", "ALUMNI", "DASMARIÑAS", "CLASS", "ID", "NO."]
        if any(b in u for b in blacklist):
            return False
        if any(p.upper() in u for p in CANONICAL_PROGRAMS):
            return False
        if re.match(r'^[0-9\-]+$', u):
            return False
        return len(line) > 2

    # ── STRUCTURAL ANCHOR ─────────────────────────────────────────────────
    # Everything before the detected program line is candidate name content.
    # If the program wasn't found at all, fall back to scanning every line
    # (previous behaviour) rather than failing outright.
    candidate_lines = lines[:program_line_idx] if program_line_idx is not None else lines
    name_lines = [l.strip() for l in candidate_lines if is_name_line(l)]

    # Normalize whitespace and reconstruct a single name string BEFORE
    # splitting it into fields. This is what makes the parser tolerant of
    # OCR line-break inconsistencies: "YEMEHIA" / "DENTON ABAD",
    # "YEMEHIA" / "DENTON" / "ABAD", "YEMEHIA DENTON" / "ABAD", and
    # "YEMEHIA DENTON ABAD" all collapse to the same normalized token
    # stream ["Yemehia", "Denton", "Abad"].
    normalized_name = re.sub(r'\s+', ' ', " ".join(name_lines)).strip()
    raw_tokens = normalized_name.split(" ") if normalized_name else []
    name_tokens = [re.sub(r"[^A-Za-z.\-']", "", t) for t in raw_tokens]
    name_tokens = [t for t in name_tokens if t]

    # Hint: how many tokens made up the first candidate line. When there is
    # more than one candidate line and that first line's token count is
    # smaller than what remains after the surname is removed, it signals a
    # (possibly multi-word) first name that OCR kept on its own line —
    # letting us recover compound first names without guessing from token
    # count alone.
    first_line_token_count = None
    if len(name_lines) >= 2:
        first_line_token_count = len(
            [t for t in re.sub(r'\s+', ' ', name_lines[0]).strip().split(" ") if t]
        )

    first_name, middle_name, last_name = "", "", ""
    if name_tokens:
        first_name, middle_name, last_name = parse_full_name(
            name_tokens, first_line_token_count=first_line_token_count
        )

    if not first_name or not last_name:
        logger.warning(
            "Incomplete name parse. normalized_candidate=%r raw_text=%r",
            normalized_name, raw_text,
        )

    return {
        "verified": True,
        "extracted": {
            "firstName":  first_name.title() if first_name else "",
            "middleName": middle_name.title() if middle_name else None,
            "lastName":   last_name.title() if last_name else "",
            "program":    program,       # canonical abbreviation, never raw OCR text
            "batchYear":  batch_year,
            "rawText":    raw_text,
        },
    }


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.post("/api/verify-alumni-id")
async def verify_id_endpoint(file: UploadFile = File(...)):
    payload = {
        'apikey': OCR_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False,
        'detectOrientation': True,
        'scale': True,
        'OCREngine': '2',
    }
    try:
        file_content = await file.read()
        files = {'file': (file.filename, file_content, file.content_type)}
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files=files,
            data=payload,
            timeout=20,
        )
        data = response.json()
        if data.get("IsErroredOnProcessing"):
            return {"verified": False, "reason": "OCR Engine error."}
        parsed_results = data.get("ParsedResults")
        if not parsed_results:
            return {"verified": False, "reason": "No text detected."}
        return parse_id_content(parsed_results[0].get("ParsedText", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def get_predictions():
    return supabase.table("predictions").select("*").execute().data


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "1.0.1"}


# ─── REFRESH PREDICTIONS ─────────────────────────────────────────────────────
@app.post("/api/refresh-predictions")
async def refresh_predictions():
    """
    Re-run the stacking ensemble (train_model.py) and push fresh
    predictions to Supabase.
    """
    try:
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            error_tail = result.stderr[-500:] if result.stderr else "train_model.py exited non-zero"
            return {"status": "error", "message": error_tail}
        return {"status": "success", "message": "Predictions refreshed successfully."}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Training timed out after 5 minutes."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── AI PREDICTIVE INSIGHTS ──────────────────────────────────────────────────
from pydantic import BaseModel
from typing import Any


class DepartmentItem(BaseModel):
    code: str
    name: str
    current_rate: float
    predicted_rate: float
    change: float


class TrendPoint(BaseModel):
    year: str
    value: float


class SelectedDepartment(BaseModel):
    name: str
    programs: Optional[list] = []


class InsightsRequest(BaseModel):
    overview_trend: Optional[list[TrendPoint]] = []
    departments: Optional[list[DepartmentItem]] = []
    current_view: Optional[str] = "overview"
    selected_department: Optional[SelectedDepartment] = None


@app.post("/api/ai/predictive-insights")
async def get_predictive_insights(payload: InsightsRequest):
    """
    Accept the analytics context from the frontend and return
    AI-shaped insights whose field names match what the UI reads:
      key_insight, trend_analysis, department_insights,
      recommendations, risk_alert
    """
    try:
        resp = supabase.table("survey_progress").select(
            "employment_information_data"
        ).execute()

        feedback_texts = []
        for row in (resp.data or []):
            emp = row.get("employment_information_data") or {}
            for key in ("feedback", "comments", "suggestions", "remarks"):
                text = emp.get(key) or ""
                if text and len(text) > 10:
                    feedback_texts.append(str(text))
                    break

        avg_len = (
            round(sum(len(t) for t in feedback_texts) / len(feedback_texts), 1)
            if feedback_texts else 0
        )

        departments = payload.departments or []
        overview    = payload.overview_trend or []

        trend_dir = "upward"
        if len(overview) >= 2:
            trend_dir = "upward" if overview[-1].value > overview[0].value else "stable or declining"

        top_dept = max(departments, key=lambda d: d.change, default=None)
        low_dept = min(departments, key=lambda d: d.change, default=None)

        year_range = (
            f"{overview[0].year}–{overview[-1].year}" if len(overview) >= 2
            else "the available period"
        )
        verbosity = "detailed" if avg_len > 100 else "brief"

        key_insight = (
            f"Overall career-to-degree alignment is trending {trend_dir} "
            f"across {len(departments)} department(s). "
            f"Analysis is based on {len(feedback_texts)} alumni feedback response(s)."
        )

        trend_analysis = (
            f"Predicted alignment rates span {year_range}. "
            f"Alumni feedback averages {avg_len} characters per response, "
            f"indicating {verbosity} engagement with the survey."
        )

        department_insights = []
        if top_dept:
            department_insights.append(
                f"{top_dept.name} leads with the highest projected growth "
                f"(+{top_dept.change:.1f}%), reaching {top_dept.predicted_rate:.1f}% by {overview[-1].year if overview else 'the target year'}."
            )
        if low_dept and low_dept.code != (top_dept.code if top_dept else None):
            department_insights.append(
                f"{low_dept.name} shows the smallest projected change "
                f"({low_dept.change:+.1f}%), currently at {low_dept.current_rate:.1f}%."
            )

        if payload.selected_department:
            programs = payload.selected_department.programs or []
            if programs:
                department_insights.append(
                    f"{payload.selected_department.name} has {len(programs)} tracked program(s) "
                    f"contributing to its overall alignment trajectory."
                )

        recommendations = [
            "Strengthen industry partnerships for programs with below-average alignment rates.",
            "Use survey verbatim responses to identify specific curriculum gaps.",
        ]
        if low_dept and low_dept.change < 5:
            recommendations.append(
                f"Prioritise targeted intervention for {low_dept.name}, "
                f"which shows minimal projected improvement."
            )

        risk_alert = None
        if low_dept and low_dept.change < 5:
            risk_alert = (
                f"{low_dept.name} has the lowest projected change "
                f"({low_dept.change:+.1f}%) and may require immediate curriculum review."
            )

        return {
            "status": "success",
            "key_insight": key_insight,
            "trend_analysis": trend_analysis,
            "department_insights": department_insights,
            "recommendations": recommendations,
            "risk_alert": risk_alert,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)