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
from pydantic import BaseModel as PydanticBaseModel
import csv
import io
import uuid
from typing import Any
import json as _json

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
supabase_admin = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
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

# ─── SURVEY CONFIG MODELS ────────────────────────────────────────────────────
class SurveyConfigPayload(PydanticBaseModel):
    config: dict


# ─── SURVEY CONFIG ENDPOINTS ─────────────────────────────────────────────────
@app.get("/api/admin/survey-config")
def get_survey_config(survey_type: str = "college"):
    try:
        if survey_type == "shs":
            resp = (
                supabase_admin.table("survey_config")
                .select("id, config")
                .contains("config", {"survey_type": "shs"})
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
        else:
            resp = (
                supabase_admin.table("survey_config")
                .select("id, config")
                .or_("config->>survey_type.is.null,config->>survey_type.eq.college")
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )

        rows = resp.data or []
        if not rows:
            return {"data": None}
        return {"data": rows[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/survey-config")
def create_survey_config(payload: SurveyConfigPayload):
    try:
        resp = (
            supabase_admin.table("survey_config")
            .insert({"config": payload.config})
            .execute()
        )
        row = resp.data[0] if resp.data else None
        return {"data": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/survey-config/{config_id}")
def update_survey_config(config_id: str, payload: SurveyConfigPayload):
    try:
        from datetime import datetime, timezone
        resp = (
            supabase_admin.table("survey_config")
            .update({
                "config": payload.config,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", config_id)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"UPDATE matched 0 rows for id={config_id}",
            )
        return {"data": rows[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── ADMIN ACCOUNT MANAGEMENT MODELS ─────────────────────────────────────────
class CreateAdminRequest(PydanticBaseModel):
    email: str
    first_name: str
    last_name: str
    role: str
    module_permissions: dict


class UpdatePermissionsRequest(PydanticBaseModel):
    module_permissions: dict


# ─── ADMIN ACCOUNT MANAGEMENT ENDPOINTS ──────────────────────────────────────
@app.post("/api/admin/create-admin")
def create_admin(payload: CreateAdminRequest):
    try:
        invite_resp = supabase_admin.auth.admin.invite_user_by_email(
            payload.email,
            {
                "data": {
                    "first_name": payload.first_name,
                    "last_name": payload.last_name,
                    "role": payload.role,
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    uid = getattr(getattr(invite_resp, "user", None), "id", None)
    if not uid:
        raise HTTPException(status_code=400, detail="Admin invite failed.")

    try:
        supabase_admin.table("users").insert({
            "id": uid,
            "email": payload.email,
            "first_name": payload.first_name,
            "last_name": payload.last_name,
            "role": payload.role,
            "account_status": "active",
            "module_permissions": payload.module_permissions,
        }).execute()
    except Exception as e:
        supabase_admin.auth.admin.delete_user(uid)
        raise HTTPException(status_code=500, detail=str(e))

    return {"uid": uid}


@app.patch("/api/admin/alumni/{user_id}/permissions")
def update_admin_permissions(user_id: str, payload: UpdatePermissionsRequest):
    try:
        supabase_admin.table("users").update(
            {"module_permissions": payload.module_permissions}
        ).eq("id", user_id).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/accounts/stats")
def get_admin_stats():
    try:
        tot = supabase_admin.table("users").select("id", count="exact") \
            .in_("role", ["admin", "superadmin"]).execute()
        act = supabase_admin.table("users").select("id", count="exact") \
            .in_("role", ["admin", "superadmin"]).eq("account_status", "active").execute()
        inact = supabase_admin.table("users").select("id", count="exact") \
            .in_("role", ["admin", "superadmin"]).eq("account_status", "inactive").execute()
        dis = supabase_admin.table("users").select("id", count="exact") \
            .in_("role", ["admin", "superadmin"]).eq("account_status", "disabled").execute()

        return {
            "data": {
                "total": tot.count or 0,
                "active": act.count or 0,
                "inactive": inact.count or 0,
                "disabled": dis.count or 0,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/accounts")
def get_admin_accounts(
    page: int = 1,
    per_page: int = 10,
    role_filter: str = "All Roles",
    status_filter: str = "All Status",
    search: str = "",
):
    try:
        q = (
            supabase_admin.table("users")
            .select(
                "id, first_name, last_name, email, role, account_status, created_at, module_permissions",
                count="exact",
            )
            .in_("role", ["admin", "superadmin"])
            .order("created_at", desc=True)
        )

        if role_filter == "Admin":
            q = q.eq("role", "admin")
        elif role_filter == "Super Admin":
            q = q.eq("role", "superadmin")

        if status_filter == "Active":
            q = q.eq("account_status", "active")
        elif status_filter == "Inactive":
            q = q.eq("account_status", "inactive")
        elif status_filter == "Disabled":
            q = q.eq("account_status", "disabled")

        if search.strip():
            q = q.or_(
                f"first_name.ilike.%{search}%,last_name.ilike.%{search}%,email.ilike.%{search}%"
            )

        start = (page - 1) * per_page
        end = page * per_page - 1
        resp = q.range(start, end).execute()

        data = resp.data or []
        count = resp.count or 0

        ids = [u["id"] for u in data]
        last_logins = {}
        if ids:
            login_resp = (
                supabase_admin.table("audit_logs")
                .select("user_id, created_at")
                .eq("action", "Login")
                .eq("status", "Success")
                .in_("user_id", ids)
                .order("created_at", desc=True)
                .execute()
            )
            for l in (login_resp.data or []):
                if l["user_id"] not in last_logins:
                    last_logins[l["user_id"]] = l["created_at"]

        result = [
            {**u, "last_login": last_logins.get(u["id"])}
            for u in data
        ]

        return {"data": result, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AlumniCSVRow(PydanticBaseModel):
    email: str
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    program: str = ""
    batch_year: Optional[int] = None
    account_status: str = "active"


class UploadAlumniRequest(PydanticBaseModel):
    rows: list[AlumniCSVRow]


@app.post("/api/admin/alumni/bulk-upload")
def bulk_upload_alumni(payload: UploadAlumniRequest):
    inserted = 0
    skipped = 0
    errors = []

    for row in payload.rows:
        if not row.email:
            skipped += 1
            continue

        existing = supabase_admin.table("users").select("id").eq("email", row.email).maybe_single().execute()
        if existing.data:
            skipped += 1
            continue

        try:
            auth_resp = supabase_admin.auth.admin.create_user({
                "email": row.email,
                "password": str(uuid.uuid4()),
                "email_confirm": True,
            })
        except Exception as e:
            errors.append({"email": row.email, "message": str(e)})
            continue

        auth_id = auth_resp.user.id

        try:
            supabase_admin.table("users").insert({
                "id": auth_id,
                "email": row.email,
                "first_name": row.first_name,
                "middle_name": row.middle_name,
                "last_name": row.last_name,
                "program": row.program,
                "batch_year": row.batch_year,
                "account_status": row.account_status,
                "role": "alumni",
            }).execute()
            inserted += 1
        except Exception as e:
            supabase_admin.auth.admin.delete_user(auth_id)
            errors.append({"email": row.email, "message": str(e)})

    return {"inserted": inserted, "skipped": skipped, "errors": errors}

class UpdateStatusRequest(PydanticBaseModel):
    status: str


@app.get("/api/admin/alumni")
def get_alumni():
    try:
        all_users = []
        page_size = 1000
        start = 0
        while True:
            resp = supabase_admin.table("users").select(
                "id, email, first_name, middle_name, last_name, program, batch_year, account_status, role"
            ).eq("role", "alumni").range(start, start + page_size - 1).execute()
            batch = resp.data or []
            all_users.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size

        all_surveys = []
        start = 0
        while True:
            resp = supabase_admin.table("survey_progress").select(
                "user_id, completed, percentage, employment_information_data"
            ).range(start, start + page_size - 1).execute()
            batch = resp.data or []
            all_surveys.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size

        return {"data": {"users": all_users, "surveys": all_surveys}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/admin/alumni/{user_id}/status")
def update_alumni_status(user_id: str, payload: UpdateStatusRequest):
    try:
        supabase_admin.table("users").update(
            {"account_status": payload.status}
        ).eq("id", user_id).execute()
        return {"success": True}
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