import os
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from dotenv import load_dotenv
from supabase import create_client

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing   import OneHotEncoder
from sklearn.cluster         import KMeans
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.base            import clone

warnings.filterwarnings("ignore")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Missing Supabase credentials. Check your .env file.")
    raise SystemExit(1)

if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
    print("  [supabase] SERVICE ROLE key active -- RLS bypassed, all rows visible.")
else:
    print("  [supabase] WARNING: SUPABASE_SERVICE_ROLE_KEY not in .env!")
    print("  [supabase] Falling back to anon key. RLS may hide alumni rows.")
    print("  [supabase] Add SUPABASE_SERVICE_ROLE_KEY to your .env to fix this.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_YEAR       = 2025
END_YEAR        = 2030
PREDICT_YEARS   = list(range(BASE_YEAR, END_YEAR + 1))
N_FOLDS         = 5
MIN_SAMPLES_ML  = 10
DEFAULT_GROWTH  = 1.5

EMPLOYED_STATUSES = {
    "Regular / Permanent", "Contractual", "Probationary",
    "Part-time", "Self-employed", "Casual",
}

ALIGNED_JOB_VALUES = {
    "Yes", "yes", "YES",
    "Very much related", "Related", "Somewhat related",
    "1", "true", "True", "TRUE",
}

_ALIAS_TABLE: dict[str, str] = {
    "BSARCH":          "BSArch",
    "BS ARCH":         "BSArch",
    "BS ARCHITECTURE": "BSArch",
    "BSARCHITECTURE":  "BSArch",

    "BSIT-MWA":        "BSIT-MWA",
    "BSIT MWA":        "BSIT-MWA",
    "BS IT-MWA":       "BSIT-MWA",
    "BS IT MWA":       "BSIT-MWA",

    "BSCE":            "BSCE",
    "BS CE":           "BSCE",
    "BS CIVIL ENGINEERING": "BSCE",

    "BSCPE":           "BSCpE",
    "BS CPE":          "BSCpE",
    "BS COMPUTER ENGINEERING": "BSCpE",

    "BSCS-ML":         "BSCS-ML",
    "BSCS ML":         "BSCS-ML",
    "BS CS-ML":        "BSCS-ML",
    "BS CS ML":        "BSCS-ML",
    "BS COMPUTER SCIENCE": "BSCS-ML",
    "BSCS":            "BSCS-ML",

    "BSPSY":           "BSPSY",
    "BS PSY":          "BSPSY",
    "BSPSYCH":         "BSPSY",
    "BS PSYCH":        "BSPSY",
    "BS PSYCHOLOGY":   "BSPSY",
    "BSPSYCHOLOGY":    "BSPSY",

    "ABCOMM":          "ABComm",
    "AB COMM":         "ABComm",
    "AB COMMUNICATION": "ABComm",
    "ABCOMMUNICATION": "ABComm",
    "BACOMM":          "ABComm",
    "BA COMM":         "ABComm",
    "BA COMMUNICATION": "ABComm",

    "BPED":            "BPEd",
    "B PED":           "BPEd",
    "BP ED":           "BPEd",
    "BPHYSICALEDUCATION": "BPEd",
    "B PHYSICAL EDUCATION": "BPEd",

    "BSBA-MKTGMGT":     "BSBA-MktgMgt",
    "BSBA MKTGMGT":     "BSBA-MktgMgt",
    "BSBA-MM":          "BSBA-MktgMgt",
    "BSBA MM":          "BSBA-MktgMgt",
    "BSBA-MARKETINGMANAGEMENT": "BSBA-MktgMgt",
    "BSBA MARKETING MANAGEMENT": "BSBA-MktgMgt",
    "BSBA-MKTGMGTM":    "BSBA-MktgMgt",
    "BSBA MKTGMGTM":    "BSBA-MktgMgt",
    "BSBA-MKTGMGT M":   "BSBA-MktgMgt",

    "BSBA-FINMGT":     "BSBA-FinMgt",
    "BSBA FINMGT":     "BSBA-FinMgt",
    "BSBA-FM":         "BSBA-FinMgt",
    "BSBA FM":         "BSBA-FinMgt",
    "BSBA-FINANCIALMANAGEMENT": "BSBA-FinMgt",
    "BSBA FINANCIAL MANAGEMENT": "BSBA-FinMgt",

    "BSBA-HRM":        "BSBA-HRM",
    "BSBA HRM":        "BSBA-HRM",
    "BSBA-HR":         "BSBA-HRM",
    "BSBA HR":         "BSBA-HRM",
    "BSBA-HUMANRESOURCEMANAGEMENT": "BSBA-HRM",
    "BSBA HUMAN RESOURCE MANAGEMENT": "BSBA-HRM",

    "BSHM":            "BSHM",
    "BS HM":           "BSHM",
    "BS HOSPITALITY MANAGEMENT": "BSHM",

    "BSMA":            "BSMA",
    "BS MA":           "BSMA",
    "BS MANAGEMENT ACCOUNTING": "BSMA",

    "BSTM":            "BSTM",
    "BS TM":           "BSTM",
    "BS TOURISM MANAGEMENT": "BSTM",

    "BSACCOUNTANCY":   "BSAccountancy",
    "BS ACCOUNTANCY":  "BSAccountancy",
    "BSACC":           "BSAccountancy",
    "BS ACC":          "BSAccountancy",
    "BSA":             "BSAccountancy",
    "BS A":            "BSAccountancy",

    # SHS Strands
    "SHS-STEM":        "SHS-STEM",
    "SHS STEM":        "SHS-STEM",

    "SHS-ABM":         "SHS-ABM",
    "SHS ABM":         "SHS-ABM",

    "SHS-HUMSS":       "SHS-HUMSS",
    "SHS HUMSS":       "SHS-HUMSS",
}


def normalise_program(raw: str) -> str:
    if not raw:
        return raw
    collapsed = " ".join(raw.strip().split())
    key       = collapsed.upper()
    canonical = _ALIAS_TABLE.get(key)
    if canonical:
        return canonical
    return collapsed


def load_dept_table():
    try:
        resp = supabase.table("department_mapping").select(
            "program_name, department_code, keyword_patterns"
        ).execute()
        if resp.data:
            print(f"  [dept-map] Loaded {len(resp.data)} rows from department_mapping.")
            return resp.data
        else:
            print("  [dept-map] department_mapping table is empty.")
    except Exception as exc:
        print(f"  [dept-map] Could not read department_mapping ({exc}).")
    return []


def match_program_to_dept(program_str: str, dept_table: list) -> str | None:
    normalised = program_str.strip()
    upper      = normalised.upper()
    stripped   = upper.replace("BS", "").replace("BA", "").replace("AB", "").strip(" .-")

    ABBREV_MAP: dict[str, str] = {
        "BSARCH":    "SECA",
        "BSIT-MWA":  "SECA",
        "BSCE":      "SECA",
        "BSCPE":     "SECA",
        "BSCS-ML":   "SECA",

        "BSPSY":     "SASE",
        "ABCOMM":    "SASE",
        "BPED":      "SASE",

        "BSBA-MKTGMGT":  "SBMA",
        "BSBA-FINMGT":   "SBMA",
        "BSBA-HRM":      "SBMA",
        "BSHM":          "SBMA",
        "BSMA":          "SBMA",
        "BSTM":          "SBMA",
        "BSACCOUNTANCY": "SBMA",
        "BSBA-MKTGMGTM": "SBMA",

        "SHS-STEM":      "SHS",
        "SHS-ABM":       "SHS",
        "SHS-HUMSS":     "SHS",
    }

    if upper in ABBREV_MAP:
        return ABBREV_MAP[upper]

    for row in dept_table:
        if row["program_name"].upper().strip() == upper:
            return row["department_code"]

    scores = {}
    for row in dept_table:
        code     = row["department_code"]
        keywords = [kw.upper() for kw in (row.get("keyword_patterns") or [])]
        if not keywords:
            continue
        fwd          = sum(1 for kw in keywords if kw in upper)
        rev_full     = sum(1 for kw in keywords if len(upper)   >= 3 and upper   in kw)
        rev_stripped = sum(1 for kw in keywords if len(stripped) >= 3 and stripped in kw)
        rev          = max(rev_full, rev_stripped)
        total        = fwd + rev
        if total > 0:
            prev = scores.get(code, (0, 0))
            if total > prev[1]:
                scores[code] = (2 if fwd >= rev else 3, total)

    if not scores:
        return None
    return max(scores, key=lambda c: scores[c][1])


def cluster_programs_into_depts(alignment_df: pd.DataFrame) -> dict:
    unresolved = alignment_df["program"].unique().tolist()
    hint = (
        "  Layer 1 (OCR/typo): add to _ALIAS_TABLE -> canonical label.\n"
        "  Layer 2 (new program): add to ABBREV_MAP -> dept code.\n"
        "  Layer 3 (no-code): insert into Supabase department_mapping table."
    )
    raise ValueError(
        f"[train_model] Department resolution failed for "
        f"{len(unresolved)} program(s): {unresolved}.\n{hint}"
    )


def assign_departments(alignment_df: pd.DataFrame) -> pd.DataFrame:
    dept_table = load_dept_table()
    dept_map   = {}

    if dept_table:
        for program in alignment_df["program"].unique():
            code = match_program_to_dept(program, dept_table)
            if code:
                dept_map[program] = code
                print(f"  [dept-map] '{program}' -> {code}")
            else:
                print(f"  [dept-map] '{program}' -> no match found, will cluster.")

    unknown = [p for p in alignment_df["program"].unique() if p not in dept_map]
    if unknown:
        print(f"  [dept-map] Clustering {len(unknown)} unmatched program(s): {unknown}")
        cluster_df  = alignment_df[alignment_df["program"].isin(unknown)].copy()
        cluster_map = cluster_programs_into_depts(cluster_df)
        dept_map.update(cluster_map)

    alignment_df = alignment_df.copy()
    alignment_df["department"] = alignment_df["program"].map(dept_map)
    return alignment_df


def build_features(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    fold_train_df: pd.DataFrame | None = None,
    years_ahead: int = 0,
    leave_one_out: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ref_df = fold_train_df if fold_train_df is not None else df

    prog_stats = (
        ref_df.groupby("degree_program")["aligned"]
        .agg(dept_avg="mean", cohort_size="count")
        .reset_index()
    )

    merged = df.merge(prog_stats, on="degree_program", how="left")

    # Leakage guard for the final full-data fit only.
    #
    # When fold_train_df is explicitly provided (the CV/OOF path), `dept_avg`
    # is already computed from the fold's training rows only, so a validation
    # row's own target never contributes to its own feature — that path is
    # unaffected by this block.
    #
    # When fold_train_df is None, ref_df == df, so `dept_avg` above is the
    # group mean of "aligned" computed INCLUDING each row's own target. That
    # is the version used for the final full-data refit of the base models.
    # If leave_one_out is requested here, replace it with each row's
    # leave-one-out program mean (this program's aligned mean, excluding this
    # respondent). Programs with exactly one respondent have no "other"
    # respondents to average over (loo denominator = 0); those rows are left
    # as NaN here and fall back to the existing global-mean fillna below,
    # same as any other unmatched/missing group already does in this function.
    if leave_one_out and fold_train_df is None:
        group_sum  = merged["dept_avg"] * merged["cohort_size"]  # sum of "aligned" for the program
        loo_count  = merged["cohort_size"] - 1
        loo_sum    = group_sum - merged["aligned"]
        merged["dept_avg"] = loo_sum / loo_count.replace(0, np.nan)

    prog_array = merged["degree_program"].values.reshape(-1, 1)
    _ohe_out   = ohe.transform(prog_array)
    prog_ohe   = _ohe_out.toarray() if sp.issparse(_ohe_out) else np.asarray(_ohe_out)

    dept_avg_vals   = merged["dept_avg"].fillna(merged["aligned"].mean()).values * 100
    cohort_sz_vals  = merged["cohort_size"].fillna(1).values.astype(float)
    years_ahead_col = np.full(len(merged), float(years_ahead))

    X = np.hstack([
        prog_ohe,
        dept_avg_vals.reshape(-1, 1),
        cohort_sz_vals.reshape(-1, 1),
        years_ahead_col.reshape(-1, 1),
    ])
    y = merged["aligned"].values.astype(float) * 100

    return X, y


class StackingEnsemble:
    BASE_MODELS = [
        ("random_forest", RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42
        )),
        ("gradient_boosting", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        )),
        ("ridge", Ridge(alpha=1.0)),
    ]

    def __init__(self, n_folds: int = N_FOLDS):
        self.n_folds     = n_folds
        self.base_models = [(name, clone(mdl)) for name, mdl in self.BASE_MODELS]
        self.meta        = Ridge(alpha=1.0)
        self.fitted_base = []

    def fit(self, df: pd.DataFrame, ohe: OneHotEncoder) -> "StackingEnsemble":
        kf      = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        indices = np.arange(len(df))

        # leave_one_out=True: this is the final full-data fit (not a CV fold),
        # so dept_avg is computed leave-one-out to avoid a respondent's own
        # target contributing to their own training feature. y_full (the
        # target column) is unaffected either way.
        X_full, y_full = build_features(df, ohe, fold_train_df=None, years_ahead=0, leave_one_out=True)
        meta_X         = np.zeros((len(df), len(self.base_models)))

        for col, (name, mdl) in enumerate(self.base_models):
            oof = np.zeros(len(df))
            for train_idx, val_idx in kf.split(indices):
                fold_train_df = df.iloc[train_idx].copy()
                fold_val_df   = df.iloc[val_idx].copy()

                X_tr, y_tr = build_features(
                    fold_train_df, ohe,
                    fold_train_df=fold_train_df, years_ahead=0
                )
                X_val, _ = build_features(
                    fold_val_df, ohe,
                    fold_train_df=fold_train_df,
                    years_ahead=0
                )

                mdl_clone = clone(mdl)
                mdl_clone.fit(X_tr, y_tr)
                oof[val_idx] = np.clip(mdl_clone.predict(X_val), 0, 100)

            meta_X[:, col] = oof
            print(f"    [{name}] OOF MAE: {mean_absolute_error(y_full, oof):.3f} | "
                  f"R2: {r2_score(y_full, oof):.3f}")

        self.fitted_base = []
        for name, mdl in self.base_models:
            mdl.fit(X_full, y_full)
            self.fitted_base.append((name, mdl))

        self.meta.fit(meta_X, y_full)
        meta_preds = self.meta.predict(meta_X)
        print(f"    [meta-learner] Train MAE : {mean_absolute_error(y_full, meta_preds):.3f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_X = np.column_stack([
            np.clip(mdl.predict(X), 0, 100)
            for _, mdl in self.fitted_base
        ])
        return np.clip(self.meta.predict(meta_X), 0, 100)

    def predict_single(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])


def compute_alignment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("  [align] No valid survey records found. Nothing to predict.")
        return pd.DataFrame(columns=["program", "alignment"])

    grouped = df.groupby("degree_program")["aligned"].mean() * 100
    result  = grouped.reset_index().rename(
        columns={"degree_program": "program", "aligned": "alignment"}
    )
    for _, row in result.iterrows():
        n = len(df[df["degree_program"] == row["program"]])
        print(f"  [align] '{row['program']}': {row['alignment']:.1f}%  (n={n} respondents)")
    return result


def infer_growth_rates(alignment_df: pd.DataFrame) -> dict:
    dept_groups = alignment_df.groupby("department")["alignment"]
    rates = {}
    for dept, vals in dept_groups:
        std  = vals.std(ddof=0) if len(vals) > 1 else 0.0
        rate = round(min(3.0, max(0.5, 1.0 + 0.05 * std)), 2)
        rates[dept] = rate
        print(f"  [growth] {dept}: {rate}% / year  (std={std:.2f})")
    return rates


def main():
    print("=" * 60)
    print("  Stacking Ensemble - Alumni Alignment Predictor")
    print("=" * 60)

    print("\n[1/8] Fetching data from Supabase ...")
    # Ordered by user_id (survey_progress's stable per-alumni key, already used
    # elsewhere in this codebase — see main.py's survey_progress reads) so the
    # same unchanged dataset is presented to KFold/RF/GB in the same row order
    # on every run. Row order was previously undefined, which meant a fixed
    # random_state did not guarantee reproducible training.
    resp = supabase.table("survey_progress").select(
        "educational_background_data",
        "employment_information_data"
    ).order("user_id", desc=False).execute()
    rows = resp.data
    print(f"      {len(rows)} rows retrieved.")

    print("[2/8] Parsing records from survey_progress ...")
    records = []
    skipped = 0
    shs_skipped = 0
    for row in rows:
        edu    = row.get("educational_background_data") or {}
        emp    = row.get("employment_information_data") or {}
        degree = edu.get("degree_program")
        status = emp.get("employment_status")

        if not degree:
            skipped += 1
            print(f"      [skip] row has no degree_program")
            continue

        raw_degree = degree
        degree     = normalise_program(degree)

        if degree != raw_degree.strip():
            print(f"      [normalise] '{raw_degree.strip()}' -> '{degree}'")

        if degree in ("SHS-STEM", "SHS-ABM", "SHS-HUMSS"):
            shs_skipped += 1
            print(f"      [skip] row excluded — SHS program ({degree}) out of College Predictive Analytics scope")
            continue

        grad_year = None
        raw_year  = edu.get("year_graduated")
        if raw_year:
            try:
                grad_year = int(raw_year)
            except (ValueError, TypeError):
                pass

        is_employed     = status in EMPLOYED_STATUSES if status else False
        job_related_raw = emp.get("job_related_to_degree")
        is_related      = str(job_related_raw).strip() in ALIGNED_JOB_VALUES if job_related_raw else False
        aligned_flag    = 1 if (is_employed and is_related) else 0

        print(f"      [record] program={repr(degree)} | "
              f"employed={is_employed} | "
              f"job_related={repr(job_related_raw)} | "
              f"aligned={aligned_flag}")

        records.append({
            "degree_program":        degree,
            "year_graduated":        grad_year,
            "employment_status":     status,
            "job_related_to_degree": job_related_raw,
            "aligned":               aligned_flag,
        })

    df = pd.DataFrame(records)
    print(f"      {len(df)} usable records  |  {skipped} skipped (no degree_program)  |  "
          f"{shs_skipped} skipped (SHS out of scope).")

    if df.empty:
        print("      No usable records found. Exiting without updating predictions.")
        return

    print("[3/8] Computing per-program alignment rates ...")
    alignment_df = compute_alignment(df)
    if alignment_df.empty:
        print("      alignment_df is empty after computation. Exiting.")
        return
    print(alignment_df.to_string(index=False))

    print("[4/8] Resolving department assignments ...")
    alignment_df = assign_departments(alignment_df)
    print(alignment_df.to_string(index=False))

    print("[5/8] Inferring department growth rates ...")
    growth_rates = infer_growth_rates(alignment_df)

    dept_alignment = (
        alignment_df.groupby("department")["alignment"]
        .mean()
        .reset_index()
    )

    print("[6/8] Training Stacking Ensemble ...")
    ensemble     = None
    ohe          = None
    use_ensemble = len(df) >= MIN_SAMPLES_ML

    if use_ensemble:
        print(f"      Sufficient data ({len(df)} records >= {MIN_SAMPLES_ML}). Using ML stack.")

        all_programs = df["degree_program"].unique().reshape(-1, 1)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(all_programs)
        print(f"      OneHotEncoder fitted on {len(ohe.categories_[0])} programs.")

        X_full, y_full = build_features(df, ohe, fold_train_df=None, years_ahead=0)
        print(f"      Feature matrix: {X_full.shape}  |  "
              f"Target range: [{y_full.min():.1f}, {y_full.max():.1f}]")

        ensemble = StackingEnsemble(n_folds=min(N_FOLDS, len(df)))
        ensemble.fit(df, ohe)
        print("      Ensemble training complete.")
    else:
        print(f"      Insufficient data ({len(df)} records < {MIN_SAMPLES_ML}). "
              f"Using rule-based projection.")

    print("[7/8] Generating predictions (2025 -> 2030) ...")
    predictions = []

    for _, arow in alignment_df.iterrows():
        program    = arow["program"]
        department = arow["department"]
        base_rate  = round(float(arow["alignment"]), 2)
        growth     = growth_rates.get(department, DEFAULT_GROWTH)

        for year in PREDICT_YEARS:
            years_ahead = year - BASE_YEAR

            if use_ensemble and ensemble is not None and ohe is not None:
                prog_cohort = df[df["degree_program"] == program]
                cohort_sz   = len(prog_cohort) if not prog_cohort.empty else 1

                _pred_ohe    = ohe.transform([[program]])
                prog_ohe_vec = (_pred_ohe.toarray() if sp.issparse(_pred_ohe) else np.asarray(_pred_ohe))[0]
                dept_avg_val    = np.array([base_rate])
                cohort_sz_val   = np.array([float(cohort_sz)])
                years_ahead_val = np.array([float(years_ahead)])

                feat = np.hstack([prog_ohe_vec, dept_avg_val,
                                  cohort_sz_val, years_ahead_val])

                ml_pred        = ensemble.predict_single(feat)
                rule_pred      = min(base_rate + growth * years_ahead, 100.0)
                predicted_rate = round(0.70 * ml_pred + 0.30 * rule_pred, 2)
            else:
                predicted_rate = round(min(base_rate + growth * years_ahead, 100.0), 2)

            predicted_rate = max(0.0, min(100.0, predicted_rate))
            predictions.append({
                "program":        program,
                "department":     department,
                "year":           year,
                "predicted_rate": predicted_rate,
                "current_rate":   base_rate,
            })

    print(f"      {len(predictions)} prediction rows generated.")

    print("[8/8] Saving model artefact and pushing predictions ...")
    model_data = {
        "programs":      alignment_df.to_dict(orient="records"),
        "departments":   dept_alignment.to_dict(orient="records"),
        "growth_rates":  growth_rates,
        "ensemble":      ensemble,
        "ohe":           ohe,
        "use_ensemble":  use_ensemble,
    }
    joblib.dump(model_data, "model.pkl")
    print("      model.pkl saved.")

    supabase.table("predictions").delete().neq("year", -1).execute()
    print("      Stale predictions cleared.")

    for pred in predictions:
        supabase.table("predictions").upsert(pred).execute()
    print("      New predictions pushed to Supabase.")

    print("\n[OK]  Done! Stacking ensemble predictions saved (2025 -> 2030).\n")


if __name__ == "__main__":
    main()