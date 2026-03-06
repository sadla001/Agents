from __future__ import annotations

import math
import os
import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.adk.agents import LlmAgent  # ADK agent class

from .data.data_loader import load_trial_data, TrialData
from .rules_engine.rules import apply_rules


load_dotenv()


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python so Pydantic can serialize them."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _to_dt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def evaluate_subject(subject_id: str) -> Dict[str, Any]:
    """
    Deterministic Layer-1 tool:
    - Loads SDTM domains + PROTOCOL_RULES
    - Computes a small set of derived fields
    - Applies domain rules to produce actions + audit hits
    Returns JSON-serializable dict for the LLM to explain.
    """
    data: TrialData = load_trial_data()

    rs = data.rs.copy()
    ae = data.ae.copy()
    lb = data.lb.copy()
    ex = data.ex.copy()
    rules_df = data.protocol_rules.copy()

    # ---- minimal typing ----
    rs = _to_dt(rs, ["VISIT_DT", "IMAGING_DATE"])
    rs = _to_num(rs, ["BASELINE_SLD_mm", "CURRENT_SLD_mm", "NADIR_SLD_mm"])
    ae = _to_dt(ae, ["AESTDTC", "AEENDTC"])
    lb = _to_dt(lb, ["LBDTC"])
    lb = _to_num(lb, ["LBORRES", "LBNRLO", "LBNRHI"])
    ex = _to_dt(ex, ["SCHEDULED_DT", "ACTUAL_DT"])
    ex = _to_num(ex, ["WINDOW_DAYS", "DOSE_LEVEL_MG", "STANDARD_DOSE_MG"])

    # ---- resolve subject ID flexibly ----
    # Extract just the numeric digits from the user's input (e.g. "patient 001", "PT-001", "1" → "001")
    _digits = re.sub(r"[^0-9]", "", str(subject_id))
    all_ids: list[str] = []
    for _df in (rs, ae, lb, ex):
        if "USUBJID" in _df.columns:
            all_ids = sorted(_df["USUBJID"].dropna().astype(str).unique().tolist())
            break
    # Find the dataset ID whose trailing digits match (ignore leading zeros)
    resolved_id = subject_id
    if subject_id not in all_ids and _digits:
        for _id in all_ids:
            if re.sub(r"[^0-9]", "", _id).lstrip("0") == _digits.lstrip("0"):
                resolved_id = _id
                break

    def sub(df: pd.DataFrame) -> pd.DataFrame:
        if "USUBJID" not in df.columns:
            return df.iloc[0:0].copy()
        return df[df["USUBJID"].astype(str) == str(resolved_id)].copy()

    rs_s = sub(rs)
    ae_s = sub(ae)
    lb_s = sub(lb)
    ex_s = sub(ex)

    # ---- validate subject exists ----
    if rs_s.empty and ae_s.empty and lb_s.empty and ex_s.empty:
        return {
            "error": f"Subject '{subject_id}' not found in the dataset.",
            "available_subjects": all_ids,
        }

    # ---- derive a few fields (examples) ----
    context: Dict[str, Any] = {"USUBJID": subject_id}

    # Latest RS snapshot
    if not rs_s.empty:
        # pick latest by IMAGING_DATE if present else VISIT_DT
        date_col = "IMAGING_DATE" if "IMAGING_DATE" in rs_s.columns else ("VISIT_DT" if "VISIT_DT" in rs_s.columns else None)
        rs_latest = rs_s.sort_values(date_col, ascending=True).iloc[-1] if date_col else rs_s.iloc[-1]

        # computed metrics if columns exist
        b = rs_latest.get("BASELINE_SLD_mm")
        c = rs_latest.get("CURRENT_SLD_mm")
        n = rs_latest.get("NADIR_SLD_mm")

        if pd.notna(b) and pd.notna(c) and float(b) != 0:
            context["SLD_REDUCTION_PCT"] = (float(b) - float(c)) / float(b) * 100.0
        if pd.notna(n) and pd.notna(c) and float(n) != 0:
            context["SLD_INCREASE_FROM_NADIR"] = (float(c) - float(n)) / float(n) * 100.0

        # also pass through common RS fields if present (rules may reference them)
        for k in ["NEW_LESION", "NON_TARGET_PD", "VISIT", "IMAGING_DATE", "VISIT_DT"]:
            if k in rs_latest.index:
                context[k] = rs_latest.get(k)

    # AE summary — map text severity to CTCAE grade number
    if not ae_s.empty:
        _sev_map = {"mild": 1, "moderate": 2, "severe": 3, "life-threatening": 4, "fatal": 5}
        grade = None
        for grade_col in ["CTCAE_GRADE", "AETOXGR"]:
            if grade_col in ae_s.columns:
                g = pd.to_numeric(ae_s[grade_col], errors="coerce")
                if g.notna().any():
                    grade = float(g.max())
                    break
        if grade is None and "AESEV" in ae_s.columns:
            mapped = ae_s["AESEV"].str.lower().map(_sev_map)
            if mapped.notna().any():
                grade = float(mapped.max())
        context["CTCAE_GRADE"] = grade
        context["AE_COUNT"] = int(len(ae_s))

    # LB — per-test context fields matching rule PARAMETER names
    if not lb_s.empty and "LBTEST" in lb_s.columns and "LBORRES" in lb_s.columns:
        for lbtest, ctx_key, use_ratio in [
            ("Hemoglobin", "HEMOGLOBIN", False),
            ("Creatinine", "CREATININE_ULN_RATIO", True),
            ("ALT", "ALT_ULN_RATIO", True),
        ]:
            mask = lb_s["LBTEST"].astype(str).str.strip() == lbtest
            sub_lb = lb_s[mask]
            if sub_lb.empty:
                continue
            vals = pd.to_numeric(sub_lb["LBORRES"], errors="coerce")
            if use_ratio and "LBNRHI" in sub_lb.columns:
                uln = pd.to_numeric(sub_lb["LBNRHI"], errors="coerce")
                ratio = vals / uln
                if ratio.notna().any():
                    context[ctx_key] = float(ratio.max())
            elif not use_ratio and vals.notna().any():
                context[ctx_key] = float(vals.min())  # lowest Hgb is worst

    # EX adherence — use PARAMETER names matching rule sheet
    if not ex_s.empty and "SCHEDULED_DT" in ex_s.columns and "ACTUAL_DT" in ex_s.columns:
        dev = (ex_s["ACTUAL_DT"] - ex_s["SCHEDULED_DT"]).dt.days
        if dev.notna().any():
            context["DEVIATION_DAYS"] = float(dev.abs().max())
    if not ex_s.empty and "DOSE_LEVEL_MG" in ex_s.columns and "STANDARD_DOSE_MG" in ex_s.columns:
        compliance = ex_s["DOSE_LEVEL_MG"] / ex_s["STANDARD_DOSE_MG"] * 100
        if compliance.notna().any():
            context["COMPLIANCE_PCT"] = float(compliance.min())

    # ---- per-visit classification → sustained visit count ----
    # Classify every RS visit individually, then count how many consecutive
    # trailing visits share the same classification as the latest visit.
    # SUSTAINED_VISITS feeds RULE-004 (unconfirmed flag).
    # SD_DURATION feeds RULE-005 (stable disease fallback).
    if not rs_s.empty:
        date_col_sv = "IMAGING_DATE" if "IMAGING_DATE" in rs_s.columns else (
            "VISIT_DT" if "VISIT_DT" in rs_s.columns else None)
        rs_sorted = rs_s.sort_values(date_col_sv, ascending=True) if date_col_sv else rs_s

        def _classify_visit(row: pd.Series) -> str:
            """Per-visit RECIST label using the same logic as the rules engine."""
            b_ = float(row.get("BASELINE_SLD_mm") or 0)
            c_ = float(row.get("CURRENT_SLD_mm") or 0)
            n_ = float(row.get("NADIR_SLD_mm") or 0)
            nl = str(row.get("NEW_LESION") or "No").strip()
            sld  = (b_ - c_) / b_ * 100.0 if b_ != 0 else 0.0
            nadir = (c_ - n_) / n_ * 100.0 if n_ != 0 else 0.0
            if sld == 100.0:
                return "CR"
            if sld >= 30.0 and nl.lower() in ("yes", "y"):
                return "MIXED"
            if sld >= 30.0:
                return "PR"
            if nadir >= 20.0 or nl.lower() in ("yes", "y"):
                return "PD"
            return "SD"

        visit_classes = [_classify_visit(row) for _, row in rs_sorted.iterrows()]
        latest_cls    = visit_classes[-1]

        # Count consecutive trailing visits that match the latest classification
        sustained = 0
        for cls in reversed(visit_classes):
            if cls == latest_cls:
                sustained += 1
            else:
                break

        context["SUSTAINED_VISITS"] = sustained
        context["SD_DURATION"]      = sustained   # RULE-005 reads SD_DURATION

    # ---- apply rules by domain (best-effort) ----
    fired_rules = []
    actions = []

    # You can rename these domains to match your PROTOCOL_RULES "DOMAIN" values
    for domain in ["RECIST", "AE", "LAB", "VISIT", "SAFETY"]:
        hits, acts = apply_rules(rules_df, domain=domain, context=context)
        fired_rules.extend([h.__dict__ for h in hits])
        actions.extend(acts)

    # Classification selection:
    # - If RECIST priority-1 rules produce both a response (CR/PR) AND progression (PD),
    #   it is a mixed response — report all of them, do not pick by priority.
    # - Otherwise, pick the highest-priority CLASSIFY_ action (lowest priority number wins).
    #   Within the same priority level, the last fired rule wins (e.g. CR beats PR).
    def _cls_core(action: str) -> str:
        """First word of the classification, e.g. 'CR', 'PR', 'PD', 'SD'."""
        return str(action).split("_", 1)[1].split()[0].upper()

    recist_p1_classify = [
        h for h in fired_rules
        if h["domain"] == "RECIST" and h["priority"] == 1
        and str(h["action"]).upper().startswith("CLASSIFY_")
    ]
    recist_cores = {_cls_core(h["action"]) for h in recist_p1_classify}
    is_mixed = bool(recist_cores & {"CR", "PR"}) and bool(recist_cores & {"PD"})

    classification = None
    classify_hits = [
        h for h in fired_rules if str(h["action"]).upper().startswith("CLASSIFY_")
    ]
    if is_mixed:
        classification = "MIXED: " + " + ".join(sorted(recist_cores))
    elif classify_hits:
        best = min(classify_hits, key=lambda h: (h["priority"], -fired_rules.index(h)))
        classification = str(best["action"]).split("_", 1)[1].upper()

    # ---- confirmed vs unconfirmed flag ----
    # RULE-004 fires FLAG_UNCONFIRMED_RESPONSE when SUSTAINED_VISITS < 2
    # for a CR or PR classification. Expose this cleanly so the LLM can
    # surface it in the Status line without having to parse fired_rules itself.
    is_unconfirmed = any(
        "FLAG_UNCONFIRMED_RESPONSE" in str(h["action"])
        for h in fired_rules
    )
    # Determine minimum confirmation visits required for the current classification
    _confirmation_required = {"CR": 3, "PR": 2}
    _cls_short = (classification or "").split(":")[0].split()[0].upper().rstrip("+")
    confirmation_visits_needed = _confirmation_required.get(_cls_short, None)

    result = {
        "subject_id": subject_id,
        "classification": classification,
        "confirmed": not is_unconfirmed,
        "unconfirmed_reason": (
            f"Only {context.get('SUSTAINED_VISITS', 0)} consecutive visit(s) show "
            f"{_cls_short}; protocol requires "
            f"{confirmation_visits_needed} to confirm (RULE-004)."
        ) if is_unconfirmed and confirmation_visits_needed else None,
        "context_used": context,
        "actions": actions,
        "fired_rules": fired_rules,
        "evidence_counts": {
            "rs_rows": int(len(rs_s)),
            "ae_rows": int(len(ae_s)),
            "lb_rows": int(len(lb_s)),
            "ex_rows": int(len(ex_s)),
        },
        "evidence_preview": {
            "rs_latest": rs_s.sort_values("IMAGING_DATE").tail(1).to_dict("records") if ("IMAGING_DATE" in rs_s.columns and not rs_s.empty) else [],
            "ae_head": ae_s.head(5).to_dict("records") if not ae_s.empty else [],
            "lb_head": lb_s.head(5).to_dict("records") if not lb_s.empty else [],
            "ex_head": ex_s.head(5).to_dict("records") if not ex_s.empty else [],
        },
    }
    return _sanitize(result)


INSTRUCTION = """You are a Trial Investigator assistant.
You MUST call the tool `evaluate_subject(subject_id)` before answering.
Do NOT invent values or infer missing data.
If the tool returns an "error" field, tell the user the subject was not found and list the available_subjects.

════════════════════════════════════════
OUTPUT RULES (internal — do NOT print)
════════════════════════════════════════

MAIN — always fixed, never changes regardless of the question:
- Tumor size: use SLD_REDUCTION_PCT from context_used, rounded to nearest whole number.
  Positive → "Tumor size reduction: X%"
  Negative → "Tumor size increase: X%" (absolute value, never print both words)
- Sustained over: use SUSTAINED_VISITS from context_used.
  Use "visit" if value is 1, "visits" if greater than 1.
- Adverse events: if AE_COUNT is 0 → "None recorded".
  If AE_COUNT > 0 → list each AE from ae_head as "AETERM (Grade N)"
  where N comes from AESEV: Mild=1, Moderate=2, Severe=3, Life-threatening=4, Fatal=5.
  Separate multiple AEs with commas.

STATUS — always fixed, never changes regardless of the question:
- Use the classification field.
  If confirmed is false → append "(Unconfirmed)" then on the next line write "⚠ " + unconfirmed_reason exactly.
  If classification starts with "MIXED:" → write "Mixed Response: PR + PD — conflicting signals, escalate to monitor".
  If classification is null → write "No classification yielded by rule table".

AI REASONING — read the user's question and answer it directly using the tool data:
- Use your own judgment to determine what is relevant to the question.
- Only use data present in the tool response (context_used, fired_rules, evidence_preview, evidence_counts).
- Never invent or infer values not in the tool output.
- Always cite specific rule IDs from fired_rules and specific values from context_used.
- The tool response contains: tumor measurements (rs), adverse events (ae), lab results (lb),
  dosing/visits (ex), fired protocol rules, and all computed context values.
- If no specific question is asked, default to explaining the classification and which rules drove it.

════════════════════════════════════════
OUTPUT FORMAT — respond ONLY in this exact structure:
════════════════════════════════════════

Clinical Trial Response Summary - Subject <subject_id>

Main:
    Tumor size reduction: <X>%
    Sustained over: <SUSTAINED_VISITS> consecutive visit(s)
    Adverse events: <ae_summary>

Status:
    <classification>
    <unconfirmed_reason>   ← only if confirmed is false

AI Reasoning:
    <answer the user's actual question using tool data, citing rule IDs and specific values>"""

# ADK agent root object (ADK Dev UI / runner imports this file and looks for `root_agent` often)
root_agent = LlmAgent(
    name="trial_investigator_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=INSTRUCTION,
    tools=[evaluate_subject],  # ADK auto-wraps as FunctionTool
)