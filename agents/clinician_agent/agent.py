from __future__ import annotations
import json
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
 
from .config import (
    GOOGLE_API_KEY, LLM_MODEL, PROTOCOL_TABLE_PATH,
    DATASET_PATH, LOINC, download_xlsx_files,
)
from .data import PatientDataLoader
from .rules import ProtocolRulesEngine
 
# Download xlsx files from GCS before loading (no-op when local)
download_xlsx_files()
 
# ── Singletons loaded once at startup ────────────────────────────────
print("Loading protocol rules...")
RULES_ENGINE = ProtocolRulesEngine(PROTOCOL_TABLE_PATH)
print(f"  {RULES_ENGINE.total_rules} rules loaded")
 
print("Loading patient dataset...")
DATA_LOADER = PatientDataLoader(DATASET_PATH)
print(f"  {DATA_LOADER.count()} patients: {DATA_LOADER.all_ids()}")
 
 
# ── Helper ───────────────────────────────────────────────────────────
def _f(v) -> float:
    try: return float(v) if v is not None else 0.0
    except (TypeError, ValueError): return 0.0
 
 
# ══════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS  (plain Python — ADK wraps them via FunctionTool)
# ══════════════════════════════════════════════════════════════════════
 
def get_patient_data(patient_id: str) -> str:
    """Fetch FHIR observations, medications and symptom notes for a patient."""
    record = DATA_LOADER.get(patient_id)
    if not record:
        return json.dumps({"error": f"Patient {patient_id!r} not found.",
                           "available": DATA_LOADER.all_ids()})
    obs = {}
    for name, loinc_code in LOINC.items():
        row = record.latest_obs(loinc_code, "FOLLOWUP")
        if row:
            obs[name] = {"value": _f(row.get("VALUE")),
                         "baseline_value": _f(row.get("BASELINE_VALUE")),
                         "nadir_value": _f(row.get("NADIR_VALUE"))}
    med  = record.current_med() or {}
    symp = record.latest_symptom() or {}
    return json.dumps({
        "patient_id": patient_id, "observations": obs,
        "medication": {
            "name": med.get("MEDICATION_NAME"),
            "dose_mg": _f(med.get("DOSE_MG")),
            "original_dose_mg": _f(med.get("ORIGINAL_DOSE_MG")),
            "dose_reduction_count": med.get("DOSE_REDUCTION_COUNT"),
        },
        "symptom_note": {
            "ecog_baseline": _f(symp.get("ECOG_BASELINE")),
            "ecog_current":  _f(symp.get("ECOG_CURRENT")),
            "pro_baseline":  _f(symp.get("PRO_SCORE_BASELINE")),
            "pro_current":   _f(symp.get("PRO_SCORE_CURRENT")),
        },
    }, default=str)
 
 
def evaluate_symptom_burden(patient_data_json: str) -> str:
    """Apply SB-01 through SB-05 rules to classify symptom burden."""
    try:
        s = json.loads(patient_data_json).get("symptom_note", {})
        r = RULES_ENGINE.evaluate_symptom_burden(
            _f(s.get("ecog_baseline")), _f(s.get("ecog_current")),
            _f(s.get("pro_baseline")),  _f(s.get("pro_current")))
        return json.dumps({"output": r.output, "rule_id": r.rule_id,
                           "severity": r.severity, "flag": r.flag})
    except Exception as e:
        return json.dumps({"error": str(e)})
def evaluate_tumor_markers(patient_data_json: str) -> str:
    """Apply TM-01 through TM-05 rules to classify CA-125 trend."""
    try:
        ca = json.loads(patient_data_json).get("observations", {}).get("CA-125", {})
        r = RULES_ENGINE.evaluate_tumor_markers(
            _f(ca.get("baseline_value")), _f(ca.get("value")))
        return json.dumps({"output": r.output, "rule_id": r.rule_id,
                           "severity": r.severity, "flag": r.flag})
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
def evaluate_renal_function(patient_data_json: str) -> str:
    """Apply RF-01 through RF-05 rules to classify renal function."""
    try:
        egfr = json.loads(patient_data_json).get("observations", {}).get("eGFR", {})
        baseline = _f(egfr.get("baseline_value"))
        current  = _f(egfr.get("value"))
        nadir    = _f(egfr.get("nadir_value")) or current
        r = RULES_ENGINE.evaluate_renal_function(baseline, current, nadir)
        return json.dumps({"output": r.output, "rule_id": r.rule_id,
                           "severity": r.severity, "flag": r.flag,
                           "decline_pct": r.decline_pct})
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
def run_alert_engine(
    sb_result_json: str, tm_result_json: str, rf_result_json: str
) -> str:
    """Apply CA-01 through CA-10 alert rules from the domain flags."""
    try:
        flags = {}
        for raw in (sb_result_json, tm_result_json, rf_result_json):
            f = json.loads(raw).get("flag")
            if f: flags[f] = True
        r = RULES_ENGINE.run_alert_engine(flags)
        return json.dumps({"alert_text": r.alert_text, "rule_id": r.rule_id,
                           "suppressed": r.suppressed})
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
# ══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════
 
SYSTEM_PROMPT = """You are the ONCO-2024-INT Clinician Agent.

Execute this 5-step pipeline IN ORDER:
STEP 1  get_patient_data(patient_id)
STEP 2  evaluate_symptom_burden(patient_data_json)
STEP 3  evaluate_tumor_markers(patient_data_json)
STEP 4  evaluate_renal_function(patient_data_json)
STEP 5  run_alert_engine(sb_result_json, tm_result_json, rf_result_json)

Never skip steps. Never calculate percentages yourself.

After all 5 steps, output exactly this (blank lines between every section):

🩺 Clinical Assessment — Patient [ID]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MAIN PANEL

  ● Symptom Burden: [one sentence result]

  ● Tumor Markers: [one sentence result]

  ● Renal Function: [one sentence result]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 CLINICAL ALERT

  [⚠ alert text if active — or — ✅ No active clinical alerts.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 CLINICAL SUMMARY

  [2-3 sentences synthesising all findings.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
---

For follow-up questions about a specific domain (e.g. "explain the renal trend"),
answer conversationally in plain English without re-running all steps.
"""
# ══════════════════════════════════════════════════════════════════════
# ADK AGENT + RUNNER
# ══════════════════════════════════════════════════════════════════════
 
def build_agent() -> Agent:
    return Agent(
        name="clinician_agent",
        model="gemini-2.0-flash",   # swap to "gemini-2.0-flash" for Gemini
        description="ONCO-2024-INT clinician agent for clinical benefit assessment.",
        instruction=SYSTEM_PROMPT,
        tools=[
            FunctionTool(get_patient_data),
            FunctionTool(evaluate_symptom_burden),
            FunctionTool(evaluate_tumor_markers),
            FunctionTool(evaluate_renal_function),
            FunctionTool(run_alert_engine),
        ],
    )
 
 
def build_runner() -> Runner:
    return Runner(
        agent=build_agent(),
        session_service=InMemorySessionService(),
        app_name="clinician_agent",
    )
 
 
# Required by ADK dev UI (adk web)
root_agent = build_agent()
