"""
S2: Improvement calculation tool.

Implements the 5 improvement rules:
  1. Anchor Baseline (earliest / IS_BASELINE=Yes) and Current (latest) per LOINC
  2. Delta = (Current - Baseline) / Baseline * 100
  3. Compare delta vs policy threshold (e.g. >= 30% reduction)
  4. Cross-check: RECIST-SLD imaging must corroborate lab delta
  5. Unit validation: reject comparison if UNIT mismatch
"""

from datetime import date
from typing import Optional

from payer_agent.tools.data_loader import get_dataframes

# Tumor markers and RECIST SLD threshold: >= 30% reduction = improvement
_TUMOR_MARKER_LOINC = {"85319-2", "2857-1"}
_SLD_CODE = "RECIST-SLD"
_NEW_LESION_CODE = "RECIST-NEW"
_PD_SLD_THRESHOLD = 20.0   # >= +20% SLD increase = progressive disease
_IMPROVEMENT_THRESHOLD = -30.0  # <= -30% delta = tumor improvement


def calculate_improvement(patient_id: str) -> dict:
    """Calculates clinical improvement metrics for a patient.

    For each LOINC code, anchors the Baseline observation (IS_BASELINE=Yes or
    earliest date) and the Current observation (most recent date), then computes
    the percentage change. Cross-checks imaging (RECIST-SLD) against lab markers.
    Detects new lesions (RECIST-NEW).

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.

    Returns:
        dict: Per-marker deltas, imaging findings, new lesion flag,
              progressive disease flag, and overall improvement verdict.
    """
    _, observations, _, _ = get_dataframes()
    patient_obs = observations[observations["PATIENT_ID"] == patient_id]

    if patient_obs.empty:
        return {
            "status": "error",
            "error_message": f"No observations found for {patient_id}",
        }

    loinc_groups = patient_obs.groupby("LOINC_CODE")
    markers: list[dict] = []
    new_lesion_detected = False
    sld_delta_pct: Optional[float] = None
    tumor_marker_delta_pct: Optional[float] = None
    tumor_marker_name: Optional[str] = None

    for loinc_code, group in loinc_groups:
        loinc_code = str(loinc_code)

        if loinc_code == _NEW_LESION_CODE:
            latest = group.sort_values("EFFECTIVE_DATE").iloc[-1]
            if float(latest["VALUE"]) >= 1:
                new_lesion_detected = True
            markers.append(
                {
                    "loinc_code": loinc_code,
                    "display_name": latest["DISPLAY_NAME"],
                    "new_lesion_detected": True,
                    "observation_date": str(latest["EFFECTIVE_DATE"]),
                }
            )
            continue

        # Find baseline (IS_BASELINE=Yes, or earliest)
        baseline_rows = group[group["IS_BASELINE"] == "Yes"]
        if baseline_rows.empty:
            baseline_row = group.sort_values("EFFECTIVE_DATE").iloc[0]
        else:
            baseline_row = baseline_rows.sort_values("EFFECTIVE_DATE").iloc[0]

        # Find current (most recent, excluding baseline if same date)
        non_baseline = group[group["EFFECTIVE_DATE"] != baseline_row["EFFECTIVE_DATE"]]
        if non_baseline.empty:
            markers.append(
                {
                    "loinc_code": loinc_code,
                    "display_name": baseline_row["DISPLAY_NAME"],
                    "baseline_value": float(baseline_row["VALUE"]),
                    "baseline_date": str(baseline_row["EFFECTIVE_DATE"]),
                    "baseline_unit": baseline_row["UNIT"],
                    "current_value": None,
                    "note": "Only baseline observation available; no follow-up.",
                }
            )
            continue

        current_row = non_baseline.sort_values("EFFECTIVE_DATE").iloc[-1]

        # Unit validation
        if baseline_row["UNIT"] != current_row["UNIT"]:
            markers.append(
                {
                    "loinc_code": loinc_code,
                    "display_name": baseline_row["DISPLAY_NAME"],
                    "baseline_value": float(baseline_row["VALUE"]),
                    "baseline_unit": baseline_row["UNIT"],
                    "current_value": float(current_row["VALUE"]),
                    "current_unit": current_row["UNIT"],
                    "unit_mismatch": True,
                    "note": "Unit mismatch — comparison rejected.",
                }
            )
            continue

        baseline_val = float(baseline_row["VALUE"])
        current_val = float(current_row["VALUE"])

        if baseline_val == 0:
            if current_val == 0:
                delta_pct = 0.0
            else:
                delta_pct = float("inf") if current_val > 0 else float("-inf")
        else:
            delta_pct = round((current_val - baseline_val) / baseline_val * 100, 2)

        marker_entry = {
            "loinc_code": loinc_code,
            "display_name": baseline_row["DISPLAY_NAME"],
            "observation_type": baseline_row["OBSERVATION_TYPE"],
            "baseline_value": baseline_val,
            "baseline_date": str(baseline_row["EFFECTIVE_DATE"]),
            "current_value": current_val,
            "current_date": str(current_row["EFFECTIVE_DATE"]),
            "unit": baseline_row["UNIT"],
            "delta_pct": delta_pct,
        }

        if loinc_code == _SLD_CODE:
            sld_delta_pct = delta_pct
            marker_entry["is_sld"] = True
            if delta_pct >= _PD_SLD_THRESHOLD:
                marker_entry["progressive_disease_signal"] = True
            elif delta_pct <= _IMPROVEMENT_THRESHOLD:
                marker_entry["improvement_signal"] = True
            elif abs(delta_pct) < 5:
                marker_entry["stable_disease_signal"] = True

        if loinc_code in _TUMOR_MARKER_LOINC:
            tumor_marker_delta_pct = delta_pct
            tumor_marker_name = baseline_row["DISPLAY_NAME"]
            if delta_pct <= _IMPROVEMENT_THRESHOLD:
                marker_entry["improvement_signal"] = True

        markers.append(marker_entry)

    # Overall assessment
    has_progressive_disease = False
    if sld_delta_pct is not None and sld_delta_pct >= _PD_SLD_THRESHOLD:
        has_progressive_disease = True
    if new_lesion_detected:
        has_progressive_disease = True

    # Determine improvement (requires tumor marker OR SLD to show >= 30% reduction)
    improvement_met = False
    if tumor_marker_delta_pct is not None and tumor_marker_delta_pct <= _IMPROVEMENT_THRESHOLD:
        improvement_met = True
    if sld_delta_pct is not None and sld_delta_pct <= _IMPROVEMENT_THRESHOLD:
        improvement_met = True

    # Stable disease: SLD change within ±5% and no new lesions
    stable_disease = False
    stable_disease_weeks: Optional[float] = None
    if sld_delta_pct is not None and abs(sld_delta_pct) < 5 and not new_lesion_detected:
        stable_disease = True
        sld_entries = [m for m in markers if m.get("is_sld")]
        if sld_entries:
            baseline_d = date.fromisoformat(sld_entries[0]["baseline_date"])
            current_d = date.fromisoformat(sld_entries[0]["current_date"])
            stable_disease_weeks = round((current_d - baseline_d).days / 7, 1)

    # Cross-check: imaging must corroborate lab marker direction
    cross_check_pass = True
    cross_check_note = ""
    if tumor_marker_delta_pct is not None and sld_delta_pct is not None:
        marker_improving = tumor_marker_delta_pct < 0
        sld_improving = sld_delta_pct <= 0
        if marker_improving and not sld_improving:
            cross_check_pass = False
            cross_check_note = (
                f"Lab marker ({tumor_marker_name}) shows improvement "
                f"(delta {tumor_marker_delta_pct}%) but imaging SLD is worsening "
                f"(delta {sld_delta_pct}%). Discordant findings."
            )
        elif not marker_improving and sld_improving:
            cross_check_pass = False
            cross_check_note = (
                f"Imaging SLD shows improvement (delta {sld_delta_pct}%) "
                f"but lab marker ({tumor_marker_name}) is worsening "
                f"(delta {tumor_marker_delta_pct}%). Discordant findings."
            )

    # If new lesion detected alongside improvement, flag as mixed response
    mixed_response = improvement_met and new_lesion_detected

    verdict = "No improvement"
    if has_progressive_disease and not improvement_met:
        verdict = "Progressive disease"
    elif mixed_response:
        verdict = "Mixed response (improvement with new lesion)"
    elif stable_disease:
        sd_text = ""
        if stable_disease_weeks:
            sd_text = f" for {stable_disease_weeks} weeks"
        verdict = f"Stable disease documented{sd_text}"
    elif improvement_met:
        verdict = "Objective improvement documented"

    return {
        "status": "success",
        "patient_id": patient_id,
        "markers": markers,
        "improvement_met": improvement_met,
        "stable_disease": stable_disease,
        "stable_disease_weeks": stable_disease_weeks,
        "new_lesion_detected": new_lesion_detected,
        "has_progressive_disease": has_progressive_disease,
        "mixed_response": mixed_response,
        "cross_check_pass": cross_check_pass,
        "cross_check_note": cross_check_note,
        "verdict": verdict,
    }
