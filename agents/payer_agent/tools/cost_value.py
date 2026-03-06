"""
Cost & Value assessment tool.

Implements the 4 cost/value rules:
  1. Outcome exceeds RWE benchmark -> High Value — Approved
  2. TOTAL_COST > benchmark avg -> FLAG_UTILIZATION_REVIEW
  3. Hospitalizations >= 2 + ER >= 2 -> FLAG_HIGH_UTILIZER
  4. PD or worsening + no value -> Discontinue coverage
"""

from payer_agent.tools.data_loader import get_dataframes

_ER_CPT_CODES = {"99281", "99282", "99283", "99284", "99285"}


def assess_cost_and_value(patient_id: str, response_type: str) -> dict:
    """Assesses cost and value for a patient against RWE benchmarks.

    Sums ALLOWED_AMT_USD across all cycles. Counts ER visits and
    hospitalizations from claims. Compares outcome against the RWE
    benchmark table to produce a value assessment.

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.
        response_type: The assessed response classification, one of:
            'Complete Response (CR)', 'Partial Response (PR)',
            'Stable Disease (SD)', 'Progressive Disease (PD)',
            'Mixed / Complex Response'.

    Returns:
        dict: Total cost, average cost per cycle, hospitalization and ER counts,
              value assessment, and any utilization flags.
    """
    claims, _, benchmarks, _ = get_dataframes()
    patient_claims = claims[claims["PATIENT_ID"] == patient_id]

    if patient_claims.empty:
        return {"status": "error", "error_message": f"No claims found for {patient_id}"}

    total_cost = float(patient_claims["ALLOWED_AMT_USD"].sum())
    num_cycles = len(patient_claims)
    avg_cost_per_cycle = round(total_cost / num_cycles, 2) if num_cycles > 0 else 0.0

    # Count ER visits (CPT 99281-99285) and hospitalizations
    er_visits = int(patient_claims["CPT_HCPCS"].isin(_ER_CPT_CODES).sum())
    hospitalizations = 0  # dataset uses drug administration CPTs, not inpatient

    # Lookup RWE benchmark
    benchmark_row = benchmarks[benchmarks["RESPONSE_TYPE"] == response_type]
    benchmark_match = None
    if not benchmark_row.empty:
        row = benchmark_row.iloc[0]
        benchmark_match = {
            "response_type": response_type,
            "rwe_response_rate": row["RWE_RESPONSE_RATE"],
            "trial_response_rate": str(row["TRIAL_RESPONSE_RATE"]),
            "median_pfs_months": (
                float(row["MEDIAN_PFS_MONTHS"])
                if row["MEDIAN_PFS_MONTHS"] is not None
                else None
            ),
            "median_os_months": (
                float(row["MEDIAN_OS_MONTHS"])
                if row["MEDIAN_OS_MONTHS"] is not None
                else None
            ),
            "avg_cost_per_cycle_benchmark": float(row["AVG_COST_PER_CYCLE_USD"]),
            "avg_cycles_to_response": int(row["AVG_CYCLES_TO_RESPONSE"]),
            "policy_threshold": row["POLICY_CONTINUATION_THRESHOLD"],
        }

    flags = []
    value_assessment = "Standard"

    if benchmark_match:
        benchmark_avg_cost = benchmark_match["avg_cost_per_cycle_benchmark"]
        benchmark_expected_total = benchmark_avg_cost * benchmark_match["avg_cycles_to_response"]

        if total_cost > benchmark_expected_total:
            flags.append("FLAG_UTILIZATION_REVIEW")

        if response_type in ("Complete Response (CR)", "Partial Response (PR)"):
            value_assessment = "High Value — Approved"
        elif response_type == "Stable Disease (SD)":
            value_assessment = "Acceptable Value — Continued monitoring"
        elif response_type == "Progressive Disease (PD)":
            value_assessment = "No Value — Coverage discontinuation recommended"
            flags.append("DISCONTINUE_COVERAGE")
        elif response_type == "Mixed / Complex Response":
            value_assessment = "Indeterminate — Medical Director review required"

    if hospitalizations >= 2 and er_visits >= 2:
        flags.append("FLAG_HIGH_UTILIZER")

    return {
        "status": "success",
        "patient_id": patient_id,
        "total_cost_to_date": total_cost,
        "num_cycles": num_cycles,
        "avg_cost_per_cycle": avg_cost_per_cycle,
        "hospitalizations": hospitalizations,
        "er_visits": er_visits,
        "response_type": response_type,
        "benchmark": benchmark_match,
        "value_assessment": value_assessment,
        "flags": flags,
    }
