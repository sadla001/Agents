"""
Data loading tools for the Payer Coverage Agent.

Reads the payer_expanded_dataset.xlsx file and provides tools to query
patient claims, clinical observations, RWE benchmarks, and policy references.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

EXCEL_PATH = Path(__file__).resolve().parent.parent / "payer_expanded_dataset.xlsx"

_claims_df: Optional[pd.DataFrame] = None
_observations_df: Optional[pd.DataFrame] = None
_benchmarks_df: Optional[pd.DataFrame] = None
_policy_df: Optional[pd.DataFrame] = None


def _load_all() -> None:
    """Load all 4 sheets from the Excel file into module-level DataFrames."""
    global _claims_df, _observations_df, _benchmarks_df, _policy_df

    if _claims_df is not None:
        return

    _claims_df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="A – Claims History (Expanded)",
        header=3,  # row 4 is the header (0-indexed row 3)
    )
    _claims_df = _claims_df.dropna(how="all")
    _claims_df = _claims_df[_claims_df["CLAIM_ID"].notna()]
    _claims_df = _claims_df[~_claims_df["CLAIM_ID"].astype(str).str.startswith("AGENT NOTE")]
    _claims_df["SERVICE_DATE"] = pd.to_datetime(_claims_df["SERVICE_DATE"]).dt.date
    _claims_df["DAYS_SUPPLY"] = _claims_df["DAYS_SUPPLY"].astype(int)

    _observations_df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="C – FHIR Observations (Raw)",
        header=3,
    )
    _observations_df = _observations_df.dropna(how="all")
    _observations_df = _observations_df[_observations_df["OBSERVATION_ID"].notna()]
    _observations_df = _observations_df[
        ~_observations_df["OBSERVATION_ID"].astype(str).str.startswith("AGENT NOTE")
    ]
    _observations_df["EFFECTIVE_DATE"] = pd.to_datetime(
        _observations_df["EFFECTIVE_DATE"]
    ).dt.date

    _benchmarks_df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="D – RWE Benchmarks (Lookup)",
        header=3,
    )
    _benchmarks_df = _benchmarks_df.dropna(how="all")
    _benchmarks_df = _benchmarks_df[_benchmarks_df["RESPONSE_TYPE"].notna()]

    _policy_df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="E – Policy Reference (RAG)",
        header=3,
    )
    _policy_df = _policy_df.dropna(how="all")
    _policy_df = _policy_df[_policy_df["POLICY_ID"].notna()]
    _policy_df = _policy_df[~_policy_df["POLICY_ID"].astype(str).str.startswith("AGENT NOTE")]


def get_dataframes() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Return (claims, observations, benchmarks, policy) DataFrames."""
    _load_all()
    return _claims_df, _observations_df, _benchmarks_df, _policy_df


# ---------------------------------------------------------------------------
# ADK tool functions
# ---------------------------------------------------------------------------

def list_patients() -> dict:
    """Lists all patients in the dataset with their diagnosis, drug, and cycle count.

    Returns:
        dict: status and a list of patient summaries.
    """
    claims, _, _, _ = get_dataframes()
    patients = []
    for patient_id, group in claims.groupby("PATIENT_ID"):
        first = group.iloc[0]
        patients.append(
            {
                "patient_id": patient_id,
                "member_id": first["MEMBER_ID"],
                "icd10_dx": first["ICD10_DX"],
                "drug_name": first["DRUG_NAME"],
                "num_cycles": len(group),
                "pa_on_file": first["PA_ON_FILE"],
            }
        )
    return {"status": "success", "patients": patients}


def get_patient_claims(patient_id: str) -> dict:
    """Retrieves all claim rows for a given patient from the X12 837 Claims History.

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.

    Returns:
        dict: status and list of claim records with service dates, days supply,
              billed/allowed amounts, PA status, and cycle numbers.
    """
    claims, _, _, _ = get_dataframes()
    patient_claims = claims[claims["PATIENT_ID"] == patient_id]
    if patient_claims.empty:
        return {"status": "error", "error_message": f"No claims found for {patient_id}"}

    records = []
    for _, row in patient_claims.iterrows():
        records.append(
            {
                "claim_id": row["CLAIM_ID"],
                "member_id": row["MEMBER_ID"],
                "patient_id": row["PATIENT_ID"],
                "service_date": str(row["SERVICE_DATE"]),
                "days_supply": int(row["DAYS_SUPPLY"]),
                "provider_npi": str(int(row["PROVIDER_NPI"])),
                "icd10_dx": row["ICD10_DX"],
                "cpt_hcpcs": row["CPT_HCPCS"],
                "ndc": row["NDC"],
                "drug_name": row["DRUG_NAME"],
                "billed_amt_usd": float(row["BILLED_AMT_USD"]),
                "allowed_amt_usd": float(row["ALLOWED_AMT_USD"]),
                "pa_on_file": row["PA_ON_FILE"],
                "cycle": row["CYCLE"],
            }
        )
    return {"status": "success", "claims": records}


def get_patient_observations(patient_id: str) -> dict:
    """Retrieves all FHIR clinical observations for a given patient.

    Args:
        patient_id: The patient identifier, e.g. 'PT-001'.

    Returns:
        dict: status and list of observation records including LOINC code,
              display name, value, unit, effective date, baseline flag,
              and observation type (Lab or Imaging).
    """
    _, observations, _, _ = get_dataframes()
    patient_obs = observations[observations["PATIENT_ID"] == patient_id]
    if patient_obs.empty:
        return {
            "status": "error",
            "error_message": f"No observations found for {patient_id}",
        }

    records = []
    for _, row in patient_obs.iterrows():
        records.append(
            {
                "observation_id": row["OBSERVATION_ID"],
                "member_id": row["MEMBER_ID"],
                "patient_id": row["PATIENT_ID"],
                "loinc_code": str(row["LOINC_CODE"]),
                "display_name": row["DISPLAY_NAME"],
                "value": float(row["VALUE"]),
                "unit": row["UNIT"],
                "ref_low": None if pd.isna(row["REF_LOW"]) else float(row["REF_LOW"]),
                "ref_high": None if pd.isna(row["REF_HIGH"]) else float(row["REF_HIGH"]),
                "effective_date": str(row["EFFECTIVE_DATE"]),
                "is_baseline": row["IS_BASELINE"],
                "observation_type": row["OBSERVATION_TYPE"],
            }
        )
    return {"status": "success", "observations": records}
