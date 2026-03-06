import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

# Default path when DATASET_PATH env is unset (add agent_input_dataset.xlsx here later)
_DEFAULT_DATASET = Path(__file__).resolve().parent / "agent_input_dataset.xlsx"


@dataclass(frozen=True)
class TrialData:
    rs: pd.DataFrame
    ae: pd.DataFrame
    lb: pd.DataFrame
    ex: pd.DataFrame
    protocol_rules: pd.DataFrame


def _read_sheet(xlsx_path: str, sheet_name: str, header_row: int = 2) -> pd.DataFrame:
    """
    Your workbook sheets have a title row, then the real headers on the next row.
    header_row=1 means "use the 2nd row as header" (0-indexed).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    df = df.dropna(how="all").reset_index(drop=True)
    # Normalize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_trial_data(dataset_path: str | None = None) -> TrialData:
    xlsx_path = dataset_path or os.getenv("DATASET_PATH") or str(_DEFAULT_DATASET)
    if not xlsx_path:
        raise ValueError("DATASET_PATH not set and dataset_path not provided.")

    rs = _read_sheet(xlsx_path, "SDTM_RS")
    ae = _read_sheet(xlsx_path, "SDTM_AE")
    lb = _read_sheet(xlsx_path, "SDTM_LB")
    ex = _read_sheet(xlsx_path, "SDTM_EX")
    protocol_rules = _read_sheet(xlsx_path, "PROTOCOL_RULES")

    return TrialData(rs=rs, ae=ae, lb=lb, ex=ex, protocol_rules=protocol_rules)


def list_subjects(data: TrialData) -> list[str]:
    # Prefer RS as primary; fall back to any domain that has USUBJID.
    for df in (data.rs, data.ae, data.lb, data.ex):
        if "USUBJID" in df.columns:
            vals = df["USUBJID"].dropna().astype(str).unique().tolist()
            vals.sort()
            return vals
    return []