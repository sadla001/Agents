import os
from pathlib import Path
 
# ── API Keys ─────────────────────────────────────────────────────────
#OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
 
# ── LLM settings ─────────────────────────────────────────────────────
LLM_MODEL       = "gemini-2.0-flash"  # Gemini Pro for best performance; Gemini 2.0 for lower cost
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 2048
AGENT_MAX_ITER  = 10
 
# ── GCS ──────────────────────────────────────────────────────────────
# Set GCS_BUCKET as an environment variable in Cloud Run.
# Leave it empty when running locally.
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
 
# ── File paths ───────────────────────────────────────────────────────
# Cloud Run: files go to /tmp (writable, ephemeral)
# Local dev: files sit next to this script
if GCS_BUCKET:
    BASE_DIR = Path("/tmp")
else:
    BASE_DIR = Path(__file__).parent
 
PROTOCOL_TABLE_PATH = BASE_DIR / "clinician_protocol_table_formula_guide.xlsx"
DATASET_PATH        = BASE_DIR / "intent_aware_interop_dataset (1).xlsx"
 
# ── GCS download helper ───────────────────────────────────────────────
def download_xlsx_files():
    """Download both xlsx files from GCS at container startup."""
    if not GCS_BUCKET:
        return  # Running locally, files already present
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    for filename in [
        "clinician_protocol_table_formula_guide.xlsx",
        "intent_aware_interop_dataset (1).xlsx",
    ]:
        dest = Path("/tmp") / filename
        if not dest.exists():
            bucket.blob(filename).download_to_filename(str(dest))
            print(f"Downloaded {filename} from gs://{GCS_BUCKET}")
 
# ── LOINC codes ──────────────────────────────────────────────────────
LOINC = {
    "eGFR":       "62238-1",
    "CA-125":     "85319-2",
    "Hemoglobin": "718-7",
    "Creatinine": "2160-0",
    "ALT":        "1742-6",
}
 
# ── Sheet names ──────────────────────────────────────────────────────
RULES_SHEET   = "Clinician_Protocol_Rules"
DATASET_SHEET = "Clinician – Patient Care"
