# Multi-Agent ADK Application

Three AI-powered agents built with [Google ADK](https://google.github.io/adk-docs/) for healthcare workflows: Payer Coverage, Pharma Trial Investigator, and Clinician.

## Overview

| Agent | Role |
|-------|------|
| **payer_agent** | Payer Medical Director — evaluates continued treatment coverage using claims, clinical outcomes, policies, and RWE benchmarks |
| **trial_investigator_agent** | Pharma Trial Investigator — assesses trial subjects using SDTM data and protocol rules (RECIST, AE, LAB, VISIT, SAFETY) |
| **clinician_agent** | Clinician — clinical benefit assessment with symptom burden, tumor markers, renal function, and alert engine |

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Add a single `.env` file at **project root** with your Google API key (used by all agents):

```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_api_key_here
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Running the Agents

### Web UI (recommended)
```bash
adk web agents
```
Then open http://localhost:8000 and select an agent from the dropdown: `payer_agent`, `trial_investigator_agent`, or `clinician_agent`.

### Terminal
```bash
adk run agents/payer_agent
adk run agents/trial_investigator_agent
adk run agents/clinician_agent
```

## Data Files

Each agent uses its own data; place files as follows:

| Agent | Required files | Location |
|-------|----------------|----------|
| **payer_agent** | `payer_expanded_dataset.xlsx` | `agents/payer_agent/` |
| **trial_investigator_agent** | `agent_input_dataset.xlsx` (SDTM sheets: SDTM_RS, SDTM_AE, SDTM_LB, SDTM_EX, PROTOCOL_RULES) | `agents/trial_investigator_agent/data/` |
| **clinician_agent** | `clinician_protocol_table_formula_guide.xlsx`, `intent_aware_interop_dataset (1).xlsx` | `agents/clinician_agent/` |

## Example Prompts

**Payer:** "Evaluate coverage for patient PT-001", "Assess PT-004 for continued treatment coverage"

**Trial Investigator:** "Assess subject PT-004", "Evaluate subject PT-001"

**Clinician:** Patient data and symptom burden evaluation prompts
