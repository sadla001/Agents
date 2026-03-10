import json, re, uuid
from typing import Optional
 
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.adk.runners import Runner
from google.genai import types as genai_types
 
from .agent import build_runner
 
app    = FastAPI(title="ONCO-2024-INT Clinician Agent")
runner: Optional[Runner] = None
 
# Serve the static chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")
 
 
@app.on_event("startup")
async def startup():
    global runner
    runner = build_runner()
    print("ADK Runner ready")
 
# ── Pydantic model ───────────────────────────────────────────────────
class IntentRequest(BaseModel):
    role:       str
    goal:       str
    patient_id: str
 
 
# ── POST /intent  (structured JSON response) ─────────────────────────
@app.post("/intent")
async def assess_clinical_benefit(request: IntentRequest):
    if request.role != "clinician":
        raise HTTPException(400, f"Role {request.role!r} not supported.")
    if request.goal != "assess_clinical_benefit":
        raise HTTPException(400, f"Goal {request.goal!r} not recognised.")
 
    session_id = str(uuid.uuid4())
    msg = genai_types.Content(role="user", parts=[genai_types.Part(text=(
        f"Patient ID: {request.patient_id}\n"
        f"Role: {request.role} | Goal: {request.goal}\n"
        "Run all 5 steps and return the Clinical Response Card as JSON."
    ))])
 
    final = ""
    async for event in runner.run_async(
        user_id=request.patient_id,
        session_id=session_id,
        new_message=msg,
    ):
        if event.is_final_response() and event.content:
            final = "".join(
                p.text for p in event.content.parts if hasattr(p, "text"))
 
    card = _parse_json(final)
    return {**card, "role": request.role,
            "goal": request.goal, "status": "success"}
 
 
# ── WebSocket /chat/{patient_id}  (chat interface) ───────────────────
# Each patient_id gets its own session so the agent remembers context.
active_sessions: dict[str, str] = {}
 
 
@app.websocket("/chat/{patient_id}")
async def chat_ws(websocket: WebSocket, patient_id: str):
    await websocket.accept()
 
    if patient_id not in active_sessions:
        active_sessions[patient_id] = str(uuid.uuid4())
    session_id = active_sessions[patient_id]
 
    await websocket.send_json({
        "type": "system",
        "text": f"Connected to ONCO-2024-INT Agent for patient {patient_id}."
    })
 
    try:
        while True:
            user_text = await websocket.receive_text()
            msg = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_text)]
            )
            async for event in runner.run_async(
                user_id=patient_id,
                session_id=session_id,
                new_message=msg,
            ):
                if not event.is_final_response() and event.content:
                    for p in event.content.parts:
                        if hasattr(p, "text") and p.text:
                            await websocket.send_json({
                                "type": "thinking", "text": p.text})
                if event.is_final_response() and event.content:
                    final = "".join(
                        p.text for p in event.content.parts
                        if hasattr(p, "text"))
                    await websocket.send_json({
                        "type": "response", "text": final})
 
    except WebSocketDisconnect:
        print(f"Disconnected: {patient_id}")
 
 
@app.delete("/chat/{patient_id}")
async def clear_session(patient_id: str):
    active_sessions.pop(patient_id, None)
    return {"cleared": patient_id}
 
 
# ── GET /  (serves the chat UI) ──────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("static/chat.html")
 
 
# ── GET /health ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    from agent import RULES_ENGINE, DATA_LOADER
    return {"status": "healthy",
            "rules_loaded": RULES_ENGINE.total_rules,
            "patients": DATA_LOADER.all_ids()}
 
 
# ── JSON helper ──────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m: raise HTTPException(500, f"No JSON in agent output: {raw[:300]}")
    return json.loads(m.group())
 
 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)