# api/main.py -- Clankers3 FastAPI backend
#
# Thin JSON API: LLM calls + Music Sheet state only. No audio synthesis.
# CLI path (session.py / conductor.run_track) is untouched.
#
# Routes:
#   POST /session/new           -> { session_id, sheet }
#   POST /chat                  -> { sheet, diff, reply, companion }
#   POST /sheet/evolve          -> { sheet }
#   GET  /sheet/{session_id}    -> { sheet }
#   PATCH /sheet/{session_id}   -> { sheet }
#
# Run:
#   uvicorn api.main:app --reload --port 8000

import sys
import json
import copy
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import llm_clients
from chatroom.chatroom import Chatroom
from conductor.conductor import evolve, _SECTION_TENSION
from api.session_store import (
    create_session, get_session, update_sheet, append_history,
)

app = FastAPI(title="Clankers3 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    brief:   str
    section: str  = "verse1"
    solo:    bool = True   # True = fast (Claude only); False = full 3-LLM debate


class NewSessionResponse(BaseModel):
    session_id: str
    sheet:      dict


class ChatRequest(BaseModel):
    session_id: str
    message:    str


class ChatResponse(BaseModel):
    sheet:     dict
    diff:      dict
    reply:     str
    companion: str


class EvolveRequest(BaseModel):
    session_id: str
    section:    str


class EvolveResponse(BaseModel):
    sheet: dict


class PatchSheetRequest(BaseModel):
    patch: dict   # partial sheet fields to merge in


class SheetResponse(BaseModel):
    sheet: dict


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/session/new", response_model=NewSessionResponse)
def session_new(req: NewSessionRequest):
    """
    Start a new session. Chatroom negotiates the opening Music Sheet.
    solo=True skips the 3-LLM debate (Claude generates in one pass — much faster).
    """
    room  = Chatroom(session_name=req.section)
    sheet = room.negotiate_section(
        brief=req.brief,
        section_name=req.section,
        solo=req.solo,
    )
    if "tension" not in sheet:
        sheet["tension"] = _SECTION_TENSION.get(req.section, 0.35)

    session_id = create_session(sheet)
    return {"session_id": session_id, "sheet": sheet}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a user message to the band. The Conductor routes it to the right
    companion, updates the Music Sheet, and returns the companion's reply.
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    old_sheet = session["sheet"]
    updated_sheet, reply, companion = _chat_evolve(
        old_sheet, req.message, session["history"]
    )

    diff = _sheet_diff(old_sheet, updated_sheet)
    update_sheet(req.session_id, updated_sheet)
    append_history(req.session_id, {"role": "user",      "content": req.message})
    append_history(req.session_id, {"role": companion,   "content": reply})

    return {"sheet": updated_sheet, "diff": diff, "reply": reply, "companion": companion}


@app.post("/sheet/evolve", response_model=EvolveResponse)
def sheet_evolve(req: EvolveRequest):
    """Evolve the current sheet to the next section (conductor LLM call)."""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    evolved = evolve(session["sheet"], req.section)
    update_sheet(req.session_id, evolved)
    return {"sheet": evolved}


@app.get("/sheet/{session_id}", response_model=SheetResponse)
def sheet_get(session_id: str):
    """Return the current Music Sheet for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"sheet": session["sheet"]}


@app.patch("/sheet/{session_id}", response_model=SheetResponse)
def sheet_patch(session_id: str, req: PatchSheetRequest):
    """Merge partial fields into the current sheet (knob / UI updates)."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    merged = copy.deepcopy(session["sheet"])
    merged.update(req.patch)
    update_sheet(session_id, merged)
    return {"sheet": merged}


# ── Chat handler ───────────────────────────────────────────────────────────

_CONDUCTOR_SYSTEM = """You are the Conductor of The Clankers 3 -- an AI electronic music band.

The band has four companion personas:
  The Bassist  -- warm, dry, musical. Talks about feel and groove.
  The Drummer  -- terse, rhythmic. Talks about energy and patterns.
  Keys         -- harmonic, opinionated. Talks about textures and progressions.
  Conductor    -- orchestrates; listens to user intent. Formal but warm.

The band's sequencer uses ClankerBoy JSON format:
  t:1  Buchla 259/292   Percussive plucks/arps (MIDI 48-72)
  t:2  Pro-One Bass     Sub bass (MIDI 0-23 primarily)
  t:6  HybridSynth Pads Chordal sustain — always include dur field
  t:10 Drums MS-20      Kick:36 Snare:38 HH_cl:42 HH_op:46

You receive the current ClankerBoy JSON sheet and a user message.
1. Pick the companion best suited to respond.
2. Update the sheet's steps array to reflect the user's request.
   You may change BPM, add/remove/modify steps, adjust CC values, swap notes.
3. Write a short in-character reply from that companion (1-3 sentences max).

RULES when editing steps:
  - Drums (t:10) always d:0.25, never use dur.
  - Pads (t:6) always use dur. One trigger, long hold.
  - Bass first note per phrase needs full CC patch: {"71":42,"73":8,"75":50,"79":80,"72":22,"18":10}
  - Bass MIDI 0-23 primarily.

Return ONLY valid JSON -- no prose, no markdown fences:
{
  "companion": "The Bassist",
  "reply": "Short in-character reply.",
  "sheet": { ...complete updated ClankerBoy JSON sheet... }
}"""


def _chat_evolve(sheet: dict, message: str, history: list) -> tuple[dict, str, str]:
    """Use Claude to parse a user message, update the sheet, return (sheet, reply, companion)."""
    client = llm_clients.get_client(config.BAND["Claude"])  # all members use Claude

    # Last 4 history entries (2 turns) for context, formatted as user messages
    ctx = ""
    for h in history[-4:]:
        ctx += f"[{h['role']}]: {h['content']}\n"

    user_content = (
        f"Current sheet:\n{json.dumps(sheet, indent=2)}\n\n"
        + (f"Recent chat:\n{ctx}\n" if ctx else "")
        + f"User message: {message}\n\n"
        "Return the updated sheet + companion reply as JSON."
    )

    response = client.send(_CONDUCTOR_SYSTEM, [{"role": "user", "content": user_content}])

    # Extract the JSON object from the response
    m = re.search(r'\{[\s\S]*\}', response)
    if m:
        try:
            data = json.loads(m.group())
            return (
                data.get("sheet", sheet),
                data.get("reply", ""),
                data.get("companion", "Conductor"),
            )
        except json.JSONDecodeError:
            pass

    # Fallback: sheet unchanged, raw response as reply
    return sheet, response[:500], "Conductor"


def _sheet_diff(old: dict, new: dict) -> dict:
    """Return top-level keys whose values changed."""
    diff = {}
    for k in set(old) | set(new):
        if old.get(k) != new.get(k):
            diff[k] = new.get(k)
    return diff
