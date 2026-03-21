# api/session_store.py -- in-memory session store
# Each session holds the current Music Sheet + chat history.

import uuid
from typing import Optional


_sessions: dict[str, dict] = {}


def create_session(sheet: dict) -> str:
    session_id = uuid.uuid4().hex[:8]
    _sessions[session_id] = {"sheet": sheet, "history": []}
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def update_sheet(session_id: str, sheet: dict) -> bool:
    if session_id not in _sessions:
        return False
    _sessions[session_id]["sheet"] = sheet
    return True


def append_history(session_id: str, message: dict) -> None:
    if session_id in _sessions:
        _sessions[session_id]["history"].append(message)
