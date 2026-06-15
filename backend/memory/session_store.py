"""Bounded in-process conversation memory keyed by session_id.

Mirrors LangChain's ConversationBufferWindowMemory semantics: ``k`` is the
number of user+assistant *turn pairs* to retain (not individual messages).
"""

from __future__ import annotations

import threading
from typing import Any


class SessionMemoryStore:
    def __init__(self, window_turns: int = 6) -> None:
        self.window_turns = max(1, window_turns)
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> list[dict[str, str]]:
        user_message = user_message.strip()
        assistant_message = assistant_message.strip()
        if not user_message or not assistant_message:
            return self.get_messages(session_id)

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            max_messages = self.window_turns * 2
            if len(history) > max_messages:
                history[:] = history[-max_messages:]
            return list(history)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


_store: SessionMemoryStore | None = None


def get_session_store(window_turns: int | None = None) -> SessionMemoryStore:
    global _store
    if _store is None:
        from backend.config import settings

        k = window_turns if window_turns is not None else settings.MEMORY_WINDOW_TURNS
        _store = SessionMemoryStore(window_turns=k)
    return _store
