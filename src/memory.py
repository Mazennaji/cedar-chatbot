import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str
    content: str 
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    messages: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.messages)

    @property
    def duration_seconds(self) -> float:
        return self.last_active - self.created_at


class ConversationMemory:
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 512,
        session_timeout: int = 3600,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.session_timeout = session_timeout
        self._sessions: dict[str, Session] = {}
        self._feedback: dict[str, list] = defaultdict(list)

    def get_or_create_session(
        self, session_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Session:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if time.time() - session.last_active > self.session_timeout:
                del self._sessions[session_id]
            else:
                session.last_active = time.time()
                return session


        new_id = session_id or str(uuid.uuid4())
        session = Session(session_id=new_id, user_id=user_id)
        self._sessions[new_id] = session
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Message:
        session = self.get_or_create_session(session_id)
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.messages.append(msg)
        session.last_active = time.time()

        if len(session.messages) > self.max_turns * 2:
            session.messages = session.messages[-(self.max_turns * 2):]

        return msg

    def get_context(self, session_id: str, max_turns: Optional[int] = None) -> str:

        turns = max_turns or self.max_turns
        session = self.get_or_create_session(session_id)

        if not session.messages:
            return ""

        recent = session.messages[-(turns * 2):]
        context_parts = []
        total_chars = 0

        for msg in recent:
            line = msg.content
            total_chars += len(line)
            if total_chars > self.max_tokens * 4: 
                break
            context_parts.append(line)

        return " \n ".join(context_parts)

    def get_history(self, session_id: str) -> list[dict]:
        session = self.get_or_create_session(session_id)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
                "metadata": msg.metadata,
            }
            for msg in session.messages
        ]

    def add_feedback(
        self, session_id: str, message_id: str, rating: int, comment: str = ""
    ):
        self._feedback[session_id].append({
            "message_id": message_id,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time(),
        })

    def get_feedback(self, session_id: Optional[str] = None) -> list:
        if session_id:
            return self._feedback.get(session_id, [])
        return [
            {"session_id": sid, **fb}
            for sid, feedbacks in self._feedback.items()
            for fb in feedbacks
        ]

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_active > self.session_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def get_stats(self) -> dict:
        self.cleanup_expired()
        total_messages = sum(
            len(s.messages) for s in self._sessions.values()
        )
        return {
            "active_sessions": len(self._sessions),
            "total_messages": total_messages,
            "total_feedback": sum(len(v) for v in self._feedback.values()),
            "avg_turns_per_session": (
                total_messages / len(self._sessions)
                if self._sessions else 0
            ),
        }