import time
import pytest
from src.memory import ConversationMemory


class TestConversationMemory:

    def setup_method(self):
        self.memory = ConversationMemory(max_turns=5, session_timeout=10)

    def test_create_session(self):
        session = self.memory.get_or_create_session("test-1")
        assert session.session_id == "test-1"
        assert session.turn_count == 0

    def test_auto_generate_session_id(self):
        session = self.memory.get_or_create_session()
        assert session.session_id is not None
        assert len(session.session_id) > 0

    def test_add_messages(self):
        self.memory.add_message("s1", "user", "Hello")
        self.memory.add_message("s1", "assistant", "Hi there!")
        history = self.memory.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_context_formatting(self):
        self.memory.add_message("s1", "user", "Hello")
        self.memory.add_message("s1", "assistant", "Hi!")
        self.memory.add_message("s1", "user", "How are you?")
        context = self.memory.get_context("s1")
        assert "Hello" in context
        assert "Hi!" in context
        assert "How are you?" in context

    def test_max_turns_limit(self):
        for i in range(20):
            self.memory.add_message("s1", "user", f"Message {i}")
            self.memory.add_message("s1", "assistant", f"Reply {i}")
        history = self.memory.get_history("s1")
        assert len(history) <= 10 

    def test_feedback(self):
        self.memory.add_feedback("s1", "msg-1", 1, "Great!")
        feedback = self.memory.get_feedback("s1")
        assert len(feedback) == 1
        assert feedback[0]["rating"] == 1

    def test_all_feedback(self):
        self.memory.add_feedback("s1", "msg-1", 1)
        self.memory.add_feedback("s2", "msg-2", -1)
        all_fb = self.memory.get_feedback()
        assert len(all_fb) == 2

    def test_clear_session(self):
        self.memory.add_message("s1", "user", "Hello")
        self.memory.clear_session("s1")
        history = self.memory.get_history("s1")
        assert len(history) == 0

    def test_stats(self):
        self.memory.add_message("s1", "user", "Hello")
        self.memory.add_message("s1", "assistant", "Hi!")
        self.memory.add_feedback("s1", "msg-1", 1)
        stats = self.memory.get_stats()
        assert stats["active_sessions"] == 1
        assert stats["total_messages"] == 2
        assert stats["total_feedback"] == 1

    def test_session_persistence(self):
        self.memory.add_message("s1", "user", "Hello")
        session = self.memory.get_or_create_session("s1")
        assert session.turn_count == 1

    def test_message_metadata(self):
        msg = self.memory.add_message(
            "s1", "user", "Hello",
            metadata={"language": "english"}
        )
        assert msg.metadata["language"] == "english"
        assert msg.message_id is not None