from django.db import models
from django.contrib.auth.models import User


class ChatSession(models.Model):

    session_id = models.CharField(max_length=64, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"

    def __str__(self):
        return f"Session {self.session_id[:8]}... ({self.message_count} msgs)"

    @property
    def message_count(self):
        return self.messages.count()

    @property
    def duration_minutes(self):
        if self.messages.exists():
            first = self.messages.earliest("timestamp")
            last = self.messages.latest("timestamp")
            return (last.timestamp - first.timestamp).total_seconds() / 60
        return 0


class ChatMessage(models.Model):

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    message_id = models.CharField(max_length=64, unique=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    original_content = models.TextField(blank=True, help_text="Original text before normalization")
    timestamp = models.DateTimeField(auto_now_add=True)

    detected_language = models.CharField(max_length=30, blank=True)
    intent = models.CharField(max_length=30, blank=True)
    sentiment_label = models.CharField(max_length=20, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["timestamp"]
        verbose_name = "Chat Message"

    def __str__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role}] {preview}"


class UserFeedback(models.Model):

    RATING_CHOICES = [
        (-1, "Bad"),
        (0, "Neutral"),
        (1, "Good"),
    ]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="feedback"
    )
    message = models.ForeignKey(
        ChatMessage, on_delete=models.CASCADE, related_name="feedback"
    )
    rating = models.IntegerField(choices=RATING_CHOICES)
    preferred_response = models.TextField(blank=True)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "User Feedback"
        verbose_name_plural = "User Feedback"

    def __str__(self):
        return f"Feedback: {self.get_rating_display()} on {self.message_id}"


class BotConfiguration(models.Model):

    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Bot Configuration"

    def __str__(self):
        return f"{self.key} = {self.value[:50]}"