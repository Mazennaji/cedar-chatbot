from django.db import models


class DailyMetrics(models.Model):

    date = models.DateField(unique=True)
    total_sessions = models.IntegerField(default=0)
    total_messages = models.IntegerField(default=0)
    unique_users = models.IntegerField(default=0)
    avg_session_duration = models.FloatField(default=0.0, help_text="Minutes")
    avg_turns_per_session = models.FloatField(default=0.0)

    english_messages = models.IntegerField(default=0)
    arabic_messages = models.IntegerField(default=0)
    arabizi_messages = models.IntegerField(default=0)

    positive_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)

    feedback_count = models.IntegerField(default=0)
    avg_rating = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-date"]
        verbose_name = "Daily Metrics"
        verbose_name_plural = "Daily Metrics"

    def __str__(self):
        return f"Metrics for {self.date}: {self.total_messages} msgs"


class ModelPerformance(models.Model):

    timestamp = models.DateTimeField(auto_now_add=True)
    model_name = models.CharField(max_length=200)
    avg_response_time_ms = models.FloatField()
    avg_reward_score = models.FloatField(null=True, blank=True)
    total_requests = models.IntegerField()
    error_count = models.IntegerField(default=0)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.model_name} @ {self.timestamp}: {self.avg_response_time_ms}ms"