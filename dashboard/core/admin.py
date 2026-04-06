from django.contrib import admin
from core.models import ChatSession, ChatMessage, UserFeedback, BotConfiguration


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0


class UserFeedbackInline(admin.TabularInline):
    model = UserFeedback
    extra = 0


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["session_id", "user", "message_count", "created_at", "is_active"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["session_id"]
    inlines = [ChatMessageInline, UserFeedbackInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["message_id", "session", "role", "content_preview", "detected_language", "intent", "sentiment_label", "timestamp"]
    list_filter = ["role", "detected_language", "intent", "sentiment_label"]
    search_fields = ["content", "message_id"]

    @admin.display(description="Content")
    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ["session", "message", "rating", "comment", "created_at"]
    list_filter = ["rating", "created_at"]


@admin.register(BotConfiguration)
class BotConfigurationAdmin(admin.ModelAdmin):
    list_display = ["key", "value", "updated_at"]
    search_fields = ["key"]


admin.site.site_header = "🌲 Cedar Chatbot Admin"
admin.site.site_title = "Cedar Admin"
admin.site.index_title = "Dashboard"