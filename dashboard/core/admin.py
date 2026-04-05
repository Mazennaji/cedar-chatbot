from django.contrib import admin
from core.models import ChatSession, ChatMessage, UserFeedback, BotConfiguration


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = [
        "message_id", "role", "content", "original_content",
        "timestamp", "detected_language", "intent",
        "sentiment_label", "sentiment_score",
    ]
    can_delete = False


class UserFeedbackInline(admin.TabularInline):
    model = UserFeedback
    extra = 0
    readonly_fields = ["message", "rating", "preferred_response", "comment", "created_at"]
    can_delete = False


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = [
        "session_id_short", "user", "message_count",
        "created_at", "updated_at", "is_active",
    ]
    list_filter = ["is_active", "created_at"]
    search_fields = ["session_id", "user__username"]
    readonly_fields = ["session_id", "created_at", "updated_at"]
    inlines = [ChatMessageInline, UserFeedbackInline]

    @admin.display(description="Session ID")
    def session_id_short(self, obj):
        return obj.session_id[:12] + "..."


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = [
        "message_id_short", "role", "content_preview",
        "detected_language", "intent", "sentiment_label", "timestamp",
    ]
    list_filter = ["role", "detected_language", "intent", "sentiment_label"]
    search_fields = ["content", "message_id"]
    readonly_fields = [
        "message_id", "session", "role", "content",
        "original_content", "timestamp",
    ]

    @admin.display(description="Message ID")
    def message_id_short(self, obj):
        return obj.message_id[:12] + "..."

    @admin.display(description="Content")
    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ["session", "rating", "comment_preview", "created_at"]
    list_filter = ["rating", "created_at"]
    readonly_fields = ["session", "message", "rating", "preferred_response", "comment"]

    @admin.display(description="Comment")
    def comment_preview(self, obj):
        return obj.comment[:60] + "..." if len(obj.comment) > 60 else obj.comment


@admin.register(BotConfiguration)
class BotConfigurationAdmin(admin.ModelAdmin):
    list_display = ["key", "value_preview", "updated_at"]
    search_fields = ["key", "description"]

    @admin.display(description="Value")
    def value_preview(self, obj):
        return obj.value[:60] + "..." if len(obj.value) > 60 else obj.value


admin.site.site_header = "🌲 Cedar Chatbot Admin"
admin.site.site_title = "Cedar Admin"
admin.site.index_title = "Dashboard"