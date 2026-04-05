from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from core.models import ChatSession, ChatMessage, UserFeedback


@login_required
def dashboard(request):
    total_sessions = ChatSession.objects.count()
    active_sessions = ChatSession.objects.filter(is_active=True).count()
    total_messages = ChatMessage.objects.count()
    total_feedback = UserFeedback.objects.count()

    positive_feedback = UserFeedback.objects.filter(rating=1).count()
    negative_feedback = UserFeedback.objects.filter(rating=-1).count()

    recent_sessions = ChatSession.objects.order_by("-updated_at")[:10]

    context = {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_messages": total_messages,
        "total_feedback": total_feedback,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "satisfaction_rate": (
            round(positive_feedback / total_feedback * 100, 1)
            if total_feedback > 0 else 0
        ),
        "recent_sessions": recent_sessions,
    }
    return render(request, "core/dashboard.html", context)


@login_required
def session_detail(request, session_id):
    session = ChatSession.objects.get(session_id=session_id)
    messages = session.messages.all()
    feedback = session.feedback.all()

    context = {
        "session": session,
        "messages": messages,
        "feedback": feedback,
    }
    return render(request, "core/session_detail.html", context)