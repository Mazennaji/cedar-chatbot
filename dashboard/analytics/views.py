from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Avg
from analytics.models import DailyMetrics, ModelPerformance
from core.models import ChatMessage


@login_required
def analytics_overview(request):
    daily_metrics = DailyMetrics.objects.order_by("-date")[:30]

    language_dist = (
        ChatMessage.objects
        .filter(role="user", detected_language__isnull=False)
        .exclude(detected_language="")
        .values("detected_language")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    intent_dist = (
        ChatMessage.objects
        .filter(role="user", intent__isnull=False)
        .exclude(intent="")
        .values("intent")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    sentiment_dist = (
        ChatMessage.objects
        .filter(role="user", sentiment_label__isnull=False)
        .exclude(sentiment_label="")
        .values("sentiment_label")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    model_perf = ModelPerformance.objects.order_by("-timestamp")[:10]

    context = {
        "daily_metrics": daily_metrics,
        "language_dist": list(language_dist),
        "intent_dist": list(intent_dist),
        "sentiment_dist": list(sentiment_dist),
        "model_perf": model_perf,
    }
    return render(request, "analytics/overview.html", context)