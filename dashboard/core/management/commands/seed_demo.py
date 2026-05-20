import random
import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import ChatSession, ChatMessage, UserFeedback, BotConfiguration
from analytics.models import DailyMetrics, ModelPerformance


class Command(BaseCommand):
    help = "Populate the dashboard database with realistic demo data for presentations."

    def add_arguments(self, parser):
        parser.add_argument(
            "--wipe",
            action="store_true",
            help="Delete all existing data before seeding (recommended on re-runs).",
        )

    def handle(self, *args, **options):
        random.seed(42)

        if options["wipe"]:
            self.stdout.write("Wiping existing data...")
            UserFeedback.objects.all().delete()
            ChatMessage.objects.all().delete()
            ChatSession.objects.all().delete()
            BotConfiguration.objects.all().delete()
            DailyMetrics.objects.all().delete()
            ModelPerformance.objects.all().delete()

        self.stdout.write("Seeding BotConfiguration...")
        self._seed_bot_configs()

        self.stdout.write("Seeding ChatSessions, ChatMessages, UserFeedback...")
        self._seed_conversations()

        self.stdout.write("Seeding DailyMetrics (7 days)...")
        self._seed_daily_metrics()

        self.stdout.write("Seeding ModelPerformance...")
        self._seed_model_performance()

        self._summary()

    def _seed_bot_configs(self):
        configs = [
            ("active_model",            "facebook/blenderbot-400M-distill", "Currently active conversational model"),
            ("max_response_length",     "128",                              "Maximum tokens for generated responses"),
            ("temperature",             "0.7",                              "Sampling temperature for response generation"),
            ("supported_languages",     "english,arabic_msa,lebanese_arabizi", "Comma-separated language codes"),
            ("rlhf_enabled",            "true",                             "Whether to collect and learn from user feedback"),
            ("session_timeout_minutes", "30",                               "Idle minutes before a session expires"),
        ]
        for key, value, description in configs:
            BotConfiguration.objects.update_or_create(
                key=key,
                defaults={"value": value, "description": description},
            )

    def _seed_conversations(self):
        now = timezone.now()

        conversations = [
            {
                "session_id": f"demo-en-{uuid.uuid4().hex[:8]}",
                "offset_hours": 2,
                "metadata": {"locale": "en-US", "client": "streamlit"},
                "messages": [
                    ("user",      "What is NLP?",                                                                                              "what is nlp", "english", "question", "neutral",  0.10),
                    ("assistant", "NLP stands for Natural Language Processing — a field of AI that helps computers understand human language.", "",            "english", "question", "neutral",  0.00),
                    ("user",      "What is your name?",                                                                                        "",            "english", "question", "neutral",  0.05),
                    ("assistant", "I'm Cedar 🌲 — a trilingual AI assistant. How can I help you?",                                              "",            "english", "question", "neutral",  0.00),
                    ("user",      "Thanks, that's helpful!",                                                                                   "",            "english", "thanks",   "positive", 0.85),
                    ("assistant", "You're welcome! Happy to help with more questions.",                                                        "",            "english", "thanks",   "positive", 0.70),
                ],
            },
            {
                "session_id": f"demo-ar-{uuid.uuid4().hex[:8]}",
                "offset_hours": 4,
                "metadata": {"locale": "ar-SA", "client": "streamlit"},
                "messages": [
                    ("user",      "ما هو التعلم العميق؟",                                                                                 "", "arabic_msa", "question", "neutral",  0.05),
                    ("assistant", "التعلم العميق هو فرع من تعلم الآلة يستخدم شبكات عصبية متعددة الطبقات لتعلم أنماط معقدة من البيانات.", "", "arabic_msa", "question", "neutral",  0.00),
                    ("user",      "ما اسمك؟",                                                                                            "", "arabic_msa", "question", "neutral",  0.05),
                    ("assistant", "أنا سيدر 🌲 — مساعد ذكاء اصطناعي ثلاثي اللغات. كيف بقدر ساعدك؟",                                       "", "arabic_msa", "question", "neutral",  0.00),
                    ("user",      "شكراً جزيلاً!",                                                                                       "", "arabic_msa", "thanks",   "positive", 0.92),
                    ("assistant", "عفواً! أهلاً فيك",                                                                                    "", "arabic_msa", "thanks",   "positive", 0.75),
                ],
            },
            {
                "session_id": f"demo-arz-{uuid.uuid4().hex[:8]}",
                "offset_hours": 8,
                "metadata": {"locale": "lb-LB", "client": "streamlit"},
                "messages": [
                    ("user",      "keefak ya zalame?",                                                                       "كيفك يا زلمة", "lebanese_arabizi", "greeting", "positive", 0.70),
                    ("assistant", "Ana mni7 hamdellah! W enta, keefak?",                                                     "",             "lebanese_arabizi", "greeting", "positive", 0.65),
                    ("user",      "shu el AI?",                                                                              "شو ال AI",     "lebanese_arabizi", "question", "neutral",  0.10),
                    ("assistant", "AI ya3ne Artificial Intelligence — el zaka2 el istina3i. El ma7ine btit3allem mitl el insan!", "",        "lebanese_arabizi", "question", "neutral",  0.00),
                    ("user",      "wallah ktir mni7!",                                                                       "والله كتير منيح", "lebanese_arabizi", "feedback", "positive", 0.95),
                    ("assistant", "Merci 3al feedback! Bi sa3edne ktir 🌲",                                                  "",             "lebanese_arabizi", "feedback", "positive", 0.80),
                ],
            },
            {
                "session_id": f"demo-mix-{uuid.uuid4().hex[:8]}",
                "offset_hours": 26,
                "metadata": {"locale": "auto", "client": "streamlit"},
                "messages": [
                    ("user",      "Hello there",                                                                                                 "", "english",    "greeting", "positive", 0.60),
                    ("assistant", "Hi! How can I help you today?",                                                                               "", "english",    "greeting", "positive", 0.55),
                    ("user",      "كيف حالك؟",                                                                                                  "", "arabic_msa", "greeting", "positive", 0.70),
                    ("assistant", "أهلاً وسهلاً! أنا منيح، كيفك أنت؟ 🌲",                                                                       "", "arabic_msa", "greeting", "positive", 0.65),
                    ("user",      "Tell me about Lebanon",                                                                                       "", "english",    "question", "neutral",  0.05),
                    ("assistant", "Lebanon is a small country on the Mediterranean coast, known for cedar forests, rich history, and cuisine.", "", "english",    "question", "neutral",  0.00),
                ],
            },
            {
                "session_id": f"demo-arz2-{uuid.uuid4().hex[:8]}",
                "offset_hours": 48,
                "metadata": {"locale": "lb-LB", "client": "streamlit"},
                "messages": [
                    ("user",      "7abibi kifak lyom?",                                                                                          "حبيبي كيفك اليوم", "lebanese_arabizi", "greeting", "positive", 0.80),
                    ("assistant", "Ahla habibi! Tamam hamdellah, w enta?",                                                                       "",                  "lebanese_arabizi", "greeting", "positive", 0.75),
                    ("user",      "shu el deep learning?",                                                                                       "شو ال deep learning", "lebanese_arabizi", "question", "neutral",  0.10),
                    ("assistant", "Deep Learning hiyye teknik bi AI btusted3mil shabakat 3asabiyye ktir tawile la tit3allem min el data.",       "",                  "lebanese_arabizi", "question", "neutral",  0.00),
                ],
            },
        ]

        feedback_pool = []

        for conv in conversations:
            session = ChatSession.objects.create(
                session_id=conv["session_id"],
                is_active=True,
                metadata=conv["metadata"],
            )
            session_start = now - timedelta(hours=conv["offset_hours"])
            session_end = session_start + timedelta(minutes=len(conv["messages"]) * 2)
            ChatSession.objects.filter(id=session.id).update(
                created_at=session_start,
                updated_at=session_end,
            )

            for i, (role, content, original, lang, intent, sentiment, score) in enumerate(conv["messages"]):
                msg = ChatMessage.objects.create(
                    session=session,
                    message_id=uuid.uuid4().hex,
                    role=role,
                    content=content,
                    original_content=original,
                    detected_language=lang,
                    intent=intent,
                    sentiment_label=sentiment,
                    sentiment_score=score,
                )
                msg_time = session_start + timedelta(minutes=i * 2)
                ChatMessage.objects.filter(id=msg.id).update(timestamp=msg_time)

                if role == "assistant":
                    feedback_pool.append((session, msg, msg_time))

        ratings_weights  = [1, 1, 1, 1, 1, 0, -1]
        comments_pool    = ["Great answer!", "Helpful, thanks", "", "Could be more detailed", "", "Spot on", "Loved the Arabizi reply"]
        preferred_pool   = ["", "", "I would have liked a longer explanation", "", "", "", ""]

        for session, msg, msg_time in random.sample(feedback_pool, k=min(8, len(feedback_pool))):
            fb = UserFeedback.objects.create(
                session=session,
                message=msg,
                rating=random.choice(ratings_weights),
                comment=random.choice(comments_pool),
                preferred_response=random.choice(preferred_pool),
            )
            UserFeedback.objects.filter(id=fb.id).update(
                created_at=msg_time + timedelta(seconds=30),
            )

    def _seed_daily_metrics(self):
        today = timezone.now().date()
        for days_ago in range(7):
            date = today - timedelta(days=days_ago)
            growth = 1.0 + (6 - days_ago) * 0.12
            DailyMetrics.objects.update_or_create(
                date=date,
                defaults={
                    "total_sessions":        int(random.randint(8, 18) * growth),
                    "total_messages":        int(random.randint(45, 120) * growth),
                    "unique_users":          int(random.randint(5, 14) * growth),
                    "avg_session_duration":  round(random.uniform(2.5, 9.0), 2),
                    "avg_turns_per_session": round(random.uniform(3.0, 7.5), 2),
                    "english_messages":      int(random.randint(15, 50) * growth),
                    "arabic_messages":       int(random.randint(10, 40) * growth),
                    "arabizi_messages":      int(random.randint(15, 55) * growth),
                    "positive_count":        int(random.randint(20, 70) * growth),
                    "negative_count":        random.randint(2, 12),
                    "neutral_count":         int(random.randint(12, 40) * growth),
                    "feedback_count":        int(random.randint(5, 22) * growth),
                    "avg_rating":            round(random.uniform(0.55, 0.92), 2),
                },
            )

    def _seed_model_performance(self):
        now = timezone.now()
        models_pool = [
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot_small-90M",
        ]
        for i in range(12):
            perf = ModelPerformance.objects.create(
                model_name=random.choice(models_pool),
                avg_response_time_ms=round(random.uniform(450, 2800), 1),
                avg_reward_score=round(random.uniform(0.55, 0.92), 3),
                total_requests=random.randint(50, 380),
                error_count=random.randint(0, 8),
            )
            perf_time = now - timedelta(hours=i * 8)
            ModelPerformance.objects.filter(id=perf.id).update(timestamp=perf_time)

    def _summary(self):
        bar = "=" * 60
        self.stdout.write("")
        self.stdout.write(bar)
        self.stdout.write(self.style.SUCCESS("  Demo data seeded successfully!"))
        self.stdout.write(bar)
        self.stdout.write(f"  Bot configurations:  {BotConfiguration.objects.count()}")
        self.stdout.write(f"  Chat sessions:       {ChatSession.objects.count()}")
        self.stdout.write(f"  Chat messages:       {ChatMessage.objects.count()}")
        self.stdout.write(f"  User feedback:       {UserFeedback.objects.count()}")
        self.stdout.write(f"  Daily metrics:       {DailyMetrics.objects.count()}")
        self.stdout.write(f"  Model performance:   {ModelPerformance.objects.count()}")
        self.stdout.write(bar)
        self.stdout.write("")
        self.stdout.write("  Visit:")
        self.stdout.write("    http://localhost:8001/admin/      (Django admin)")
        self.stdout.write("    http://localhost:8001/            (Dashboard view)")
        self.stdout.write("    http://localhost:8001/analytics/  (Charts)")
        self.stdout.write("")
        self.stdout.write("  Re-seed cleanly with: python manage.py seed_demo --wipe")
        self.stdout.write("")