from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.utils import timezone
from analytics.models import DailyMetrics, ModelPerformance
import datetime

User = get_user_model()


class DailyMetricsModelTests(TestCase):

    def setUp(self):
        self.metrics = DailyMetrics.objects.create(
            date=datetime.date.today(),
            total_sessions=10,
            total_messages=50,
            unique_users=5,
            avg_session_duration=3.5,
            avg_turns_per_session=4.2,
            english_messages=30,
            arabic_messages=15,
            arabizi_messages=5,
            positive_count=20,
            negative_count=10,
            neutral_count=20,
            feedback_count=8,
            avg_rating=4.1,
        )

    def test_str(self):
        self.assertEqual(str(self.metrics), f"Metrics for {datetime.date.today()}: 50 msgs")

    def test_date_is_unique(self):
        with self.assertRaises(Exception):
            DailyMetrics.objects.create(date=datetime.date.today())

    def test_default_values(self):
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        m = DailyMetrics.objects.create(date=yesterday)
        self.assertEqual(m.total_sessions, 0)
        self.assertEqual(m.avg_rating, 0.0)

    def test_ordering_is_descending(self):
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        DailyMetrics.objects.create(date=yesterday)
        metrics = list(DailyMetrics.objects.all())
        self.assertGreater(metrics[0].date, metrics[1].date)


class ModelPerformanceModelTests(TestCase):

    def setUp(self):
        self.perf = ModelPerformance.objects.create(
            model_name="cedar-v1",
            avg_response_time_ms=120.5,
            avg_reward_score=0.87,
            total_requests=200,
            error_count=3,
        )

    def test_str(self):
        self.assertIn("cedar-v1", str(self.perf))
        self.assertIn("120.5ms", str(self.perf))

    def test_avg_reward_score_nullable(self):
        perf = ModelPerformance.objects.create(
            model_name="cedar-v2",
            avg_response_time_ms=95.0,
            avg_reward_score=None,
            total_requests=100,
        )
        self.assertIsNone(perf.avg_reward_score)

    def test_default_error_count(self):
        perf = ModelPerformance.objects.create(
            model_name="cedar-v3",
            avg_response_time_ms=80.0,
            total_requests=50,
        )
        self.assertEqual(perf.error_count, 0)


class AnalyticsViewTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    def test_analytics_overview_loads(self):
        response = self.client.get(reverse('analytics_overview'))
        self.assertEqual(response.status_code, 200)

    def test_analytics_overview_redirects_unauthenticated(self):
        self.client.logout()
        response = self.client.get(reverse('analytics_overview'))
        self.assertEqual(response.status_code, 302)

    def test_context_contains_expected_keys(self):
        response = self.client.get(reverse('analytics_overview'))
        for key in ['daily_metrics', 'language_dist', 'intent_dist', 'sentiment_dist', 'model_perf']:
            self.assertIn(key, response.context)

    def test_daily_metrics_limited_to_30(self):
        for i in range(35):
            DailyMetrics.objects.create(
                date=datetime.date.today() - datetime.timedelta(days=i)
            )
        response = self.client.get(reverse('analytics_overview'))
        self.assertLessEqual(len(response.context['daily_metrics']), 30)

    def test_model_perf_limited_to_10(self):
        for i in range(15):
            ModelPerformance.objects.create(
                model_name=f"model-{i}",
                avg_response_time_ms=100.0,
                total_requests=50,
            )
        response = self.client.get(reverse('analytics_overview'))
        self.assertLessEqual(len(response.context['model_perf']), 10)