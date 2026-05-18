from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model

User = get_user_model()


class CoreViewTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    def test_dashboard_loads(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_dashboard_redirects_unauthenticated(self):
        self.client.logout()
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)

    def test_session_detail_loads(self):
        response = self.client.get(reverse('session_detail', args=['test-session-123']))
        self.assertIn(response.status_code, [200, 404])

    def test_session_detail_redirects_unauthenticated(self):
        self.client.logout()
        response = self.client.get(reverse('session_detail', args=['test-session-123']))
        self.assertEqual(response.status_code, 302)