from django.urls import path
from core import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("session/<str:session_id>/", views.session_detail, name="session_detail"),
]