from django.urls import path
from .views import get_knowledge, add_knowledge, delete_knowledge, simple_ask_question

urlpatterns = [
    path("get_knowledge/", get_knowledge, name="get_knowledge"),
    path("add_knowledge/", add_knowledge, name="add_knowledge"),
    path("delete_knowledge/<str:doc_id>/", delete_knowledge, name="delete_knowledge"),
    path("ask_gemini/", simple_ask_question),
]
