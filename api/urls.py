from django.urls import path
from .views import smart_ask_question

urlpatterns = [
        path("smart_ask_gemini/", smart_ask_question, name="smart_ask_gemini"),

]
