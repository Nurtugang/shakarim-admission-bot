from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

from api.bot_api import smart_ask_gemini
from .serializers import KnowledgeBaseSerializer

# умный запрос к Gemini AI с использованием базы знаний
@api_view(["GET"])
def smart_ask_question(request):
    question = request.GET.get("question", None)
    user_id = request.GET.get("user_id", None)

    if not question:
        return Response({"error": "Вопрос не задан."}, status=400)

    if not user_id:
        return Response({"error": "Не указан идентификатор пользователя."}, status=400)

    answer = smart_ask_gemini(question, user_id)
    return Response({"answer": answer})
