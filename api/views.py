from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import KnowledgeBaseSerializer
from shakarim_admission_bot.firebase_config import firebase_db
from shakarim_admission_bot.gemini_config import ask_gemini

# получение всех данных из базы знаний (firebase)
@api_view(["GET"])
def get_knowledge(request):
    category = request.GET.get("category", None)

    if category:
        docs = firebase_db.collection("knowledge-base").where("category", "==", category).stream()
    else:
        docs = firebase_db.collection("knowledge-base").stream()

    data = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    return Response(data, status=status.HTTP_200_OK)


# добавление данных в базу знаний (firebase)
@api_view(["POST"])
def add_knowledge(request):
    serializer = KnowledgeBaseSerializer(data=request.data)
    if serializer.is_valid():
        firebase_db.collection("knowledge-base").add(serializer.validated_data)
        return Response({"message": "Знание добавлено!"}, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# удаление данных из базы знаний (firebase)
@api_view(["DELETE"])
def delete_knowledge(request, doc_id):
    doc_ref = firebase_db.collection("knowledge-base").document(doc_id)

    if doc_ref.get().exists:
        doc_ref.delete()
        return Response({"message": "Знание удалено!"}, status=status.HTTP_204_NO_CONTENT)
    else:
        return Response({"error": "Документ не найден"}, status=status.HTTP_404_NOT_FOUND)

# базовый запрос к Gemini AI
@api_view(["GET"])
def simple_ask_question(request):
    question = request.GET.get("question", None)
    
    if not question:
        return Response({"error": "Вопрос не задан."}, status=400)

    answer = ask_gemini(question)
    return Response({"answer": answer})