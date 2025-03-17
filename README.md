# shakarim-admission-bot
Shakarim Admission Bot - ИИ-агент от Университета Шакарима города Семей, в рамках инновационной программы AI Sana.

# Как работать с этим репозиторием?
1) Установка зависимостей. Основные зависимости с котороми мы будем работать.
Бэк:
Django              5.1.7
djangorestframework 3.15.2

Языковая модель (LLM), используем gemini-1.5-flash:
google-auth         2.38.0
google-genai        1.5.0

Firebase:
firebase-admin           6.7.0

Ключи храним тут (например ключи API от Gemini и от Firebase):
python-dotenv       1.0.1

2) Получение аутентификационных данных. Ключ API Gemini(он в виде строки) и ключ Firebase (он в виде JSON-файла). Ключ API Gemini берете сами, с ключом Firebase я вам помогу.
3) Создание .env файла в корне проекта, там и пишите свои ключи в формате:
   FIREBASE_APPLICATION_CREDENTIALS=путь к вашему ключу в виде .JSON-файла 
   GEMINI_API_KEY=ваш ключ в виде строки
4) python manage.py runserver

# Текущая API-документация

📌 Получить ответ от Gemini
URL: /api/ask_gemini/
Метод: GET
Пример запроса: http://127.0.0.1:8000/api/ask_gemini/?question=Какие документы нужны для поступления?
Пример ответа:
{
  "answer": "Для поступления вам понадобятся паспорт, аттестат и сертификат ЕНТ."
}

📌 Получить знания из Firestore
URL: /api/get_knowledge/
Метод: GET
Пример запроса: http://127.0.0.1:8000/api/get_knowledge/?category=гранты
Ответ:
[
  {"id": "doc1", "category": "гранты", "text": "Гранты доступны для студентов с высокими баллами."},
  {"id": "doc2", "category": "гранты", "text": "Срок подачи заявки на грант – до 30 августа."}
]

📌 Добавить знания в Firestore
URL: /api/add_knowledge/
Метод: POST
Пример запроса:
{
  "category": "документы",
  "text": "Какие то новые документы в базе знаний."
}

📌 Удалить знания из Firestore
URL: /api/delete_knowledge/<doc_id>/
Метод: DELETE
Пример запроса:
http://127.0.0.1:8000/api/delete_knowledge/fUuYi6ojkJW...

