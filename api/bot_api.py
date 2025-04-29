from shakarim_admission_bot.gemini_config import ask_gemini
from shakarim_admission_bot.firebase_config import firebase_db

def get_relevant_knowledge(question):
    """
    Получает релевантные знания из базы данных на основе вопроса.
    Простая реализация без сложных алгоритмов.
    """
    # Получаем все документы из базы знаний
    docs = firebase_db.collection("knowledge-base").stream()
    
    knowledge_list = []
    
    for doc in docs:
        try:
            # Пытаемся получить текст из документа
            knowledge_list.append(doc.to_dict()["text"])
        except KeyError:
            # Если не удалось найти ключ 'text', выводим информацию
            print(f"Документ с ID {doc.id} не содержит поля 'text'.")
    
    # Объединяем всю базу знаний в один текст
    knowledge_text = "\n".join(knowledge_list)
    
    return knowledge_text


def smart_ask_gemini(question):
    """
    Улучшенная версия функции ask_gemini, которая включает контекст
    из базы знаний Firebase для более точных ответов.
    """

    # Получаем релевантные знания из базы данных
    knowledge = get_relevant_knowledge(question)
    
    # Создаем системную инструкцию, которая включает контекст
    system_instruction = f"""
    Ты - бот-помощник для поступающих в Университет Шакарима города Семей.
    
    Твоя задача - давать точные и полезные ответы на вопросы о поступлении.
    Будь дружелюбным, информативным и кратким.
    
    Используй предоставленные знания для ответов на вопросы:
    
    {knowledge}
    
    Если в предоставленных знаниях нет ответа на вопрос, используй свои общие знания.
    Но всегда отдавай приоритет информации из базы знаний.
    
    Отвечай на языке котором написал пользователь.
    """
    
    # Отправляем запрос к Gemini с контекстом
    response = ask_gemini(
        question=question,
        system_instruction=system_instruction,
        max_output_tokens=500,  # Увеличиваем максимальную длину ответа
        temperature=0.2  # Делаем ответы более точными
    )
    
    return response