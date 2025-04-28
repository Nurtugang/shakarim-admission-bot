import logging
from google import genai
from google.genai import types
from shakarim_admission_bot.gemini_config import gemini_model
from .knowledge_functions import knowledge_tools

logger = logging.getLogger(__name__)


def smart_ask_gemini(question):
    """
    Uses Gemini Function Calling (Automatic Python SDK version) to answer questions
    about Shakarim University based on predefined knowledge functions.
    Handles relevance based on the system prompt and available tools.
    """
    if not gemini_model:
        logger.error("Gemini model not initialized for smart_ask_gemini.")
        return "Извините, возникла проблема с конфигурацией AI-помощника. Попробуйте позже."

    system_instruction = f"""
    Ты - дружелюбный и информативный бот-помощник приемной комиссии Университета Шакарима города Семей.
    Твоя задача - отвечать ТОЛЬКО на вопросы абитуриентов и студентов об Университете Шакарима (поступление, обучение, студенческая жизнь, структура, контакты и т.д.).

    Для получения информации используй ТОЛЬКО предоставленные тебе функции (tools).
    Проанализируй вопрос пользователя.
    1. Если вопрос касается Университета Шакарима, вызови одну или несколько подходящих функций для получения ответа.
    2. Если вопрос НЕ касается Университета Шакарима или является общим (например, "как дела?", "расскажи анекдот"), НЕ ИСПОЛЬЗУЙ функции. Вежливо ответь, что ты можешь помочь только с вопросами об Университете Шакарима.
    3. Если вопрос касается университета, но у тебя НЕТ подходящей функции для ответа, НЕ ИСПОЛЬЗУЙ другие функции и не придумывай ответ. Вежливо сообщи, что информацией по этому конкретному аспекту ты не обладаешь, но можешь ответить на другие вопросы об университете.

    Формулируй ответы на основе информации, полученной ИСКЛЮЧИТЕЛЬНО из вызванных функций.
    Отвечай всегда на русском языке. Будь кратким, но точным и полным в пределах полученной информации.
    Не упоминай в ответе, что ты используешь функции или инструменты. Просто предоставь ответ пользователю.
    """

    try:
        logger.info(f"Sending question to Gemini with function calling and integrated relevance check: '{question}'")

        generation_config = types.GenerateContentConfig(
            tools=knowledge_tools,
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens = 1000
        )
        contents = [
            {"role": "user", "parts": [{"text": question}]}
        ]

        response = gemini_model.client.models.generate_content(
            model=gemini_model.model_name,
            contents=contents,
            config=generation_config
        )
        final_answer = response.text
        logger.info(f"Received final answer from Gemini: '{final_answer[:50]}...'")
        return final_answer

    except Exception as e:
        logger.error(f"Error during smart_ask_gemini: {e}", exc_info=True)
        return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
