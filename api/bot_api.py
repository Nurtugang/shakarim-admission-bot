import time
import logging
from google import genai
from google.genai import types
from shakarim_admission_bot.gemini_config import gemini_model
from .knowledge_functions import knowledge_tools

logger = logging.getLogger(__name__)

user_chats = {}


def retry_with_backoff_with_fallback(send_func_factory, max_retries=3, base_delay=1):
    """
    Retries send_func (from a factory) with exponential backoff.
    Switches to a fallback model on the final retry.

    Args:
        send_func_factory (callable): Function that returns a callable send_func.
        max_retries (int): Number of retry attempts.
        base_delay (float): Delay in seconds.

    Returns:
        The result from send_func or raises last exception.
    """
    for attempt in range(max_retries):
        try:
            send_func = send_func_factory(attempt)
            return send_func()
        except Exception as e:
            logging.warning(f"[Retry {attempt+1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise


def smart_ask_gemini(question, user_id):
    """
    Uses Gemini Function Calling (Automatic Python SDK version) to answer questions
    about Shakarim University based on predefined knowledge functions.
    Handles relevance based on the system prompt and available tools.
    """
    if not gemini_model:
        logger.error("Gemini model not initialized for smart_ask_gemini.")
        return "Извините, возникла проблема с конфигурацией AI-помощника. Попробуйте позже."

    system_instruction = f"""
    Ты — дружелюбный и информативный бот-помощник приёмной комиссии Университета Шакарима в городе Семей.
    Твоя задача — отвечать на вопросы абитуриентов и студентов об Университете Шакарима (поступление, обучение, студенческая жизнь, структура, контакты и т.д.).

    Для получения информации используй предоставленные тебе функции (tools).
    Анализируй вопрос пользователя:
    1. Если вопрос касается Университета Шакарима, вызови одну или несколько соответствующих функций для получения ответа.
    2. Если вопрос НЕ касается Университета Шакарима или является общим (например, "как дела?", "расскажи анекдот"), вежливо ответь, что ты можешь помочь только с вопросами об Университете Шакарима.
    3. Если вопрос касается университета, но у тебя НЕТ подходящей функции для ответа, вежливо сообщи, что у тебя нет информации по данному аспекту, но ты можешь ответить на другие вопросы об университете.

    Отвечай всегда на русском языке. Будь кратким, но точным и полным в рамках полученной информации.
    Если в информации присутствуют ссылки, обязательно включай их в ответ в полном виде.
    Никогда не упоминай названия функций в ответе. Отвечай как человек, а не как программа.
    Не упоминай, что ты используешь функции или инструменты — просто предоставь ответ.
    Отвечай сразу по существу: не задавай встречных вопросов и не предлагай "поделиться" или "отправить" информацию — просто делай это.
    При необходимости используйте свои общие знания.
    """


    try:
        if user_id not in user_chats:
            logger.info(f"Creating new chat session for user: {user_id}")
            user_chats[user_id] = gemini_model.client.chats.create(
                model=gemini_model.model_name,
                config=types.GenerateContentConfig(
                    tools=knowledge_tools,
                    temperature=0.2,
                    max_output_tokens=1500,
                    system_instruction=system_instruction
                )
            )

        def send_func_factory(attempt):
            logger.info(f"Sending message to Gemini chat for user {user_id}: {question}")
            if attempt < 2:
                chat = user_chats[user_id]
                return lambda: (
                    logger.info(f"[Try {attempt+1}] Using default model for user {user_id}"),
                    chat.send_message(question)
                )[1]
            else:
                fallback_key = f"{user_id}_fallback"
                if fallback_key not in user_chats:
                    logger.warning("Creating fallback chat session...")
                    user_chats[fallback_key] = gemini_model.client.chats.create(
                        model="gemini-1.5-flash-002",
                        config=types.GenerateContentConfig(
                            tools=knowledge_tools,
                            temperature=0.2,
                            max_output_tokens=1500,
                            system_instruction=system_instruction
                        )
                    )
                chat = user_chats[fallback_key]
                return lambda: chat.send_message(question)

        response = retry_with_backoff_with_fallback(send_func_factory, max_retries=3)
        return response.text

    except Exception as e:
        logger.error(f"Error during smart_ask_gemini: {e}", exc_info=True)
        return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."



