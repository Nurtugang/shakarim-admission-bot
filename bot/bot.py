import os
import logging
import requests
from dotenv import load_dotenv
from api.bot_api import user_chats
from telegram import Update, ParseMode
from shakarim_admission_bot.gemini_config import client
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_BASE_URL = "http://127.0.0.1:8000/api"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Обработчик команды /start
def start(update: Update, context: CallbackContext) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    user_first_name = update.effective_user.first_name
    welcome_message = (
        f"👋 Привет, {user_first_name}!\n\n"
        f"Я бот-помощник для поступающих в Университет Шакарима города Семей. "
        f"Задайте мне вопрос о поступлении, и я постараюсь на него ответить.\n\n"
        f"Например:\n"
        f"- Какие документы нужны для поступления?\n"
        f"- Какие есть гранты?\n"
        f"- Когда начинается прием документов?"
    )
    update.message.reply_text(welcome_message)

# Обработчик команды /help
def help_command(update: Update, context: CallbackContext) -> None:
    """Отправляет информацию о возможностях бота при команде /help."""
    help_message = (
        "🔍 Вот что я умею:\n\n"
        "- Отвечать на вопросы о поступлении в Университет Шакарима\n"
        "- Предоставлять информацию о грантах, документах и сроках\n"
        "- Помогать с процессом поступления\n\n"
        "Просто напишите свой вопрос, и я постараюсь помочь!"
    )
    update.message.reply_text(help_message)

# Обработчик текстовых сообщений
def handle_message(update: Update, context: CallbackContext) -> None:
    user_question = update.message.text
    user_id = update.effective_user.id

    context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    try:
        response = requests.get(
            f"{API_BASE_URL}/smart_ask_gemini/",
            params={
                "question": user_question,
                "user_id": user_id
            }
        )

        if response.status_code == 200:
            answer = response.json().get("answer", "Извините, я не смог найти ответ на ваш вопрос.")
            update.message.reply_text(answer)
        else:
            update.message.reply_text(
                "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
            )
    except Exception as e:
        logger.error(f"Error: {e}")
        update.message.reply_text(
            "Извините, произошла ошибка. Пожалуйста, попробуйте позже."
        )

def reset_command(update: Update, context: CallbackContext) -> None:
    """Сбросить историю чата пользователя."""
    user_id = update.effective_user.id

    if user_id in user_chats:
        del user_chats[user_id]
        update.message.reply_text("✅ История вашего чата была очищена. Вы можете начать новый диалог.")
    else:
        update.message.reply_text("ℹ️ У вас нет активного диалога для сброса.")

def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("reset", reset_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
