"""
Скрипт для запуска Telegram-бота.
Запускайте этот скрипт ПОСЛЕ запуска Django-сервера (python manage.py runserver).
"""

import os
import sys
import django

# Настроим Django перед импортом моделей
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shakarim_admission_bot.settings')
django.setup()

# Теперь можно импортировать модули из нашего проекта
from bot.bot import main

if __name__ == "__main__":
    print("Запуск бота для Университета Шакарима...")
    main()