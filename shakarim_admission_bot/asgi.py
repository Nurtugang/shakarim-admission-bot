import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shakarim_admission_bot.settings')

application = get_asgi_application()
