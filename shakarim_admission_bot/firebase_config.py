import os
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore

# загрузка переменных из .env файла
load_dotenv()

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_APPLICATION_CREDENTIALS")

cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)
firebase_db = firestore.client()
