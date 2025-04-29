import os
import logging
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_PRIORITY_LIST = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-lite-001",
]

client = None
gemini_model = None

class GeminiModelWrapper:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def generate_content(self, prompt, **kwargs):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            **kwargs
        )
        return response

    def generate_content_stream(self, prompt, **kwargs):
        return self.client.models.generate_content_stream(
            model=self.model_name,
            contents=[prompt],
            **kwargs
        )

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not found. Gemini features will be disabled.")
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")

        for model_name in MODEL_PRIORITY_LIST:
            try:
                logger.info(f"Testing Gemini model '{model_name}'...")
                test_model = GeminiModelWrapper(client, model_name)
                test_response = test_model.generate_content("test")
                logger.info(f"Model '{model_name}' is working.")
                gemini_model = test_model
                break
            except Exception as test_error:
                logger.warning(f"Model '{model_name}' failed: {test_error}")

        if not gemini_model:
            logger.error("All Gemini models failed. No model is available.")

    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
        gemini_model = None
