import os
import logging
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.0-flash"
client = None
gemini_model = None

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not found. Gemini features will be disabled.")
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")

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

        gemini_model = GeminiModelWrapper(client, MODEL_NAME)

        try:
            logger.info(f"Testing Gemini model '{MODEL_NAME}'...")
            test_response = gemini_model.generate_content("test")
            logger.info("Test call succeeded.")
        except Exception as test_error:
            logger.error(f"Test call failed: {test_error}")
            gemini_model = None

    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
        gemini_model = None
