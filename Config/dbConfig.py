from dotenv import load_dotenv
import os

# Load credentials from .env file
load_dotenv()

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")
