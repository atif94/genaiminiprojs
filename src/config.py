import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    'ALPHA_VANTAGE_KEY': os.getenv('ALPHA_VANTAGE_KEY'),
    'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
    'CACHE_TTL': 3600,
    'MAX_RETRIES': 3,
    'TIMEOUT': 10,
} 