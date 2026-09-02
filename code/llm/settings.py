"""Runtime API settings."""

import os

# Setup your API_BASE_URL and API_KEY here
API_BASE_URL = os.getenv("API_BASE_URL", "http://xxx")
API_KEY = os.getenv("API_KEY", "sk-xxx")
API_TIMEOUT = 300.0
API_MAX_RETRIES = 2
