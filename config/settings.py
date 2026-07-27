# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# ========== 1. ПУТИ К ФАЙЛАМ ==========
CREDS_PATH = BASE_DIR / 'credentials' / 'service_account.json'
CACHE_DIR = BASE_DIR / 'cache'
LOGS_DIR = BASE_DIR / 'logs'

# Проверяем что credentials существуют
if not CREDS_PATH.exists():
    raise FileNotFoundError(f"❌ Файл credentials не найден: {CREDS_PATH}")

# ========== 2. API КЛЮЧИ (ИЗ .ENV) ==========
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DEEPSEEK_API_KEY не найден в .env!")

# ========== 3. TELEGRAM (ИЗ .ENV) ==========
ADMIN_TELEGRAM_ID = os.getenv('ADMIN_TELEGRAM_ID')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# ========== 4. НАСТРОЙКИ (ИЗ .ENV) ==========
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# ========== 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def get_google_credentials():
    """Загружает credentials для Google Sheets"""
    import json
    with open(CREDS_PATH, 'r') as f:
        return json.load(f)