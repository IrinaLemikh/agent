"""
Модуль для обнаружения всех Google Sheets таблиц и их листов,
доступных сервисному аккаунту.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Добавляем корневую папку проекта в sys.path, чтобы найти config.settings
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import CREDS_PATH
from loguru import logger

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]


class TableDiscovery:
    """
    Класс для поиска всех доступных таблиц и их листов.
    """
    def __init__(self, credentials_path: str = str(CREDS_PATH)):
        self.credentials_path = credentials_path
        self.drive_service = None
        self.sheets_service = None
        self.creds = None
        self._authenticate()

    def _authenticate(self):
        """Аутентификация и создание сервисов Drive и Sheets."""
        self.creds = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES)
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        self.sheets_service = build('sheets', 'v4', credentials=self.creds)

    def get_credentials(self):
        """Возвращает учетные данные для использования в других модулях."""
        if not hasattr(self, 'creds') or self.creds is None:
            self._authenticate()
        return self.creds
        
    def get_all_tables(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех таблиц (Google Sheets), доступных сервисному аккаунту.
        Каждый элемент: {'id': '...', 'name': '...', 'sheets': [...]}
        """
        tables = []
        page_token = None
        try:
            while True:
                response = self.drive_service.files().list(
                    q="mimeType='application/vnd.google-apps.spreadsheet'",
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageToken=page_token
                ).execute()
                for file in response.get('files', []):
                    table_info = {
                        'id': file['id'],
                        'name': file['name'],
                        'sheets': self._get_sheets(file['id'])
                    }
                    tables.append(table_info)
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
        return tables

    def _get_sheets(self, spreadsheet_id: str) -> List[Dict[str, str]]:
        """
        Возвращает список листов для конкретной таблицы.
        Каждый элемент: {'id': sheet_id, 'name': sheet_name}
        """
        sheets = []
        try:
            spreadsheet = self.sheets_service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                fields='sheets.properties'
            ).execute()
            for sheet in spreadsheet.get('sheets', []):
                props = sheet.get('properties', {})
                sheets.append({
                    'id': props.get('sheetId'),
                    'name': props.get('title')
                })
        except HttpError as error:
            logger.error(f'An error occurred while fetching sheets for {spreadsheet_id}: {error}')
        return sheets

    def refresh_cache(self):
        """
        Обновляет кэш (если потребуется). Пока просто возвращает список.
        """
        return self.get_all_tables()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("🚀 Запуск теста TableDiscovery")
    td = TableDiscovery()
    tables = td.get_all_tables()
    logger.success(f"✅ Найдено таблиц: {len(tables)}")
    for t in tables:
        logger.info(f"📄 Таблица: {t['name']} ({t['id']}) – листов: {len(t['sheets'])}")
        for s in t['sheets']:
            logger.info(f"   📑 Лист: {s['name']} (id: {s['id']})")