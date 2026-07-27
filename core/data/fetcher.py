"""
Модуль для загрузки и нормализации данных из всех доступных Google Sheets таблиц.
Читает каждую таблицу, каждый лист, нормализует данные, сохраняет в Parquet и обновляет индекс.
"""

import sys
import time
import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Добавляем корень проекта в sys.path для импорта config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data.table_discovery import TableDiscovery
from core.data.indexer import SheetIndex
from core.llm.client import DeepSeekClient
from loguru import logger

# Настройка логгера
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/app/logs/normalization.log", rotation="10 MB", level="DEBUG")

# =============================================================================
# НОВЫЕ ЦЕЛЕВЫЕ КОЛОНКИ
# =============================================================================
TARGET_COLUMNS = [
    "status",
    "date",
    "ticket_id",
    "event_group",
    "event_type",
    "client_raw",               # исходное название клиента
    "client_normalized",         # нормализованное (без лишних слов, с уникальной частью)
    "address_raw",               # исходный адрес
    "address_normalized",        # нормализованный с городом (по умолчанию Екатеринбург)
    "problem_raw",               # исходное описание проблемы
    "problem_normalized",        # краткое резюме (5-7 слов)
    "assignee",
    "author",
    "comment",
    "_table_name",
    "_sheet_name",
    "point_key"                  # ключ для группировки по точке (клиент | адрес)
]

# Словарь для маппинга заголовков (синонимы) — оставляем как есть
HEADER_MAP = {
    # Статус
    "статус": "status",

    # Дата 
    "дата": "date",

    # Номер тикета 
    "номер": "ticket_id",

    # Группа события
    "группа события": "event_group",

    # Вид события 
    "вид события": "event_type",

    # Клиент → сырое поле
    "контрагент": "client_raw",

    # Адрес → сырое поле 
    "точка": "address_raw",
    "торг точка": "address_raw",

    # Описание проблемы → сырое поле
    "описание": "problem_raw",

    # Содержание → комментарий
    "содержание": "comment",

    # Ответственный
    "ответственный": "assignee",

    # Автор
    "автор": "author",
}

FUZZY_THRESHOLD = 80
LLM_CALLS_PER_SECOND = 3
MIN_LLM_DELAY = 1.0 / LLM_CALLS_PER_SECOND

class Fetcher:
    """
    Класс для загрузки и нормализации всех данных из Google Sheets.
    """
    def __init__(self):
        self.discovery = TableDiscovery()
        self.llm_client = DeepSeekClient()
        self.cache = self._load_cache()
        self.all_records = []
        self.indexer = SheetIndex()
        self.stats = {
            "tables_processed": 0,
            "sheets_processed": 0,
            "rows_processed": 0,
            "rows_skipped": 0,
            "llm_calls": 0,
            "cache_hits": 0
        }

        # =====================================================================
        # НОВОЕ: флаг очистки кэша (переменная окружения CLEAR_CACHE=true)
        # =====================================================================
        if os.getenv("CLEAR_CACHE", "false").lower() == "true":
            self._clear_cache()

    def _clear_cache(self):
        """Полностью очищает кэш (удаляет JSON-файлы)."""
        cache_dir = Path("/app/cache")
        for key in ["address", "client", "problem"]:
            path = cache_dir / f"{key}_mappings.json"
            if path.exists():
                path.unlink()
                logger.info(f"🗑️ Кэш для '{key}' удалён")
        # Перезагружаем пустой кэш
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, str]]:
        """Загружает кэш маппингов из JSON-файлов."""
        cache = {"address": {}, "client": {}, "problem": {}}
        cache_dir = Path("/app/cache")
        cache_dir.mkdir(exist_ok=True)

        for key in cache:
            path = cache_dir / f"{key}_mappings.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cache[key] = json.load(f)
                    logger.info(f"📦 Загружен кэш для '{key}': {len(cache[key])} записей")
        return cache

    def _save_cache(self):
        """Сохраняет обновлённый кэш в JSON-файлы."""
        cache_dir = Path("/app/cache")
        cache_dir.mkdir(exist_ok=True)

        for key, data in self.cache.items():
            path = cache_dir / f"{key}_mappings.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug("💾 Кэш сохранён")

    # =========================================================================
    # НОВОЕ: сбор уникальных значений для нормализации (теперь собираем сырые)
    # =========================================================================
    def _collect_unique_values(self, rows: List[List[str]], mapping: Dict[str, int]) -> Dict[str, set]:
        """
        Собирает все уникальные сырые значения для полей, требующих нормализации.
        Возвращает словарь {поле: set(уникальных значений)}
        Поля: client_raw, address_raw, problem_raw
        """
        unique = defaultdict(set)
        fields_to_normalize = {"client_raw", "address_raw", "problem_raw"}

        for row in tqdm(rows, desc="📊 Сбор уникальных значений", leave=False):
            for field in fields_to_normalize:
                # В mapping лежат сырые поля (client_raw, address_raw, problem_raw)
                if field in mapping:
                    idx = mapping[field]
                    if idx < len(row) and row[idx]:
                        value = str(row[idx]).strip()
                        if value:
                            unique[field].add(value)

        return unique

    # =========================================================================
    # ИЗМЕНЕНО: нормализация батча для трёх типов полей
    # =========================================================================
    def _normalize_batch(self, field_type: str, raw_values: set) -> Dict[str, str]:
        """
        Нормализует пачку уникальных значений одного типа с помощью батчевого вызова LLM.
        field_type: 'client', 'address' или 'problem' (ключ кэша)
        raw_values: множество сырых строк
        Возвращает словарь {исходное_значение: нормализованное}
        """
        if not raw_values:
            return {}

        logger.info(f"🦙 Начинаем нормализацию {len(raw_values)} уникальных значений для поля '{field_type}'")
        results = {}

        # Сортируем для стабильности
        sorted_values = sorted(raw_values)

        # Размер батча — 50
        batch_size = 50
        total_batches = (len(sorted_values) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sorted_values))
            batch = sorted_values[start:end]

            # Отфильтруем значения, которые уже есть в кэше
            to_normalize = []
            for val in batch:
                key = val.lower()
                if key in self.cache[field_type]:
                    results[val] = self.cache[field_type][key]  # по сырому ключу отдаём нормализованное
                    self.stats["cache_hits"] += 1
                else:
                    to_normalize.append(val)

            if to_normalize:
                try:
                    # Вызываем батчевую нормализацию
                    normalized_list = self.llm_client.normalize_batch(to_normalize, field_type)
                    self.stats["llm_calls"] += 1

                    if len(normalized_list) != len(to_normalize):
                        logger.warning(f"⚠️ Длина ответа ({len(normalized_list)}) не совпадает с длиной запроса ({len(to_normalize)}). Используем исходные значения.")
                        normalized_list = to_normalize  # fallback

                    for orig, norm in zip(to_normalize, normalized_list):
                        results[orig] = norm
                        # сохраняем в кэш по ключу orig.lower()
                        self.cache[field_type][orig.lower()] = norm
                        logger.debug(f"    ✅ {field_type}: '{orig[:30]}...' -> '{norm[:30]}...'")
                except Exception as e:
                    logger.error(f"❌ Ошибка при нормализации батча: {e}")
                    # В случае ошибки возвращаем исходные значения, не сохраняем в кэш
                    for orig in to_normalize:
                        results[orig] = orig
            else:
                logger.debug(f"Все значения батча уже были в кэше")

            # Задержка между батчами (кроме последнего)
            if batch_idx < total_batches - 1:
                time.sleep(MIN_LLM_DELAY)

            # Сохраняем кэш после каждого батча
            self._save_cache()
            logger.debug(f"💾 Кэш сохранён после батча {batch_idx+1}/{total_batches}")

        logger.info(f"✅ Нормализация поля '{field_type}' завершена. Вызовов LLM: {self.stats['llm_calls']}")
        return results

    # =========================================================================
    # ИЗМЕНЕНО: быстрая нормализация с кэшем (теперь возвращает нормализованное по сырому)
    # =========================================================================
    def _normalize_with_cache(self, raw_value: str, field_type: str) -> str:
        """
        Быстрая нормализация с использованием только кэша (без вызова LLM).
        field_type: 'client', 'address', 'problem'
        Возвращает нормализованную строку (если нет в кэше — логируем и возвращаем сырую)
        """
        if not raw_value or not isinstance(raw_value, str):
            return ""

        key = raw_value.strip().lower()
        cached = self.cache[field_type].get(key)

        if cached:
            self.stats["cache_hits"] += 1
            return cached
        else:
            # Если вдруг значение не нашлось в кэше (например, после очистки кэша в процессе работы)
            # Это маловероятно, но на всякий случай делаем одиночный вызов.
            self.stats["llm_calls"] += 1
            logger.warning(f"⚠️ Значение '{raw_value}' не найдено в кэше, вызываем LLM на лету")
            normalized = self.llm_client.normalize(raw_value, field_type)
            self.cache[field_type][key] = normalized
            time.sleep(MIN_LLM_DELAY)
            return normalized

    def _detect_headers(self, rows: List[List[str]]) -> Tuple[bool, Optional[List[str]]]:
        """
        Определяет, есть ли строка заголовков.
    
        Returns:
            (True, headers) — если заголовки найдены
            (False, None) — если нет
        """
        sample_size = min(5, len(rows))
    
        for i in range(sample_size):
            row = rows[i]
            logger.debug(f"🔍 Проверяем строку {i}: {row[:3]}...")  # ← ДОБАВИТЬ

            matches = 0
            for cell in row:
                if not isinstance(cell, str):
                    continue
                cell_lower = cell.lower().strip()
                for header_word in HEADER_MAP.keys():
                    if header_word in cell_lower or fuzz.ratio(header_word, cell_lower) > 80:
                        matches += 1
                        break
        
            logger.debug(f"  Строка {i}: matches={matches}")  # ← ДОБАВИТЬ

            if matches >= 2:
                logger.debug(f"✅ Заголовки найдены: {row}")
                return True, row
    
        # Не нашли — возвращаем False вместо исключения
        logger.warning(f"⚠️ Заголовки не найдены, лист будет пропущен")
        return False, None

    def _map_columns(self, headers: Optional[List[str]], sample_rows: List[List[str]]) -> Dict[str, int]:
        """
        Создаёт маппинг целевых полей на индексы колонок.
        Если заголовков нет — возвращает пустой словарь.
        """
        mapping = {}
    
        if not headers:
            return mapping  # пустой словарь, лист пропустится
    
        for idx, header in enumerate(headers):
            if not isinstance(header, str) or not header.strip():
                continue
            header_clean = header.strip().lower()
            best_match = None
            best_score = 0
        
            for pattern, target in HEADER_MAP.items():
                score = fuzz.ratio(header_clean, pattern)
                if score > best_score:
                    best_score = score
                    best_match = target
        
            if best_score >= FUZZY_THRESHOLD:
                mapping[best_match] = idx
                logger.debug(f"  📌 Колонка '{header}' -> {best_match} (совпадение: {best_score}%)")
    
        return mapping

    def _parse_date(self, value: str) -> Optional[str]:
        """Универсальный парсер дат."""
        if not isinstance(value, str) or not value.strip():
            return None
        from dateutil import parser
        try:
            dt = parser.parse(value, dayfirst=True, fuzzy=True)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            # dd.mm.yyyy HH:MM:SS
            m = re.match(r'(\d{2})[./](\d{2})[./](\d{4})\s+(\d{2}):(\d{2}):(\d{2})', value)
            if m:
                d, mo, y, h, mi, s = m.groups()
                return f"{y}-{mo}-{d} {h}:{mi}:{s}"
            # dd.mm.yyyy HH:MM
            m = re.match(r'(\d{2})[./](\d{2})[./](\d{4})\s+(\d{2}):(\d{2})', value)
            if m:
                d, mo, y, h, mi = m.groups()
                return f"{y}-{mo}-{d} {h}:{mi}:00"
            # dd.mm.yy HH:MM:SS
            m = re.match(r'(\d{2})[./](\d{2})[./](\d{2})\s+(\d{2}):(\d{2}):(\d{2})', value)
            if m:
                d, mo, y, h, mi, s = m.groups()
                y = f"20{y}"
                return f"{y}-{mo}-{d} {h}:{mi}:{s}"
            # dd.mm.yyyy
            m = re.match(r'(\d{2})[./](\d{2})[./](\d{4})$', value.strip())
            if m:
                d, mo, y = m.groups()
                return f"{y}-{mo}-{d} 00:00:00"
            # yyyy-mm-dd HH:MM:SS
            m = re.match(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})', value)
            if m:
                y, mo, d, h, mi, s = m.groups()
                return f"{y}-{mo}-{d} {h}:{mi}:{s}"
            return None

    def _extract_ticket_id(self, value: str) -> str:
        if not isinstance(value, str):
            return ""
        m = re.search(r'(\d+-\d+)', value)
        return m.group(1) if m else value.strip()

    # =========================================================================
    # ИЗМЕНЕНО: обработка листа с двумя проходами (сбор сырых, нормализация, запись)
    # =========================================================================
    def _process_sheet(self, table_name: str, sheet_name: str, rows: List[List[str]]):
        """Обрабатывает один лист с предварительным сбором уникальных значений."""
        logger.debug(f"📑 Начало обработки листа {table_name}/{sheet_name}, всего строк: {len(rows)}")

        if not rows or len(rows) < 2:
            logger.warning(f"Лист {table_name}/{sheet_name} пуст или содержит недостаточно строк")
            return

        # Определяем заголовки
        logger.debug(f"🔍 Определяем заголовки для {sheet_name}...")
        has_headers, headers = self._detect_headers(rows)
        logger.debug(f"✅ Заголовки определены: has_headers={has_headers}")

        start_row = 1 if has_headers else 0
        data_rows = rows[start_row:]
        logger.debug(f"📊 Строк данных после заголовков: {len(data_rows)}")

        # Маппинг колонок
        logger.debug(f"🔄 Маппинг колонок для {sheet_name}...")
        mapping = self._map_columns(headers, data_rows[:10])
        logger.debug(f"✅ Маппинг получен: {mapping}")

        if not mapping:
            logger.warning(f"❌ Не удалось определить колонки для {table_name}/{sheet_name}, пропускаем")
            return

        # =====================================================================
        # ЭТАП 1: собираем уникальные сырые значения для LLM
        # =====================================================================
        logger.info(f"🔍 Сбор уникальных значений для {sheet_name}...")
        unique_raw_values = self._collect_unique_values(data_rows, mapping)
        # unique_raw_values содержит ключи: client_raw, address_raw, problem_raw

        # =====================================================================
        # ЭТАП 2: нормализуем пачками с задержками
        # =====================================================================
        # Для каждого типа поля нормализуем сырые значения
        # field_type в кэше: 'client', 'address', 'problem'
        field_type_map = {
            "client_raw": "client",
            "address_raw": "address",
            "problem_raw": "problem"
        }

        for raw_field, field_type in field_type_map.items():
            if raw_field in unique_raw_values and unique_raw_values[raw_field]:
                # Проверяем, каких сырых значений нет в кэше
                values_to_normalize = set()
                for val in unique_raw_values[raw_field]:
                    if val.lower() not in self.cache[field_type]:
                        values_to_normalize.add(val)

                if values_to_normalize:
                    logger.info(f"📦 Для поля '{field_type}' нужно нормализовать {len(values_to_normalize)} новых значений")
                    self._normalize_batch(field_type, values_to_normalize)
                else:
                    logger.info(f"✅ Все значения поля '{field_type}' уже есть в кэше")

        # =====================================================================
        # ЭТАП 3: обрабатываем строки с использованием кэша
        # =====================================================================
        sheet_records = []
        rows_processed = 0
        rows_skipped = 0

        for i, row in enumerate(tqdm(data_rows, desc=f"📝 Обработка строк {sheet_name}", leave=False)):
            if not row or all(not cell for cell in row):
                rows_skipped += 1
                continue

            record = {col: None for col in TARGET_COLUMNS}
            record["_table_name"] = table_name
            record["_sheet_name"] = sheet_name

            for target, idx in mapping.items():
                if idx >= len(row):
                    continue
                value = row[idx] if row[idx] is not None else ""

                # Обработка в зависимости от типа поля
                if target == "date":
                    parsed = self._parse_date(value)
                    record[target] = parsed if parsed else value
                elif target == "ticket_id":
                    record[target] = self._extract_ticket_id(value)
                elif target in ("client_raw", "address_raw", "problem_raw"):
                    # Сохраняем сырое значение
                    record[target] = value.strip() if isinstance(value, str) else value

                    # Определяем тип для нормализации
                    if target == "client_raw":
                        norm_field = "client_normalized"
                        cache_type = "client"
                    elif target == "address_raw":
                        norm_field = "address_normalized"
                        cache_type = "address"
                    elif target == "problem_raw":
                        norm_field = "problem_normalized"
                        cache_type = "problem"
                    else:
                        continue

                    # Получаем нормализованное значение из кэша
                    if value and isinstance(value, str):
                        normalized = self._normalize_with_cache(value, cache_type)
                        record[norm_field] = normalized
                    else:
                        record[norm_field] = ""
                else:
                    # status, event_group, assignee, author
                    record[target] = value.strip() if isinstance(value, str) else value

            # Вычисляем point_key на основе нормализованных клиента и адреса
            client_norm = record.get('client_normalized')
            addr_norm = record.get('address_normalized')
            if client_norm and addr_norm:
                record['point_key'] = f"{client_norm} | {addr_norm}"
            elif client_norm:
                record['point_key'] = client_norm
            elif addr_norm:
                record['point_key'] = addr_norm
            else:
                record['point_key'] = None

            self.all_records.append(record)
            sheet_records.append(record)
            rows_processed += 1

        # Обновляем индекс для этого листа
        if sheet_records:
            df_sheet = pd.DataFrame(sheet_records)
            self.indexer.update_from_data(df_sheet, table_name, sheet_name)
            logger.debug(f"✅ Лист {table_name}/{sheet_name} обработан, добавлено {len(sheet_records)} записей")
        else:
            logger.warning(f"⚠️ Лист {table_name}/{sheet_name} не дал записей")

        # Обновляем статистику
        self.stats["sheets_processed"] += 1
        self.stats["rows_processed"] += rows_processed
        self.stats["rows_skipped"] += rows_skipped

    def fetch_all(self):
        """Главный метод: загружает все данные, сохраняет Parquet и обновляет индекс."""
        logger.info("🚀 Начинаем загрузку данных из Google Sheets")

        # Получаем список таблиц
        tables = self.discovery.get_all_tables()
        logger.info(f"✅ Найдено таблиц: {len(tables)}")

        # Очищаем старые данные перед загрузкой
        self.all_records = []    # ← ВОТ ЭТО ДОБАВИТЬ
        self.indexer.clear()     # ← Это уже есть

        # Подсчитываем общее количество листов для прогресс-бара
        total_sheets = sum(len(table["sheets"]) for table in tables)

        # Основной цикл обработки с общим прогресс-баром
        with tqdm(total=total_sheets, desc="📊 Общий прогресс", unit="лист") as pbar:
            for table in tables:
                table_id = table["id"]
                table_name = table["name"]
                logger.info(f"📄 Обработка таблицы: {table_name}")
                self.stats["tables_processed"] += 1

                for sheet in table["sheets"]:
                    sheet_name = sheet["name"]
                    pbar.set_description(f"📑 {table_name}/{sheet_name}")

                    try:
                        # Читаем данные листа через Sheets API
                        credentials = self.discovery.get_credentials()
                        service = build('sheets', 'v4', credentials=credentials)
                        result = service.spreadsheets().values().get(
                            spreadsheetId=table_id,
                            range=sheet_name
                        ).execute()
                        rows = result.get('values', [])
                        self._process_sheet(table_name, sheet_name, rows)
                    except HttpError as e:
                        logger.error(f"❌ Ошибка при чтении листа {table_name}/{sheet_name}: {e}")
                    except Exception as e:
                        logger.error(f"❌ Неожиданная ошибка при обработке {table_name}/{sheet_name}: {e}")

                    pbar.update(1)

        # Сохраняем все записи в Parquet
        if self.all_records:
            df = pd.DataFrame(self.all_records)

           # Удаляем дубликаты по ticket_id (если есть)
            if "ticket_id" in df.columns:
                initial_count = len(df)
                df = df.drop_duplicates(subset=["ticket_id"], keep="first")
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    logger.info(f"🗑️ Удалено {duplicates_removed} дубликатов по ticket_id")

            parquet_path = Path("/app/core/data/current.parquet")
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            logger.success(f"💾 Сохранено {len(df)} записей в {parquet_path}")

            # Сохраняем кэш нормализации
            self._save_cache()

            # Сохраняем индекс
            self.indexer.save()
            # Логируем статистику индекса
            logger.info(f"📊 Статистика индекса: {len(self.indexer.sheets)} листов")  # ← ИСПРАВЛЕНО

            # Выводим итоговую статистику
            logger.info("📊 Итоговая статистика:")
            logger.info(f"  • Обработано таблиц: {self.stats['tables_processed']}")
            logger.info(f"  • Обработано листов: {self.stats['sheets_processed']}")
            logger.info(f"  • Обработано строк: {self.stats['rows_processed']}")
            logger.info(f"  • Пропущено пустых строк: {self.stats['rows_skipped']}")
            logger.info(f"  • Вызовов LLM: {self.stats['llm_calls']}")
            logger.info(f"  • Попаданий в кэш: {self.stats['cache_hits']}")
            logger.info(f"  • Всего записей сохранено: {len(self.all_records)}")
        else:
            logger.warning("⚠️ Нет данных для сохранения")