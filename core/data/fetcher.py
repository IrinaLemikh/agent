"""
Модуль для загрузки и нормализации данных из всех доступных Google Sheets таблиц.
Читает каждую таблицу, каждый лист, нормализует данные, сохраняет в Parquet и обновляет индекс.

=============================================================================
ИНТЕРФЕЙСЫ ДРУГИХ ФАЙЛОВ (реализованы, актуально)
=============================================================================
1) core/data/categories.py -> класс CategoriesManager (реализован)
   - __init__(self, path: Path = Path("/app/config/categories.json"))
   - get_category_names(self) -> List[str]
       Список названий категорий (для fuzzy-сравнения в слое 1 и enum в function calling).
   - to_prompt_text(self) -> str
       Развёрнутый текст со списком категорий + их description/hint,
       готовый для вставки в промпт LLM (слой 2).

2) core/data/glossary.py -> класс Glossary (реализован, to_prompt_text() добавлен)
   - to_prompt_text(self) -> str
       Весь словарь целиком плоским текстом для вставки в промпт — основной
       способ подачи словаря модели. lookup()/list_terms()/тул-схемы оставлены
       про запас, в текущей архитектуре не используются.

3) core/llm/client.py -> класс DeepSeekClient (УЖЕ ПЕРЕПИСАН, актуально)
   - categorize_and_normalize_batch(items, category_names, categories_text, glossary_text)
       -> Dict[str, Dict[str, Any]] — {"1": {"normalized": "...", "tags": ["..."]}, ...}.
       Слой 2 для проблем, function calling с enum по категориям.
   - normalize_batch_dict(items: Dict[str, str], field_type: str) -> Dict[str, str]
       -> {"1": "нормализованная строка", ...}. Батч для client/address,
       тот же паттерн dict-по-id, что и у categorize_and_normalize_batch.
       Заменил собой старый текстовый normalize_batch() (построчный парсинг +
       позиционный zip()) — метод удалён из client.py, легаси не осталось.
   - normalize(text, field_type) -> str — одиночная нормализация (live-фоллбэк
       в fetcher._normalize_with_cache), самостоятельный текстовый промпт,
       не зависит от normalize_batch_dict.
   Оба батчевых метода при исчерпании ретраев бросают LLMCallError —
   fetcher.py ловит именно этот тип и НЕ кэширует результат при ошибке.

Все три файла реализованы и согласованы между собой по сигнатурам и путям
(единая конвенция /app/... — см. core/data/categories.py и core/data/glossary.py).
=============================================================================
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
from core.llm.client import DeepSeekClient, LLMCallError
from core.data.glossary import Glossary
# НОВОЕ: менеджер категорий (файл ещё предстоит написать — см. контракт выше)
from core.data.categories import CategoriesManager
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
    "problem_tags",               # НОВОЕ: список категорий (1-3 тега на обращение)
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

FUZZY_THRESHOLD = 80  # порог для сопоставления ЗАГОЛОВКОВ КОЛОНОК (не путать с категориями)
LLM_CALLS_PER_SECOND = 3
MIN_LLM_DELAY = 1.0 / LLM_CALLS_PER_SECOND

# =============================================================================
# НОВОЕ: константы слоя 1 (детерминированный fuzzy-матч по категориям) и
# размер батча для слоя 2 именно для проблем (отдельно от client/address,
# т.к. увеличиваем ради консистентности формулировок problem_normalized)
# =============================================================================
CATEGORY_FUZZY_THRESHOLD = 90
PROBLEM_BATCH_SIZE = 100


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
            "cache_hits": 0,
            # НОВОЕ: отдельная метрика — сколько проблем закрыл слой 1 без LLM
            "layer1_matches": 0,
        }

        # НОВОЕ: словарь и категории — читаются один раз, текст для промпта
        # кэшируем в атрибутах, чтобы не пересобирать на каждый батч
        self.glossary = Glossary()
        self.categories = CategoriesManager()
        self._glossary_prompt_text = self.glossary.to_prompt_text()
        self._categories_prompt_text = self.categories.to_prompt_text()
        self._category_names = self.categories.get_category_names()

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

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Загружает кэш маппингов из JSON-файлов.

        Формат значений разный по типам:
        - address / client: {raw_lower: "нормализованная строка"}
        - problem (НОВОЕ): {raw_lower: {"normalized": "...", "tags": [...]}}
        Сам метод загрузки/сохранения не зависит от формата значения — json
        одинаково хранит и строки, и словари, поэтому логика ниже не меняется.
        """
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
    # ИЗМЕНЕНО: нормализация батча client/address — теперь тот же паттерн,
    # что и у _categorize_and_normalize_problems_batch: id→значение,
    # dict-ответ от LLM (function calling), то что не пришло в ответе —
    # НЕ кэшируем, оставляем на следующий запуск. Старый текстовый
    # normalize_batch() (построчный парсинг + позиционный zip()) удалён из
    # client.py — вызывающий код здесь уже не проверяет длину ответа,
    # рассинхрон по количеству элементов больше структурно невозможен.
    # =========================================================================
    def _normalize_batch(self, field_type: str, raw_values: set) -> Dict[str, str]:
        """
        Нормализует пачку уникальных значений одного типа (client/address).
        raw_values — уже отфильтрованы вызывающим кодом (_process_sheet) от
        значений, которые есть в кэше, повторной проверки кэша здесь нет.
        Возвращает {исходное_значение: нормализованное} — только для тех
        значений, которые реально попали в кэш в этом вызове.
        """
        if not raw_values:
            return {}

        logger.info(f"🦙 Начинаем нормализацию {len(raw_values)} уникальных значений для поля '{field_type}'")
        results: Dict[str, str] = {}

        sorted_values = sorted(raw_values)
        batch_size = 50
        total_batches = (len(sorted_values) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sorted_values))
            batch = sorted_values[start:end]

            # id локальный для этого вызова (не завязан на позицию в батче
            # содержательно — просто ключ для сопоставления с ответом LLM)
            id_to_value = {str(i + 1): val for i, val in enumerate(batch)}

            try:
                response = self.llm_client.normalize_batch_dict(
                    items=id_to_value,
                    field_type=field_type,
                )
                self.stats["llm_calls"] += 1

                for item_id, val in id_to_value.items():
                    if item_id in response:
                        normalized = response[item_id]
                        results[val] = normalized
                        self.cache[field_type][val.lower()] = normalized
                        logger.debug(f"    ✅ {field_type}: '{val[:30]}...' -> '{normalized[:30]}...'")
                    else:
                        # Модель потеряла пункт — НЕ кэшируем мусор, оставляем
                        # значение необработанным до следующего запуска.
                        logger.warning(f"⚠️ LLM не вернула ответ для пункта {item_id} ('{val[:40]}...'), "
                                       f"пропускаем без кэширования")

            except LLMCallError as e:
                logger.error(f"❌ Ошибка при нормализации батча '{field_type}': {e}")
                # Ничего не кэшируем — весь батч останется "трудным остатком"
                # и будет повторно обработан при следующем запуске.

            if batch_idx < total_batches - 1:
                time.sleep(MIN_LLM_DELAY)

            self._save_cache()
            logger.debug(f"💾 Кэш сохранён после батча {batch_idx+1}/{total_batches}")

        logger.info(f"✅ Нормализация поля '{field_type}' завершена. Вызовов LLM: {self.stats['llm_calls']}")
        return results

    # =========================================================================
    # НОВОЕ: Слой 1 — детерминированный fuzzy-матч сырого текста с названиями
    # категорий, без обращения к LLM.
    # =========================================================================
    def _match_category_by_fuzzy(self, problem_raw: str) -> Optional[str]:
        """
        Сравнивает сырой текст обращения с названиями категорий из categories.json.
        Использует token_sort_ratio (устойчив к перестановке слов/пунктуации),
        порог CATEGORY_FUZZY_THRESHOLD = 90.

        Возвращает название категории при уверенном совпадении, иначе None.
        """
        if not problem_raw or not isinstance(problem_raw, str):
            return None

        best_match = None
        best_score = 0
        for category_name in self._category_names:
            score = fuzz.token_sort_ratio(problem_raw, category_name)
            if score > best_score:
                best_score = score
                best_match = category_name

        if best_score >= CATEGORY_FUZZY_THRESHOLD:
            logger.debug(f"    🎯 Слой 1: '{problem_raw[:40]}...' -> '{best_match}' (score={best_score})")
            return best_match

        return None

    # =========================================================================
    # НОВОЕ: Слой 2 — LLM-батч только для "трудного остатка" проблем.
    # Dict-формат запроса/ответа по номеру пункта (не позиционный список).
    # =========================================================================
    def _categorize_and_normalize_problems_batch(self, raw_values: set) -> Dict[str, Dict[str, Any]]:
        """
        Нормализует и категоризирует пачку уникальных сырых описаний проблем.

        Для каждого значения:
        1. Сначала пробуем слой 1 (fuzzy-матч по категориям) — без LLM.
        2. То, что не совпало — уходит в LLM батчами по PROBLEM_BATCH_SIZE,
           с dict-запросом/ответом по номеру пункта.

        Возвращает {исходное_значение: {"normalized": "...", "tags": [...]}}.
        """
        if not raw_values:
            return {}

        logger.info(f"🦙 Начинаем нормализацию+категоризацию {len(raw_values)} уникальных проблем")
        results: Dict[str, Dict[str, Any]] = {}
        remaining = []

        # ---- Слой 1: fuzzy-матч, без LLM ----
        for val in sorted(raw_values):
            matched_category = self._match_category_by_fuzzy(val)
            if matched_category:
                entry = {"normalized": matched_category, "tags": [matched_category]}
                results[val] = entry
                self.cache["problem"][val.lower()] = entry
                self.stats["layer1_matches"] += 1
            else:
                remaining.append(val)

        logger.info(f"✅ Слой 1 закрыл {len(raw_values) - len(remaining)} значений без LLM, "
                    f"осталось {len(remaining)} для слоя 2")

        self._save_cache()

        if not remaining:
            return results

        # ---- Слой 2: LLM батчами ----
        total_batches = (len(remaining) + PROBLEM_BATCH_SIZE - 1) // PROBLEM_BATCH_SIZE

        for batch_idx in range(total_batches):
            start = batch_idx * PROBLEM_BATCH_SIZE
            end = min(start + PROBLEM_BATCH_SIZE, len(remaining))
            batch = remaining[start:end]

            # id -> сырое значение, id локальный для этого вызова (не завязан
            # на индекс строки/колонки — это и защищает от "схлопывания",
            # если в таблице вдруг нет какой-то колонки)
            id_to_value = {str(i + 1): val for i, val in enumerate(batch)}

            try:
                response = self.llm_client.categorize_and_normalize_batch(
                    items=id_to_value,
                    category_names=self._category_names,
                    categories_text=self._categories_prompt_text,
                    glossary_text=self._glossary_prompt_text,
                )
                self.stats["llm_calls"] += 1

                for item_id, val in id_to_value.items():
                    if item_id in response:
                        entry = response[item_id]
                        # Небольшая защита от кривого ответа модели
                        normalized = entry.get("normalized") or val
                        tags = entry.get("tags") or ["Нераспределено"]
                        entry = {"normalized": normalized, "tags": tags}
                        results[val] = entry
                        self.cache["problem"][val.lower()] = entry
                        logger.debug(f"    ✅ problem: '{val[:30]}...' -> '{normalized[:30]}...' {tags}")
                    else:
                        # Модель потеряла пункт — НЕ кэшируем мусор, оставляем
                        # значение необработанным до следующего запуска.
                        logger.warning(f"⚠️ LLM не вернула ответ для пункта {item_id} ('{val[:40]}...'), "
                                       f"пропускаем без кэширования")

            except LLMCallError as e:
                logger.error(f"❌ Ошибка при категоризации батча проблем: {e}")
                # Ничего не кэшируем — весь батч останется "трудным остатком"
                # и будет повторно обработан при следующем запуске.

            if batch_idx < total_batches - 1:
                time.sleep(MIN_LLM_DELAY)

            self._save_cache()
            logger.debug(f"💾 Кэш сохранён после батча проблем {batch_idx+1}/{total_batches}")

        logger.info(f"✅ Нормализация+категоризация проблем завершена. "
                    f"Слой 1: {self.stats['layer1_matches']}, вызовов LLM: {self.stats['llm_calls']}")
        return results

    # =========================================================================
    # ИЗМЕНЕНО: быстрая нормализация с кэшем для client/address (без изменений)
    # =========================================================================
    def _normalize_with_cache(self, raw_value: str, field_type: str) -> str:
        """
        Быстрая нормализация с использованием только кэша (без вызова LLM).
        field_type: 'client' или 'address' (для 'problem' — см.
        _normalize_with_cache_problem ниже).
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
            try:
                normalized = self.llm_client.normalize(raw_value, field_type)
                self.cache[field_type][key] = normalized
                time.sleep(MIN_LLM_DELAY)
                return normalized
            except LLMCallError as e:
                # Не кэшируем и не роняем обработку всего листа из-за одного
                # значения — строка получит сырой текст вместо нормализованного,
                # а на следующем прогоне (значения нет в кэше) попытка повторится.
                logger.error(f"❌ Ошибка при одиночной нормализации '{raw_value[:40]}...': {e}")
                return raw_value

    # =========================================================================
    # НОВОЕ: быстрая нормализация+категоризация проблемы с использованием
    # только кэша (без вызова LLM). Отдельный метод, т.к. возвращает dict,
    # а не строку.
    # =========================================================================
    def _normalize_with_cache_problem(self, raw_value: str) -> Dict[str, Any]:
        """
        Возвращает {"normalized": "...", "tags": [...]} для сырого описания
        проблемы, используя кэш. Если значения нет в кэше (редкий случай,
        например кэш почистили в процессе работы) — прогоняет его через
        слой 1, а при неудаче — одиночным вызовом LLM (батч из одного пункта).
        """
        if not raw_value or not isinstance(raw_value, str):
            return {"normalized": "", "tags": []}

        key = raw_value.strip().lower()
        cached = self.cache["problem"].get(key)

        if cached:
            self.stats["cache_hits"] += 1
            return cached

        logger.warning(f"⚠️ Проблема '{raw_value}' не найдена в кэше, обрабатываем на лету")

        # Пробуем слой 1 даже "на лету"
        matched_category = self._match_category_by_fuzzy(raw_value)
        if matched_category:
            entry = {"normalized": matched_category, "tags": [matched_category]}
            self.cache["problem"][key] = entry
            self.stats["layer1_matches"] += 1
            return entry

        # Слой 2 — одиночный вызов (батч из одного пункта)
        try:
            response = self.llm_client.categorize_and_normalize_batch(
                items={"1": raw_value},
                category_names=self._category_names,
                categories_text=self._categories_prompt_text,
                glossary_text=self._glossary_prompt_text,
            )
            self.stats["llm_calls"] += 1
            entry_raw = response.get("1")
            if entry_raw:
                entry = {
                    "normalized": entry_raw.get("normalized") or raw_value,
                    "tags": entry_raw.get("tags") or ["Нераспределено"],
                }
                self.cache["problem"][key] = entry
                time.sleep(MIN_LLM_DELAY)
                return entry
            else:
                logger.warning(f"⚠️ LLM не вернула ответ для одиночной проблемы '{raw_value[:40]}...'")
                return {"normalized": raw_value, "tags": ["Нераспределено"]}
        except LLMCallError as e:
            logger.error(f"❌ Ошибка при одиночной категоризации проблемы: {e}")
            return {"normalized": raw_value, "tags": ["Нераспределено"]}

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
                    # ИЗМЕНЕНО: для проблем — отдельный метод (слой 1 + слой 2 с dict-форматом),
                    # для client/address — прежняя логика без изменений
                    if field_type == "problem":
                        self._categorize_and_normalize_problems_batch(values_to_normalize)
                    else:
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
                elif target == "problem_raw":
                    # НОВОЕ: отдельная ветка для проблем — кэш возвращает dict
                    # {"normalized": ..., "tags": [...]}, а не строку
                    record[target] = value.strip() if isinstance(value, str) else value
                    if value and isinstance(value, str):
                        result = self._normalize_with_cache_problem(value)
                        record["problem_normalized"] = result.get("normalized", "")
                        record["problem_tags"] = result.get("tags", [])
                    else:
                        record["problem_normalized"] = ""
                        record["problem_tags"] = []
                elif target in ("client_raw", "address_raw"):
                    # Сохраняем сырое значение
                    record[target] = value.strip() if isinstance(value, str) else value

                    # Определяем тип для нормализации
                    if target == "client_raw":
                        norm_field = "client_normalized"
                        cache_type = "client"
                    elif target == "address_raw":
                        norm_field = "address_normalized"
                        cache_type = "address"
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

           # Дедупликация по ticket_id — ТОЛЬКО среди строк, где он реально
            # заполнен. Строки без ticket_id (None/пустая строка — например,
            # лист без колонки "Номер") дедупликации не подвергаются: без
            # номера мы не можем надёжно отличить дубликат от двух разных
            # обращений, а pandas считает все NaN/None равными друг другу —
            # раньше это могло схлопнуть в одну строку ВСЕ записи без номера
            # тикета по всему датасету сразу. Лучше оставить лишнюю строку,
            # чем случайно стереть настоящую.
            if "ticket_id" in df.columns:
                has_ticket_mask = df["ticket_id"].notna() & (df["ticket_id"].astype(str).str.strip() != "")
                with_ticket = df[has_ticket_mask]
                without_ticket = df[~has_ticket_mask]

                initial_with_ticket = len(with_ticket)
                with_ticket = with_ticket.drop_duplicates(subset=["ticket_id"], keep="first")
                duplicates_removed = initial_with_ticket - len(with_ticket)
                if duplicates_removed > 0:
                    logger.info(f"🗑️ Удалено {duplicates_removed} дубликатов по ticket_id")

                if len(without_ticket) > 0:
                    logger.warning(
                        f"⚠️ {len(without_ticket)} строк без ticket_id — дедупликация для них "
                        f"НЕ применялась (нет надёжного способа отличить дубликат от разных обращений)"
                    )

                df = pd.concat([with_ticket, without_ticket], ignore_index=True)

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
            logger.info(f"  • Слой 1 (без LLM) для проблем: {self.stats['layer1_matches']}")
            logger.info(f"  • Всего записей сохранено: {len(self.all_records)}")
        else:
            logger.warning("⚠️ Нет данных для сохранения")