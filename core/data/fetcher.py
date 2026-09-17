"""
Модуль для загрузки и нормализации данных из всех доступных Google Sheets таблиц.
Читает каждую таблицу, каждый лист, нормализует данные, сохраняет в Parquet и обновляет индекс.

=============================================================================
ИНТЕРФЕЙСЫ ДРУГИХ ФАЙЛОВ (актуально)
=============================================================================
1) core/data/categories.py -> CategoriesManager: get_category_names(),
   to_prompt_text() — слой 1 (fuzzy) и слой 2 (LLM) для проблем.

2) core/data/glossary.py -> Glossary: to_prompt_text(section="problem"|
   "client_address"), alias_map(section="client_address") — квалификаторы
   направлений ("франч" -> "Франшиза"), standalone_brands(section=
   "client_address") — компании, которые НЕ Пивко (Ротор, Пивстанция).
   Раздел "client_address" содержит только контекст про названия точек,
   НЕ весь технический словарь.

3) core/llm/client.py -> DeepSeekClient:
   - categorize_and_normalize_batch(items, category_names, categories_text,
     glossary_text) -> {"1": {"normalized": ..., "tags": [...]}, ...}
   - normalize_client_address_batch(pairs, point_name_glossary_text,
     point_name_aliases) -> {"1": {"client_normalized", "point_name",
     "address_normalized"}, ...} — совместная нормализация пары
     (client_raw, address_raw) одним вызовом (см. core/data/reconciler.py
     и обсуждение в чате: раздельная нормализация не видит, что город/
     бренд/юрлицо перепутаны между двумя полями одной строки).
   Оба батчевых метода при исчерпании ретраев бросают LLMCallError —
   fetcher.py обязан поймать её и НЕ кэшировать результат.
=============================================================================
"""

import sys
import time
import json
import re
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Set
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
from core.data.reconciler import reconcile
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
    "client_normalized",         # юрлицо/ФИО или пусто (никогда бренд — см. point_name)
    "address_raw",               # исходный адрес
    "address_normalized",        # только география, город без дефолта (см. reconciler)
    "address_city",              # НОВОЕ: город отдельным полем. Пусто = города в
                                 # адресе НЕТ (факт, а не догадка) — реконсилер
                                 # по нему подтягивает город и ставит дефолт.
    "point_name",                 # НОВОЕ: коммерческое название точки (бренд/франшиза).
                                 # ВАЖНО: НЕ источник географии. Город внутри него —
                                 # имя территории франшизы, а не место точки:
                                 # "Проф Краснодар" включает точки в Яблоновском.
                                 # Фильтровать по городу можно только по address_city.
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

# Батч client+address (пары) и параллельность батчей. MAX_WORKERS скромный
# осознанно — лимит API (2500 конкурентных соединений) далеко не узкое место,
# число ограничено, чтобы не рисковать качеством/не гнаться за скоростью
# сильнее, чем нужно (см. обсуждение в чате).
CLIENT_ADDRESS_BATCH_SIZE = 100
CLIENT_ADDRESS_MAX_WORKERS = 5

# Сколько строк обрабатываем между проверками на прерывание в ЭТАПЕ 3
CANCEL_CHECK_EVERY_ROWS = 500


class ProcessingCancelled(Exception):
    """
    Прогон прерван снаружи — например, пользователь обновил страницу и
    сессия Streamlit, запустившая обработку, закрылась.

    Прерывание кооперативное: Fetcher сам зовёт cancel_check() в заранее
    размеченных местах (между листами, между батчами, раз в
    CANCEL_CHECK_EVERY_ROWS строк), потому что убить поток снаружи Python
    не умеет, а внутри fetch_all нет ни одного вызова st.*, на котором
    Streamlit мог бы остановить скрипт сам.

    Прерывать в этих точках безопасно: кэш нормализации сохраняется после
    каждого батча, а parquet перезаписывается только в самом конце
    fetch_all. Прерванный прогон теряет время, но не данные — следующий
    запуск подхватит кэш и продолжит почти с того же места.
    """


class Fetcher:
    """
    Класс для загрузки и нормализации всех данных из Google Sheets.
    """
    def __init__(self, cancel_check=None):
        # cancel_check — функция без аргументов, бросающая ProcessingCancelled,
        # если продолжать больше не нужно. Собирается на стороне UI
        # (см. core/main.py), чтобы fetcher ничего не знал про Streamlit.
        # По умолчанию — заглушка: консольный запуск ничем не прерывается.
        self._cancel_check = cancel_check if callable(cancel_check) else (lambda: None)
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
        self._glossary_prompt_text = self.glossary.to_prompt_text()  # section="problem" по умолчанию
        self._categories_prompt_text = self.categories.to_prompt_text()
        self._category_names = self.categories.get_category_names()

        # НОВОЕ: отдельный маленький блок глоссария (суб-бренды) для батча
        # client+address — не весь технический словарь, см. Glossary.to_prompt_text
        self._point_name_glossary_text = self.glossary.to_prompt_text(section="client_address")
        self._point_name_aliases = self.glossary.alias_map(section="client_address")
        self._point_name_brands = self.glossary.standalone_brands(section="client_address")

        # =====================================================================
        # НОВОЕ: флаг очистки кэша (переменная окружения CLEAR_CACHE=true)
        # =====================================================================
        if os.getenv("CLEAR_CACHE", "false").lower() == "true":
            self._clear_cache()

    def _clear_cache(self):
        """Полностью очищает кэш (удаляет JSON-файлы)."""
        cache_dir = Path("/app/cache")
        for key in ["client_address", "problem"]:
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
        - client_address (НОВОЕ, ключ f"{client_raw}||{address_raw}".lower()):
          {"client_normalized": "...", "point_name": "...", "address_normalized": "..."}
        - problem: {raw_lower: {"normalized": "...", "tags": [...]}}
        Сам метод загрузки/сохранения не зависит от формата значения — json
        одинаково хранит и строки, и словари, поэтому логика ниже не меняется.
        """
        cache = {"client_address": {}, "problem": {}}
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
    # ИЗМЕНЕНО: собираем уникальные ПАРЫ (client_raw, address_raw) вместо двух
    # раздельных множеств значений — см. обсуждение в чате: раздельная
    # нормализация не видит, что город/бренд/юрлицо перепутаны между полями
    # одной строки. problem_raw остаётся отдельным множеством, как и раньше.
    # =========================================================================
    def _collect_unique_pairs(self, rows: List[List[str]], mapping: Dict[str, int]) -> Tuple[Set[Tuple[str, str]], set]:
        """
        Возвращает (pairs, problem_values):
          pairs — set уникальных (client_raw, address_raw); пара пропускается,
            только если ОБА поля пусты (одно пустое — нормальный случай,
            обрабатывается промптом как "(пусто)").
          problem_values — set уникальных problem_raw, без изменений.
        """
        pairs: Set[Tuple[str, str]] = set()
        problem_values: set = set()

        client_idx = mapping.get("client_raw")
        address_idx = mapping.get("address_raw")
        problem_idx = mapping.get("problem_raw")

        for row in tqdm(rows, desc="📊 Сбор уникальных пар", leave=False):
            if client_idx is not None or address_idx is not None:
                client_val = str(row[client_idx]).strip() if client_idx is not None and client_idx < len(row) and row[client_idx] else ""
                address_val = str(row[address_idx]).strip() if address_idx is not None and address_idx < len(row) and row[address_idx] else ""
                if client_val or address_val:
                    pairs.add((client_val, address_val))

            if problem_idx is not None and problem_idx < len(row) and row[problem_idx]:
                value = str(row[problem_idx]).strip()
                if value:
                    problem_values.add(value)

        return pairs, problem_values

    # =========================================================================
    # НОВОЕ: совместная нормализация client+address по парам, параллельно по
    # батчам (см. обсуждение в чате про потокобезопасность кэша): каждый
    # воркер обрабатывает СВОЙ батч и возвращает {cache_key: результат} —
    # ключ уже переведён из локального id ответа LLM в сам факт пары, так что
    # результаты разных батчей не могут пересечься по ключу иначе как для
    # буквально одинаковой пары. Запись в self.cache происходит только в
    # главном потоке, последовательно, после завершения каждого future.
    # =========================================================================
    def _normalize_client_address_worker(self, batch: List[Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
        """Обрабатывает один батч пар в отдельном потоке. Кэш не трогает."""
        id_to_pair = {str(i + 1): pair for i, pair in enumerate(batch)}

        response = self.llm_client.normalize_client_address_batch(
            pairs=id_to_pair,
            point_name_glossary_text=self._point_name_glossary_text,
            point_name_aliases=self._point_name_aliases,
            point_name_brands=self._point_name_brands,
        )

        remapped: Dict[str, Dict[str, str]] = {}
        for item_id, result in response.items():
            pair = id_to_pair.get(item_id)
            if pair is None:
                # Модель вернула id, которого не было в запросе — защита от
                # реального (пусть и редкого) риска, не должно случаться,
                # но лучше пропустить с логом, чем упасть или перепутать пару.
                logger.warning(f"⚠️ Модель вернула неожиданный id '{item_id}', не входивший в батч — пропускаем")
                continue
            client_raw, address_raw = pair
            cache_key = f"{client_raw}||{address_raw}".lower()
            remapped[cache_key] = result

        return remapped

    def _normalize_client_address_pairs(self, pairs: Set[Tuple[str, str]]):
        """
        Нормализует пачку уникальных пар (client_raw, address_raw) —
        батчами по CLIENT_ADDRESS_BATCH_SIZE, параллельно по
        CLIENT_ADDRESS_MAX_WORKERS воркерам. Результат сразу пишется в
        self.cache["client_address"] — вызывающий код читает из кэша, а не
        из возврата этого метода (тот же паттерн, что был у _normalize_batch).
        """
        if not pairs:
            return

        logger.info(f"🦙 Начинаем нормализацию {len(pairs)} уникальных пар client+address")

        sorted_pairs = sorted(pairs)
        batch_size = CLIENT_ADDRESS_BATCH_SIZE
        batches = [sorted_pairs[i:i + batch_size] for i in range(0, len(sorted_pairs), batch_size)]

        with ThreadPoolExecutor(max_workers=CLIENT_ADDRESS_MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._normalize_client_address_worker, batch): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_idx):
                # Точка прерывания: батчи здесь самые долгие, и продолжать
                # их, когда результата уже никто не ждёт, дороже всего
                try:
                    self._cancel_check()
                except ProcessingCancelled:
                    logger.warning("⏹️ Обработка прервана — отменяю оставшиеся батчи client+address")
                    for pending in future_to_idx:
                        pending.cancel()
                    raise

                idx = future_to_idx[future]
                try:
                    remapped = future.result()
                except LLMCallError as e:
                    logger.error(f"❌ Ошибка при нормализации батча client+address #{idx+1}: {e}")
                    # Ничего не кэшируем — батч останется "трудным остатком"
                    # и будет повторно обработан при следующем запуске.
                    continue

                self.stats["llm_calls"] += 1
                self.cache["client_address"].update(remapped)
                logger.debug(f"    ✅ батч #{idx+1}/{len(batches)}: обработано {len(remapped)} пар")
                self._save_cache()

        logger.info(f"✅ Нормализация client+address завершена. Вызовов LLM: {self.stats['llm_calls']}")

    # =========================================================================
    # НОВОЕ: Слой 1 — детерминированный fuzzy-матч сырого текста с названиями
    # категорий, без обращения к LLM.
    # =========================================================================
    def _match_category_by_fuzzy(self, problem_raw: str) -> Optional[str]:
        """
        Сравнивает сырой текст обращения с названиями категорий из categories.json.
        Использует token_set_ratio (в отличие от token_sort_ratio не штрафует за
        разницу в длине строк — сырой текст почти всегда длиннее короткого имени
        категории, а совпадение по общим токенам всё равно означает точный матч,
        см. обсуждение в чате: "консультация по ремонту пк" против категории
        "консультация" раньше давал ~60, теперь 100), порог CATEGORY_FUZZY_THRESHOLD = 90.

        ВАЖНО (известный компромисс, см. обсуждение в чате): на 90 короткие
        акронимные категории вроде "ККТ"/"УТМ" иногда ложно перетягивают к себе
        похожие по буквам, но семантически разные обращения (например, "ФМУ:
        некорректная работа" может уйти в "УТМ"). Порог 95 эту проблему решает,
        но при этом режет часть настоящих совпадений там, где в сыром тексте не
        хватает одного-двух слов из длинного названия категории (например,
        "Прогрузка весов шаблоном этикетки" не дотягивает до полного названия
        "Весы: прогрузка шаблоном этикетки, редактирование этикетки (Пропал PRO
        шаблон)"). Осознанно оставлено 90 — нужна проверка на реальных данных
        перед повторным поднятием порога.

        Возвращает название категории при уверенном совпадении, иначе None.

        БАГ-ФИКС (найден на реальных данных, см. обсуждение в чате): при
        строгом "score > best_score" победитель при ничьей — первая по
        порядку в categories.json категория, а не более специфичная. Реальный
        случай: "Сканер: некорректная работа" и "Сканер: некорректная работа
        (повторные обращения)" — сырой текст, дословно совпадающий со ВТОРЫМ
        (более специфичным) названием, давал token_set_ratio=100 против ОБОИХ
        одновременно (второе название полностью содержит токены первого), и
        побеждала короткая категория просто потому что шла раньше в списке.
        Теперь при равном счёте побеждает БОЛЕЕ ДЛИННОЕ название — оно как
        правило более специфичное (например, содержит уточняющую пометку в
        скобках), а не более общее.
        """
        if not problem_raw or not isinstance(problem_raw, str):
            return None

        best_match = None
        best_score = 0
        for category_name in self._category_names:
            score = fuzz.token_set_ratio(problem_raw, category_name)
            if score > best_score or (score == best_score and best_match and len(category_name) > len(best_match)):
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
            self._cancel_check()  # точка прерывания между батчами проблем

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
    # ИЗМЕНЕНО: быстрая нормализация с кэшем — теперь по ПАРЕ (client_raw,
    # address_raw), возвращает сразу все три поля одним словарём.
    # =========================================================================
    def _normalize_pair_with_cache(self, client_raw: str, address_raw: str) -> Dict[str, str]:
        """
        Быстрая нормализация пары с использованием только кэша (без вызова
        LLM). Возвращает {"client_normalized", "point_name",
        "address_normalized", "city"}.
        Если пары нет в кэше (редкий случай, например кэш почистили в
        процессе работы) — делает одиночный вызов (батч из одной пары).
        """
        empty = {"client_normalized": "", "point_name": "", "address_normalized": "", "city": ""}
        if not client_raw and not address_raw:
            return empty

        key = f"{client_raw}||{address_raw}".lower()
        cached = self.cache["client_address"].get(key)

        if cached:
            self.stats["cache_hits"] += 1
            return cached
        else:
            # Маловероятно, но на всякий случай делаем одиночный вызов.
            self.stats["llm_calls"] += 1
            logger.warning(f"⚠️ Пара '{client_raw[:30]}...' | '{address_raw[:30]}...' не найдена в кэше, вызываем LLM на лету")
            try:
                response = self.llm_client.normalize_client_address_batch(
                    pairs={"1": (client_raw, address_raw)},
                    point_name_glossary_text=self._point_name_glossary_text,
                    point_name_aliases=self._point_name_aliases,
                    point_name_brands=self._point_name_brands,
                )
                result = response.get("1", empty)
                self.cache["client_address"][key] = result
                return result
            except LLMCallError as e:
                # Не кэшируем и не роняем обработку всего листа из-за одной
                # пары — строка получит пустые нормализованные поля, а на
                # следующем прогоне (пары нет в кэше) попытка повторится.
                logger.error(f"❌ Ошибка при одиночной нормализации пары: {e}")
                return empty

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
        # ЭТАП 1: собираем уникальные пары (client_raw, address_raw) + problem_raw
        # =====================================================================
        logger.info(f"🔍 Сбор уникальных пар для {sheet_name}...")
        unique_pairs, unique_problems = self._collect_unique_pairs(data_rows, mapping)

        # =====================================================================
        # ЭТАП 2: нормализуем пачками (client+address параллельно, problem — слой 1+2)
        # =====================================================================
        pairs_to_normalize = {p for p in unique_pairs if f"{p[0]}||{p[1]}".lower() not in self.cache["client_address"]}
        if pairs_to_normalize:
            logger.info(f"📦 Нужно нормализовать {len(pairs_to_normalize)} новых пар client+address")
            self._normalize_client_address_pairs(pairs_to_normalize)
        else:
            logger.info("✅ Все пары client+address уже есть в кэше")

        problems_to_normalize = {v for v in unique_problems if v.lower() not in self.cache["problem"]}
        if problems_to_normalize:
            logger.info(f"📦 Нужно нормализовать {len(problems_to_normalize)} новых проблем")
            self._categorize_and_normalize_problems_batch(problems_to_normalize)
        else:
            logger.info("✅ Все проблемы уже есть в кэше")

        # =====================================================================
        # ЭТАП 3: обрабатываем строки с использованием кэша
        # =====================================================================
        sheet_records = []
        rows_processed = 0
        rows_skipped = 0

        for i, row in enumerate(tqdm(data_rows, desc=f"📝 Обработка строк {sheet_name}", leave=False)):
            # Точка прерывания: сами строки быстрые (всё уже в кэше), но
            # на большом листе цикл всё равно идёт заметное время
            if i % CANCEL_CHECK_EVERY_ROWS == 0:
                self._cancel_check()

            if not row or all(not cell for cell in row):
                rows_skipped += 1
                continue

            record = {col: None for col in TARGET_COLUMNS}
            record["_table_name"] = table_name
            record["_sheet_name"] = sheet_name

            # НОВОЕ: client_raw и address_raw обрабатываются СОВМЕСТНО, ДО
            # общего цикла по mapping — нормализация идёт по паре одним
            # поиском в кэше, а не двумя независимыми (см. обсуждение в чате).
            client_idx = mapping.get("client_raw")
            address_idx = mapping.get("address_raw")
            client_val = str(row[client_idx]).strip() if client_idx is not None and client_idx < len(row) and row[client_idx] else ""
            address_val = str(row[address_idx]).strip() if address_idx is not None and address_idx < len(row) and row[address_idx] else ""
            record["client_raw"] = client_val
            record["address_raw"] = address_val
            if client_val or address_val:
                pair_result = self._normalize_pair_with_cache(client_val, address_val)
                record["client_normalized"] = pair_result.get("client_normalized", "")
                record["point_name"] = pair_result.get("point_name", "")
                record["address_normalized"] = pair_result.get("address_normalized", "")
                record["address_city"] = pair_result.get("city", "")
            else:
                record["client_normalized"] = ""
                record["point_name"] = ""
                record["address_normalized"] = ""
                record["address_city"] = ""

            for target, idx in mapping.items():
                if target in ("client_raw", "address_raw"):
                    continue  # уже обработано выше
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
                else:
                    # status, event_group, assignee, author
                    record[target] = value.strip() if isinstance(value, str) else value

            # point_key больше НЕ считается здесь построчно — построчный расчёт
            # не видел других строк с тем же клиентом/адресом, из-за чего разные
            # написания одной и той же точки (см. обсуждение в чате: "Иванов ИИ"
            # и "ИП Иванов Иван Иванович" на одном адресе) получали разные
            # point_key. Теперь это единственный векторизованный проход по всему
            # df в reconciler.recompute_point_key(), вызывается из fetch_all()
            # после реконсиляции client_normalized/address_normalized.

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

                    # Точка прерывания перед каждым листом — самый дешёвый
                    # момент выйти, ещё до похода в Google Sheets API
                    self._cancel_check()

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
                    except ProcessingCancelled:
                        # ОБЯЗАТЕЛЬНО выше общего except Exception, иначе
                        # прерывание будет проглочено как "ошибка листа"
                        # и цикл спокойно поедет на следующий лист
                        raise
                    except HttpError as e:
                        logger.error(f"❌ Ошибка при чтении листа {table_name}/{sheet_name}: {e}")
                    except Exception as e:
                        logger.error(f"❌ Неожиданная ошибка при обработке {table_name}/{sheet_name}: {e}")

                    pbar.update(1)

        # Сохраняем все записи в Parquet
        if self.all_records:
            df = pd.DataFrame(self.all_records)

            # Реконсиляция: сведение вариантов написания одного клиента/адреса
            # через общий якорь (адрес для клиента, клиент для адреса) и
            # пересчёт point_key одним проходом по всему df — см.
            # core/data/reconciler.py. Кэш нормализации (self.cache) этим не
            # затрагивается, реконсиляция работает только над DataFrame и
            # пересчитывается заново при каждом прогоне.
            df, recon_report = reconcile(df)
            logger.info(
                f"🔗 Реконсиляция: слито групп клиентов — {len(recon_report['client_merges'])}, "
                f"подтянуто городов — {len(recon_report['address_backfills'])}, "
                f"сведено написаний адреса — {recon_report['canonical_stats']['addresses']}"
            )

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