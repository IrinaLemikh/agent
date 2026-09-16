"""
Универсальный клиент для DeepSeek API.

Все батч-методы (categorize_and_normalize_batch, normalize_client_address_batch)
следуют одному паттерну: dict-по-id + принудительный tool_choice на конкретную
функцию + disable_thinking=True. disable_thinking нужен по двум причинам сразу
(см. документацию DeepSeek): в thinking mode нельзя форсировать конкретный тул
через tool_choice, и temperature молча игнорируется в этом режиме.

При исчерпании ретраев _call_api бросает LLMCallError — вызывающий код
(fetcher.py) обязан поймать её и НЕ кэшировать результат.
"""
import os
import json
import re
import requests
import time
from typing import List, Optional, Dict, Any, Tuple
from config.settings import DEEPSEEK_API_KEY
from loguru import logger
from .prompts import SYSTEM_PROMPTS, TASK_PROMPTS


class LLMCallError(Exception):
    """Ретраи исчерпаны или ответ модели пришёл в неожиданной форме.
    Вызывающий код обязан НЕ кэшировать результат при этой ошибке."""
    pass


# Имя функции-инструмента для категоризации проблем (function calling)
CATEGORIZATION_TOOL_NAME = "submit_categorization"

# Имя функции-инструмента для совместной нормализации пары client+address
# (см. обсуждение в чате: раздельная нормализация не видит, что город/бренд/
# юрлицо могут быть перепутаны между двумя сырыми полями одной строки)
CLIENT_ADDRESS_TOOL_NAME = "submit_client_address"


def _compose_address(city: str, street: str, house: str, korpus: str, apartment: str) -> str:
    """
    Склеивает address_normalized из структурных полей через запятую, в
    фиксированном порядке и НИЧЕГО не дописывая:

        "Екатеринбург, Белинского, 86, 98"
        "Белоярский, мкр. Геологический, 2б"

    Обозначения ("ул.", "д.", "кв.", "пом.") отбрасывает сама модель по
    промпту - кроме случаев, где тип это часть названия ("4 мкрн"). Раньше
    наоборот: модель просили тип убрать, а сборщик приписывал "ул." и "кв."
    обратно, сверяясь со списком известных типов. Список неизбежно оказывался
    неполным, и на живых данных выходили "ул. мкр 5 А" и "кв. помещ 98".
    Теперь приписывать нечего - склейка про типы не знает вообще.

    Корпус клеится к дому через "/" ("30-А/1") - единственный добавляемый
    символ, и он не зависит от того, что написала модель.
    """
    parts = []
    if city:
        parts.append(city)
    if street:
        parts.append(street)
    if house:
        parts.append(f"{house}/{korpus}" if korpus else house)
    if apartment:
        parts.append(apartment)
    return ", ".join(parts)


def _word_key(token: str) -> str:
    """Слово без регистра и пунктуации — ключ для сравнения со словарём."""
    return ''.join(ch for ch in token.lower() if ch.isalnum())


DEFAULT_POINT_BRAND = "Пивко"


def _expand_point_name(
    raw: str,
    aliases: Dict[str, str],
    brands: Optional[Dict[str, str]] = None,
) -> str:
    """
    Приводит название точки к виду "<Бренд> <Квалификатор> <гео и остальное>".

        "Проф Краснодар"      -> "Пивко Проф Краснодар"
        "Москва ПРОФ"         -> "Пивко Проф Москва"
        "Пивко Франч"         -> "Пивко Франшиза"
        "Франч Ротор"         -> "Ротор Франшиза"
        "франч Самара Ротор"  -> "Ротор Франшиза Самара"

    Бренд по умолчанию — Пивко: техподдержка обслуживает разные сети, но
    подавляющее большинство клиентов это Пивко, и операторы не пишут его
    в каждом обращении (ровно как не пишут "Екатеринбург" — офис здесь).
    Перебить умолчание может только имя из списка самостоятельных компаний
    (Glossary.standalone_brands) — Ротор, Пивстанция и т.п. Но если "Пивко"
    написано явно, оно выигрывает: "Франч Пивко // Разливной" остаётся
    Пивко, а Разливной уезжает в хвост.

    Совпадение ищется по ЦЕЛЫМ словам: улица "Профсоюзная" и слово
    "спорт-проф" не превращаются в "Пивко Проф" (обе ловушки есть в данных).

    Город из названия НЕ трогаем и в адрес не переносим: это имя
    региональной территории франшизы, а не место точки — под "Проф Сургут"
    есть точки в Когалыме и Покачах.

    aliases/brands приходят параметрами из fetcher: client.py сам
    glossary.py не импортирует.
    """
    if not raw:
        return ""
    raw = raw.strip()
    if not aliases:
        return raw

    brands = brands or {}
    qualifier = None
    brand = None
    explicit_default_brand = False
    tail: List[str] = []

    for token in raw.split():
        key = _word_key(token)
        if not key:
            continue  # "//", "|" и прочие обрывки разделителей смысла не несут
        if qualifier is None and key in aliases:
            qualifier = aliases[key]
            continue
        if key == _word_key(DEFAULT_POINT_BRAND):
            explicit_default_brand = True
            continue
        if brand is None and key in brands:
            brand = brands[key]
            continue
        tail.append(token)

    if qualifier is None:
        return raw

    if explicit_default_brand:
        # Явное "Пивко" сильнее имени из списка: сам бренд, если он тоже
        # был назван, остаётся уточнением в хвосте
        if brand:
            tail.insert(0, brand)
        brand = DEFAULT_POINT_BRAND

    return " ".join([brand or DEFAULT_POINT_BRAND, qualifier] + tail)


class DeepSeekClient:
    """Универсальный шлюз для всех вызовов LLM."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY не найден")

        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # =========================================================================
    # НОВОЕ: общий низкоуровневый вызов API с ретраями. Используется и ask(),
    # и categorize_and_normalize_batch() — чтобы не дублировать retry-логику.
    # =========================================================================
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.1,
        max_retries: int = 2,
        base_delay: float = 1.0,
        disable_thinking: bool = False,
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к DeepSeek, возвращает распарсенный JSON ответа
        целиком (не текст, а сырой response.json()).

        При исчерпании ретраев или невозможности получить валидный JSON
        бросает LLMCallError — НИКОГДА не возвращает "суррогатный" ответ.

        disable_thinking: если True, добавляет {"thinking": {"type": "disabled"}}
        в payload. НУЖНО для двух вещей сразу (см. документацию DeepSeek):
        1) в thinking mode нельзя принудительно указать конкретную функцию
           через tool_choice — только с отключённым thinking это работает;
        2) в thinking mode параметр temperature молча игнорируется — то есть
           наш temperature=0.0 для категоризации/нормализации реально
           применяется только при disable_thinking=True.
        """
        payload = {
            "model": "deepseek-flash",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 50000,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if disable_thinking:
            payload["thinking"] = {"type": "disabled"}

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=180
                )
                response.raise_for_status()
                result = response.json()

                # Минимальная проверка формы ответа — если её нет, это тоже
                # повод для ретрая, а не для того, чтобы тянуть мусор дальше
                if "choices" not in result or not result["choices"]:
                    raise ValueError(f"Неожиданная форма ответа API: отсутствует 'choices' -> {result}")

                return result

            except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
                last_error = e
                # НОВОЕ: логируем тело ответа API, если оно есть — raise_for_status()
                # сам по себе даёт только код статуса, а не причину от DeepSeek
                # (например, какой именно параметр в tools/tool_choice не понравился)
                error_body = None
                response_obj = getattr(e, "response", None)
                if response_obj is not None:
                    try:
                        error_body = response_obj.text
                    except Exception:
                        error_body = None
                body_suffix = f" | Тело ответа API: {error_body}" if error_body else ""
                logger.warning(f"Попытка {attempt + 1}/{max_retries + 1} не удалась: {e}{body_suffix}")
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info(f"Повтор через {sleep_time:.1f}с...")
                    time.sleep(sleep_time)

        # Ретраи исчерпаны — бросаем ошибку, НЕ возвращаем суррогат
        logger.error(f"❌ Все попытки исчерпаны, вызов LLM провалился: {last_error}")
        raise LLMCallError(f"Не удалось получить ответ от LLM после {max_retries + 1} попыток: {last_error}")

    def ask(self,
            user_prompt: str,
            system_role: Optional[str] = None,
            temperature: float = 0.1,
            max_retries: int = 2,
            base_delay: float = 1.0,
            **kwargs) -> str:
        """
        Универсальный метод для отправки запроса к LLM.

        Args:
            user_prompt: пользовательский промпт (может содержать placeholders)
            system_role: роль для системного промпта (ключ в SYSTEM_PROMPTS)
            temperature: температура
            max_retries: количество повторных попыток при ошибке
            base_delay: базовая задержка перед повторной попыткой (экспоненциально растёт)
            **kwargs: аргументы для форматирования user_prompt

        Returns:
            Ответ LLM (строка)

        Raises:
            LLMCallError: если ретраи исчерпаны или ответ пришёл в неожиданной
            форме. ВАЖНО (изменение поведения): раньше в этом случае метод
            тихо возвращал user_prompt — так делать было небезопасно (см.
            комментарий в шапке файла). Теперь вызывающий код обязан ловить
            LLMCallError и НЕ кэшировать результат.
        """
        # Форматируем промпт, если переданы аргументы
        if kwargs:
            user_prompt = user_prompt.format(**kwargs)

        # Собираем сообщения
        messages = []
        if system_role:
            system_prompt = SYSTEM_PROMPTS.get(system_role)
            if not system_prompt:
                logger.warning(f"Системная роль '{system_role}' не найдена, пропускаем")
            else:
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        logger.debug(f"🦙 Запрос к LLM (role={system_role}): {user_prompt[:100]}...")

        result = self._call_api(
            messages=messages,
            temperature=temperature,
            max_retries=max_retries,
            base_delay=base_delay,
        )

        try:
            answer = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise LLMCallError(f"Не удалось извлечь текст ответа из результата API: {e}")

        logger.debug(f"✅ Ответ получен (длина {len(answer)})")
        return answer

    def _build_client_address_tool_schema(self) -> List[Dict[str, Any]]:
        """
        JSON-схема инструмента для совместной нормализации пары client+address
        через function calling. Каждый пункт результата — не одна строка, а
        объект с тремя независимыми сущностями (client_normalized, point_name,
        структурные адресные поля) — см. TASK_NORMALIZE_CLIENT_ADDRESS.
        Пустая строка = "не нашлось", поле всё равно обязано присутствовать
        (required), чтобы JSON был одной формы и парсинг не падал на
        отсутствующих ключах.
        """
        address_field = {"type": "string"}
        return [
            {
                "type": "function",
                "function": {
                    "name": CLIENT_ADDRESS_TOOL_NAME,
                    "description": (
                        "Верни разбор клиента, названия точки и адреса для каждой "
                        "пары из пронумерованного списка. Ключи в 'results' должны "
                        "в точности совпадать с номерами из списка."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "object",
                                "description": (
                                    "Ключ — номер пункта из списка (строкой). Значение — "
                                    "разбор пары на client_normalized/point_name/адресные поля."
                                ),
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "client_normalized": {"type": "string"},
                                        "point_name": {"type": "string"},
                                        "city": address_field,
                                        "street": address_field,
                                        "house": address_field,
                                        "korpus": address_field,
                                        "apartment": address_field,
                                    },
                                    "required": [
                                        "client_normalized", "point_name",
                                        "city", "street", "house", "korpus", "apartment",
                                    ],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                },
            }
        ]

    def normalize_client_address_batch(
        self,
        pairs: Dict[str, Tuple[str, str]],
        point_name_glossary_text: str = "",
        point_name_aliases: Optional[Dict[str, str]] = None,
        point_name_brands: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Совместная нормализация пар (client_raw, address_raw) одним вызовом —
        заменяет собой раздельные normalize_batch_dict("client"/"address").
        Тот же паттерн dict-по-id + принудительный tool_choice + disable_thinking,
        что и у других батч-методов этого класса.

        Args:
            pairs: {"1": (client_raw, address_raw), ...} — id локальный для
                этого вызова, задаётся вызывающим кодом (fetcher).
            point_name_glossary_text: маленький блок из
                Glossary.to_prompt_text(section="client_address") — контекст
                про известные суб-бренды (например "Пивко Проф"), НЕ весь
                технический глоссарий. Может быть пустой строкой.

        Returns:
            {"1": {"client_normalized": "...", "point_name": "...",
                    "address_normalized": "..."}, ...} — address_normalized уже
            собран из структурных полей детерминированно (_compose_address),
            point_name уже прогнан через алиас-словарь (_expand_point_name).
            Если модель не вернула ответ для какого-то id, этот ключ в
            результате отсутствует (вызывающий код обязан не кэшировать).

        Raises:
            LLMCallError: ретраи исчерпаны, модель не вызвала инструмент,
            или аргументы инструмента не парсятся как JSON.
        """
        if not pairs:
            return {}

        point_name_aliases = point_name_aliases or {}
        point_name_brands = point_name_brands or {}

        task_prompt = TASK_PROMPTS.get("normalize_client_address")
        if not task_prompt:
            raise ValueError("В TASK_PROMPTS отсутствует шаблон 'normalize_client_address'")

        numbered = "\n".join(
            f"{item_id}. КЛИЕНТ: {client or '(пусто)'} | АДРЕС: {address or '(пусто)'}"
            for item_id, (client, address) in pairs.items()
        )
        user_prompt = task_prompt.format(texts=numbered, glossary_text=point_name_glossary_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get("normalizer", "")},
            {"role": "user", "content": user_prompt},
        ]

        tools = self._build_client_address_tool_schema()
        tool_choice = {"type": "function", "function": {"name": CLIENT_ADDRESS_TOOL_NAME}}

        logger.debug(f"🦙 Нормализация клиент+адрес батча из {len(pairs)} пар...")

        result = self._call_api(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            disable_thinking=True,
        )

        try:
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                raise ValueError("Модель не вызвала инструмент submit_client_address")
            arguments_raw = tool_calls[0]["function"]["arguments"]
            arguments = json.loads(arguments_raw)
            results = arguments.get("results")
            if not isinstance(results, dict):
                raise ValueError(f"Поле 'results' отсутствует или имеет неверный тип: {arguments}")
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            raise LLMCallError(f"Не удалось извлечь результат нормализации клиент+адрес из ответа API: {e}")

        cleaned_results: Dict[str, Dict[str, str]] = {}
        for item_id, entry in results.items():
            if not isinstance(entry, dict):
                logger.warning(f"⚠️ Некорректная запись для пункта {item_id}, пропускаем: {entry}")
                continue

            client_normalized = (entry.get("client_normalized") or "").strip()
            point_name = _expand_point_name(
                (entry.get("point_name") or "").strip(),
                point_name_aliases,
                point_name_brands,
            )
            address_normalized = _compose_address(
                city=(entry.get("city") or "").strip(),
                street=(entry.get("street") or "").strip(),
                house=(entry.get("house") or "").strip(),
                korpus=(entry.get("korpus") or "").strip(),
                apartment=(entry.get("apartment") or "").strip(),
            )
            cleaned_results[item_id] = {
                "client_normalized": client_normalized,
                "point_name": point_name,
                "address_normalized": address_normalized,
                # Город отдельным полем — чтобы реконсилер знал факт его
                # наличия, а не угадывал по виду строки (пустое значение = в
                # адресе города действительно нет).
                "city": (entry.get("city") or "").strip(),
            }

        logger.debug(f"✅ Нормализация клиент+адрес завершена, обработано {len(cleaned_results)}/{len(pairs)}")
        return cleaned_results

    # =========================================================================
    # НОВОЕ: слой 2 для проблем — нормализация + категоризация через
    # function calling с динамическим enum по категориям.
    # =========================================================================
    def _build_categorization_tool_schema(self, category_names: List[str]) -> List[Dict[str, Any]]:
        """
        Строит JSON-схему инструмента для function calling. Тег обязан быть
        одним из category_names — модель физически не может вернуть
        значение вне списка (в отличие от текстового промпта).
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": CATEGORIZATION_TOOL_NAME,
                    "description": (
                        "Верни нормализованную формулировку и категории (теги) "
                        "для каждого обращения из пронумерованного списка. "
                        "Ключи в 'results' должны в точности совпадать с номерами "
                        "из списка обращений."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "object",
                                "description": (
                                    "Ключ — номер пункта из списка обращений (строкой, "
                                    "например '1', '2'). Значение — результат для этого пункта."
                                ),
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "normalized": {
                                            "type": "string",
                                            "description": "Краткая нормализованная формулировка проблемы (5-7 слов)"
                                        },
                                        "tags": {
                                            "type": "array",
                                            "description": "Одна-три категории, применимые к обращению",
                                            "items": {
                                                "type": "string",
                                                "enum": category_names
                                            },
                                            "minItems": 1,
                                            "maxItems": 3
                                        }
                                    },
                                    "required": ["normalized", "tags"]
                                }
                            }
                        },
                        "required": ["results"]
                    }
                }
            }
        ]

    def categorize_and_normalize_batch(
        self,
        items: Dict[str, str],
        category_names: List[str],
        categories_text: str,
        glossary_text: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Нормализует формулировку и присваивает категории пачке сырых описаний
        проблем за один вызов LLM.

        Args:
            items: {"1": "сырой текст 1", "2": "сырой текст 2", ...} — id
                локальный для этого вызова, задаётся вызывающим кодом (fetcher).
            category_names: список допустимых названий категорий (для enum
                в function calling — берётся из CategoriesManager.get_category_names()).
            categories_text: развёрнутый текст категорий с пояснениями/hint
                (CategoriesManager.to_prompt_text()) — для описания в промпте,
                помогает модели дизамбигуировать, а не для enum напрямую.
            glossary_text: словарь сленга целиком текстом (Glossary.to_prompt_text()).

        Returns:
            {"1": {"normalized": "...", "tags": ["..."]}, ...} — по тем же
            ключам, что в items. Если модель не вернула ответ для какого-то
            id, этот ключ в результате отсутствует (вызывающий код должен
            сам решить, что делать — см. fetcher._categorize_and_normalize_problems_batch,
            где такие пропуски логируются и НЕ кэшируются).

        Raises:
            LLMCallError: ретраи исчерпаны, модель не вызвала инструмент,
            или аргументы инструмента не парсятся как JSON. Вызывающий код
            обязан поймать эту ошибку и не кэшировать результат для всего
            батча.
        """
        if not items:
            return {}

        task_prompt = TASK_PROMPTS.get("categorize_problem")
        if not task_prompt:
            raise ValueError("В TASK_PROMPTS отсутствует шаблон 'categorize_problem'")

        numbered = "\n".join(f"{item_id}. {text}" for item_id, text in items.items())
        user_prompt = task_prompt.format(
            items=numbered,
            categories_text=categories_text,
            glossary_text=glossary_text,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get("categorizer", "")},
            {"role": "user", "content": user_prompt},
        ]

        tools = self._build_categorization_tool_schema(category_names)
        tool_choice = {"type": "function", "function": {"name": CATEGORIZATION_TOOL_NAME}}

        logger.debug(f"🦙 Категоризация батча из {len(items)} проблем...")

        result = self._call_api(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            disable_thinking=True,
        )

        try:
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                raise ValueError("Модель не вызвала инструмент submit_categorization")
            arguments_raw = tool_calls[0]["function"]["arguments"]
            arguments = json.loads(arguments_raw)
            results = arguments.get("results")
            if not isinstance(results, dict):
                raise ValueError(f"Поле 'results' отсутствует или имеет неверный тип: {arguments}")
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            raise LLMCallError(f"Не удалось извлечь результат категоризации из ответа API: {e}")

        # Лёгкая валидация тегов на случай, если модель всё же вернула
        # значение вне enum (не все API одинаково строго валидируют enum)
        valid_categories = set(category_names)
        cleaned_results: Dict[str, Dict[str, Any]] = {}
        for item_id, entry in results.items():
            if not isinstance(entry, dict):
                logger.warning(f"⚠️ Некорректная запись для пункта {item_id}, пропускаем: {entry}")
                continue

            normalized = entry.get("normalized") or items.get(item_id, "")
            raw_tags = entry.get("tags") or []
            tags = [t for t in raw_tags if t in valid_categories]

            invalid_tags = [t for t in raw_tags if t not in valid_categories]
            if invalid_tags:
                logger.warning(f"⚠️ Пункт {item_id}: модель вернула категории вне списка {invalid_tags}, отброшены")

            if not tags:
                tags = ["Нераспределено"]

            cleaned_results[item_id] = {"normalized": normalized, "tags": tags}

        logger.debug(f"✅ Категоризация батча завершена, обработано {len(cleaned_results)}/{len(items)}")
        return cleaned_results