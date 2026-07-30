"""
Универсальный клиент для DeepSeek API.

=============================================================================
ЧТО ИЗМЕНИЛОСЬ И ПОЧЕМУ
=============================================================================
1) БАГ-ФИКС: раньше ask() при исчерпании ретраев ТИХО возвращал user_prompt
   (сам промпт) как будто это ответ модели — "чтобы не ломать поток". Из-за
   этого normalize_batch() не попадал в except-ветку (ошибки формально не
   было), парсил промпт как будто это нормальный ответ, и мусорный результат
   уходил в кэш НАВСЕГДА. Теперь при исчерпании ретраев бросается исключение
   LLMCallError — вызывающий код (fetcher.py) обязан его поймать и НЕ
   кэшировать такие значения. fetcher.py уже ловит широкий Exception в
   нужных местах, так что менять его прямо сейчас не обязательно.

2) РЕФАКТОРИНГ: общая retry-логика вынесена в приватный _call_api(), чтобы
   не дублировать её между ask() и новым categorize_and_normalize_batch().

3) НОВОЕ: categorize_and_normalize_batch() — метод слоя 2 для проблем.
   Категории передаются через function calling с динамическим enum
   (модель физически не может вернуть тег вне списка категорий), словарь и
   пояснения категорий — обычным текстом в промпте. Ответ — dict по номеру
   пункта (id из items), а не позиционный список.

4) НОВОЕ: normalize_batch_dict() для client/address — тот же паттерн
   dict-по-id + function calling, что и у categorize_and_normalize_batch(),
   только без enum (просто произвольная нормализованная строка на id).
   Заменяет собой старый normalize_batch() (построчный парсинг ответа +
   позиционный zip() с исходным списком) — легаси-метод удалён полностью,
   не оставлен даже закомментированным (решение принято в чате: сносим).
   normalize() (одиночная нормализация "на лету", когда значения нет в
   кэше) переписан как самостоятельный метод — больше не вызывает
   normalize_batch внутри себя, т.к. того метода больше не существует.

5) БАГ-ФИКС ПО ИТОГАМ СМОУК-ТЕСТА: модель по умолчанию работает в
   "thinking mode" (reasoning), а эта модель у DeepSeek запрещает
   принудительный tool_choice (400: "Thinking mode does not support this
   tool_choice"). Финальное решение — не переход на tool_choice="auto"
   (промежуточный вариант), а явное отключение thinking через
   {"thinking": {"type": "disabled"}} в payload (параметр disable_thinking
   у _call_api, включён для normalize_batch_dict и
   categorize_and_normalize_batch). Это чинит сразу два эффекта:
   (а) снова разрешает форсировать конкретную функцию через tool_choice —
       надёжнее, чем "auto", модель гарантированно её вызовет;
   (б) возвращает реальный эффект от temperature=0.0 — согласно
       документации DeepSeek, thinking mode молча ИГНОРИРУЕТ temperature
       (не бросает ошибку, просто не применяет), так что раньше наша
       "детерминированная" категоризация температуру фактически не имела.

=============================================================================
ОЖИДАЕМЫЕ КЛЮЧИ В prompts.py (файл сам ещё не переписан, это контракт для
следующего шага)
=============================================================================
- SYSTEM_PROMPTS["normalizer"]   — уже существует, не меняется
- SYSTEM_PROMPTS["categorizer"]  — НОВОЕ, роль для категоризации проблем
- TASK_PROMPTS["normalize_client"] / ["normalize_address"] — уже существуют
- TASK_PROMPTS["categorize_problem"] — НОВОЕ, шаблон с плейсхолдерами
      {items}            — нумерованный список сырых обращений
      {categories_text}  — текст категорий с пояснениями (CategoriesManager.to_prompt_text())
      {glossary_text}    — текст словаря целиком (Glossary.to_prompt_text())
  Сам шаблон не обязан включать инструкцию по формату ответа — формат задаётся
  через function calling (модель обязана вызвать инструмент), а не через
  просьбу "ответь в формате JSON" в тексте.
=============================================================================
"""
import os
import json
import requests
import time
from typing import List, Optional, Dict, Any
from config.settings import DEEPSEEK_API_KEY
from loguru import logger
from .prompts import SYSTEM_PROMPTS, TASK_PROMPTS


class LLMCallError(Exception):
    """Ретраи исчерпаны или ответ модели пришёл в неожиданной форме.
    Вызывающий код обязан НЕ кэшировать результат при этой ошибке."""
    pass


# Имя функции-инструмента для категоризации проблем (function calling)
CATEGORIZATION_TOOL_NAME = "submit_categorization"

# Имя функции-инструмента для нормализации client/address (function calling)
NORMALIZATION_TOOL_NAME = "submit_normalization"


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
            "model": "deepseek-v4-flash",
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
                    timeout=120
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

    def _build_normalization_tool_schema(self) -> List[Dict[str, Any]]:
        """
        JSON-схема инструмента для нормализации client/address через
        function calling. В отличие от категоризации, здесь нет enum —
        просто произвольная нормализованная строка на каждый id.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": NORMALIZATION_TOOL_NAME,
                    "description": (
                        "Верни нормализованную формулировку для каждого значения "
                        "из пронумерованного списка. Ключи в 'results' должны в "
                        "точности совпадать с номерами из списка."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "object",
                                "description": (
                                    "Ключ — номер пункта из списка (строкой, например "
                                    "'1', '2'). Значение — нормализованная строка для этого пункта."
                                ),
                                "additionalProperties": {"type": "string"}
                            }
                        },
                        "required": ["results"]
                    }
                }
            }
        ]

    def normalize_batch_dict(self, items: Dict[str, str], field_type: str) -> Dict[str, str]:
        """
        Нормализует пачку значений одного поля (client / address) через
        function calling, dict-по-id — тот же паттерн, что и
        categorize_and_normalize_batch(). Заменяет собой прежний
        normalize_batch() с построчным парсингом и позиционным zip()
        (тот метод удалён — легаси не осталось, см. решение в чате).

        Args:
            items: {"1": "сырой текст 1", "2": "сырой текст 2", ...} — id
                локальный для этого вызова, задаётся вызывающим кодом (fetcher).
            field_type: 'address' или 'client'
                (для 'problem' используется categorize_and_normalize_batch —
                там нужна ещё и категоризация, а не только нормализация).

        Returns:
            {"1": "нормализованная строка", ...} — по тем же ключам, что в
            items. Если модель не вернула ответ для какого-то id, этот ключ
            в результате отсутствует (вызывающий код — fetcher._normalize_batch —
            логирует это и НЕ кэширует значение).

        Raises:
            LLMCallError: ретраи исчерпаны, модель не вызвала инструмент,
            или аргументы инструмента не парсятся как JSON. Вызывающий код
            обязан поймать эту ошибку и не кэшировать результат для всего
            батча.
        """
        if not items:
            return {}

        task_name = f"normalize_{field_type}"
        task_prompt = TASK_PROMPTS.get(task_name)
        if not task_prompt:
            raise ValueError(f"Неизвестный тип поля '{field_type}'. Допустимые: address, client")

        numbered = "\n".join(f"{item_id}. {text}" for item_id, text in items.items())
        user_prompt = task_prompt.format(texts=numbered)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get("normalizer", "")},
            {"role": "user", "content": user_prompt},
        ]

        tools = self._build_normalization_tool_schema()
        # ИСПРАВЛЕНО по итогам смоук-теста: изначальная проблема была не в
        # tool_choice как таковом, а в том, что модель по умолчанию работает
        # в thinking mode, а она запрещает принудительный выбор функции.
        # Отключаем thinking (disable_thinking=True ниже) — это же заодно
        # чинит и temperature=0.0, которая в thinking mode молча
        # игнорировалась (см. документацию DeepSeek). С отключённым thinking
        # можно снова форсировать конкретную функцию — это надёжнее "auto",
        # модель гарантированно её вызовет.
        tool_choice = {"type": "function", "function": {"name": NORMALIZATION_TOOL_NAME}}

        logger.debug(f"🦙 Нормализация батча из {len(items)} значений ('{field_type}')...")

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
                raise ValueError("Модель не вызвала инструмент submit_normalization")
            arguments_raw = tool_calls[0]["function"]["arguments"]
            arguments = json.loads(arguments_raw)
            results = arguments.get("results")
            if not isinstance(results, dict):
                raise ValueError(f"Поле 'results' отсутствует или имеет неверный тип: {arguments}")
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            raise LLMCallError(f"Не удалось извлечь результат нормализации из ответа API: {e}")

        cleaned_results: Dict[str, str] = {}
        for item_id, value in results.items():
            if isinstance(value, str) and value.strip():
                cleaned_results[item_id] = value.strip()
            else:
                logger.warning(f"⚠️ Некорректное значение для пункта {item_id}, пропускаем: {value!r}")

        logger.debug(f"✅ Нормализация батча завершена, обработано {len(cleaned_results)}/{len(items)}")
        return cleaned_results

    def normalize(self, text: str, field_type: str) -> str:
        """
        Одиночная нормализация одного значения — самостоятельный метод, НЕ
        зависит от normalize_batch_dict (используется как редкий
        live-фоллбэк в fetcher._normalize_with_cache, когда значения не
        нашлось в кэше). Обычный текстовый промпт без function calling —
        для одного значения проблема потери позиции неактуальна.

        Raises:
            LLMCallError: пробрасывается из ask() при исчерпании ретраев —
            вызывающий код обязан поймать её и не кэшировать результат.
        """
        task_name = f"normalize_{field_type}"
        task_prompt = TASK_PROMPTS.get(task_name)
        if not task_prompt:
            raise ValueError(f"Неизвестный тип поля '{field_type}'. Допустимые: address, client")

        user_prompt = task_prompt.format(texts=f"1. {text}")

        response = self.ask(
            user_prompt=user_prompt,
            system_role="normalizer",
            temperature=0.0
        )

        # Одна строка ответа — убираем возможную нумерацию модели
        first_line = response.strip().split("\n")[0]
        cleaned = first_line.lstrip("0123456789. ").strip()
        return cleaned if cleaned else text

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
        # См. подробный комментарий в normalize_batch_dict — отключаем
        # thinking (disable_thinking=True ниже), после чего принудительный
        # tool_choice снова работает и temperature=0.0 реально применяется.
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