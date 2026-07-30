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
        tool_choice: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        max_retries: int = 2,
        base_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к DeepSeek, возвращает распарсенный JSON ответа
        целиком (не текст, а сырой response.json()).

        При исчерпании ретраев или невозможности получить валидный JSON
        бросает LLMCallError — НИКОГДА не возвращает "суррогатный" ответ.
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
                logger.warning(f"Попытка {attempt + 1}/{max_retries + 1} не удалась: {e}")
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

    def normalize_batch(self, texts: List[str], field_type: str) -> List[str]:
        """
        Нормализует пачку значений одного поля (client / address).

        Args:
            texts: список исходных строк
            field_type: 'address' или 'client'
            (для 'problem' используется categorize_and_normalize_batch —
            там нужна категоризация, а не только нормализация формулировки)

        Returns:
            список нормализованных строк (в том же порядке)

        Raises:
            LLMCallError: пробрасывается из ask() при исчерпании ретраев —
            вызывающий код (fetcher._normalize_batch) обязан поймать её и
            не кэшировать результат.
        """
        if not texts:
            return []

        # Определяем имя задачи
        task_name = f"normalize_{field_type}"
        task_prompt = TASK_PROMPTS.get(task_name)
        if not task_prompt:
            raise ValueError(f"Неизвестный тип поля '{field_type}'. Допустимые: address, client")

        # Формируем нумерованный список
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_prompt = task_prompt.format(texts=numbered)

        # Вызываем LLM с ролью normalizer (может бросить LLMCallError — не ловим
        # здесь намеренно, пусть решает вызывающий код)
        response = self.ask(
            user_prompt=user_prompt,
            system_role="normalizer",
            temperature=0.0
        )

        # Парсим ответ: ожидаем строки, разделённые переводами строк
        lines = response.strip().split('\n')
        # Очищаем каждую строку от возможных номеров и лишних пробелов
        cleaned = []
        for line in lines:
            # Убираем ведущие цифры, точки, пробелы
            clean = line.lstrip('0123456789. ').strip()
            if clean:
                cleaned.append(clean)

        # Если получили меньше строк, чем было — дополняем исходными
        if len(cleaned) < len(texts):
            logger.warning(f"Ответ содержал {len(cleaned)} строк вместо {len(texts)}. Заполняем пропуски.")
            cleaned.extend(texts[len(cleaned):])
        # Если больше — обрезаем
        cleaned = cleaned[:len(texts)]

        return cleaned

    def normalize(self, text: str, field_type: str) -> str:
        """
        Одиночная нормализация (для обратной совместимости).
        """
        return self.normalize_batch([text], field_type)[0]

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