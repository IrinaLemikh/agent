"""
Универсальный клиент для DeepSeek API.
"""
import os
import requests
import time
from typing import List, Optional, Dict, Any
from config.settings import DEEPSEEK_API_KEY
from loguru import logger
from .prompts import SYSTEM_PROMPTS, TASK_PROMPTS

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

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "deepseek-v4-flash",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 50000
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                logger.debug(f"✅ Ответ получен (длина {len(answer)})")
                return answer

            except requests.exceptions.RequestException as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries + 1} не удалась: {e}")
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info(f"Повтор через {sleep_time:.1f}с...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Все попытки исчерпаны, возвращаем исходный промпт")
                    # В случае полной неудачи возвращаем user_prompt (чтобы не ломать поток)
                    return user_prompt

    def normalize_batch(self, texts: List[str], field_type: str) -> List[str]:
        """
        Нормализует пачку значений одного поля.

        Args:
            texts: список исходных строк
            field_type: 'address', 'client', 'problem'

        Returns:
            список нормализованных строк (в том же порядке)
        """
        if not texts:
            return []

        # Определяем имя задачи
        task_name = f"normalize_{field_type}"
        task_prompt = TASK_PROMPTS.get(task_name)
        if not task_prompt:
            raise ValueError(f"Неизвестный тип поля '{field_type}'. Допустимые: address, client, problem")

        # Формируем нумерованный список
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_prompt = task_prompt.format(texts=numbered)

        # Вызываем LLM с ролью normalizer
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