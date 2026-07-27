"""
Главный диспетчер: получает вопрос, выбирает инструменты и возвращает ответ.
"""

import json
import re
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from core.data.loader import DataLoader
from core.llm.client import DeepSeekClient
from core.tools.registry import TOOL_REGISTRY, execute_tool
from core.llm.prompts import SYSTEM_TOOL_SELECTOR, SYSTEM_FALLBACK_ANALYST, TASK_FALLBACK_ANALYST
from core.utils.dialog_logger import dialog_logger


class AgentDispatcher:
    """
    Координирует загрузку данных, выбор инструментов и генерацию ответа.
    """

    def __init__(self, llm_client: DeepSeekClient, indexer=None):
        """
        Инициализация диспетчера.
        
        Args:
            llm_client: клиент для общения с LLM (DeepSeek)
        """
        self.llm = llm_client
        self.loader = DataLoader()
        self.indexer = indexer

    def process(self, question: str, selected_sheets: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Основной метод обработки вопроса пользователя.
        
        Алгоритм:
        1. Загружаем данные ТОЛЬКО из явно выбранных листов (мультиселект в UI)
        2. Пытаемся выбрать инструмент через LLM
        3. Если LLM не справилась — используем fallback на правилах
        4. Если и fallback не сработал — отдаём данные напрямую LLM как ассистенту
        5. Выполняем выбранный инструмент и возвращаем результат

        Args:
            question: вопрос пользователя на естественном языке
            selected_sheets: список словарей с ключами 'table_name' и 'sheet_name',
                             выбранных пользователем в UI

        Returns:
            словарь с ключами:
                - answer (str): текстовый ответ для пользователя
                - table (pd.DataFrame): данные для отображения в UI
                - figure (опционально): объект plotly для графика
        """
        start_time = time.time()
        logger.info(f"Диспетчер получил вопрос: «{question}»")
        logger.debug(f"Явно выбрано листов: {len(selected_sheets)}")

        try:
            # ------------------------------------------------------------
            # Шаг 1. Загрузка данных из выбранных листов
            # ------------------------------------------------------------
            df = self.loader.get_sheets(selected_sheets)
            logger.info(f"Загружено {len(df)} строк из выбранных листов")

            if df.empty:
                answer = "В выбранных листах нет данных для анализа."
                dialog_logger.log_dialog(
                    question=question,
                    answer=answer,
                    selected_sheets=selected_sheets,
                    response_time=round(time.time() - start_time, 2)
                )
                return {
                    "answer": answer,
                    "preview_data": pd.DataFrame(),
                    "figure": None
                }

            # ------------------------------------------------------------
            # Шаг 2. Попытка выбрать инструмент через LLM
            # ------------------------------------------------------------
            tool_call = self._ask_llm_to_choose_tool(question)

            # ------------------------------------------------------------
            # Шаг 3. Fallback на правилах, если LLM не вернула инструмент
            # ------------------------------------------------------------
            if not tool_call:
                tool_call = self._fallback_tool_selection(question)

            # ------------------------------------------------------------
            # Шаг 4. Если ничего не выбрано — LLM как ассистент
            # ------------------------------------------------------------
            if not tool_call:
                logger.info("Инструмент не выбран, использую LLM как ассистента с данными")
                result = self._ask_llm_directly(question, df)
                dialog_logger.log_dialog(
                    question=question,
                    answer=result.get("answer", ""),
                    selected_sheets=selected_sheets,
                    response_time=round(time.time() - start_time, 2)
                )
                return result
                

            # ------------------------------------------------------------
            # Шаг 5. Выполнение выбранного инструмента
            # ------------------------------------------------------------
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("arguments", {})
            logger.info(f"Выбран инструмент: {tool_name} с аргументами {tool_args}")

            result = execute_tool(tool_name, tool_args, df, llm=self.llm)

            # ------------------------------------------------------------
            # Шаг 6. Формирование ответа
            # ------------------------------------------------------------
            answer = result.get("answer", "Инструмент выполнен, но не сформировал ответ.")
            preview_data = result.get("preview_data", pd.DataFrame())
            figure = result.get("figure")
            
            # Логируем диалог
            dialog_logger.log_dialog(
                question=question,
                answer=answer,
                selected_sheets=selected_sheets,
                response_time=round(time.time() - start_time, 2)
            )
            
            return {
                "answer": answer,
                "preview_data": preview_data,
                "figure": figure
            }

        except Exception as e:
            logger.exception(f"Критическая ошибка в диспетчере: {e}")
            answer = "Ой! Что-то пошло не так. Пожалуйста, попробуйте ещё раз или переформулируйте вопрос."

            dialog_logger.log_dialog(
                question=question,
                answer=answer,
                selected_sheets=selected_sheets,
                response_time=round(time.time() - start_time, 2)
            )
            return {
                "answer": answer,
                "preview_data": pd.DataFrame(),
                "figure": None
            }

    # =========================================================================
    # LLM КАК АССИСТЕНТ (запасной вариант, если инструмент не выбран)
    # =========================================================================

    def _ask_llm_directly(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Использует LLM как аналитика, передавая ВСЕ данные.
        Вызывается, когда не удалось выбрать конкретный инструмент.
        """
        logger.info(f"Прямой вызов LLM с данными: {len(df)} строк")

        # Берём все полезные колонки
        norm_cols = [
            'date', 'ticket_id', 'status',
            'client_normalized', 'address_normalized', 
            'problem_normalized', 'problem_raw',
            'point_key'
        ]
        available_cols = [col for col in norm_cols if col in df.columns]

        if not available_cols:
            return {
                "answer": ("В данных нет нормализованных колонок для анализа. "
                      "Возможно, данные ещё не были обработаны. "
                      "Попробуйте нажать кнопку «Обновить данные»."),
                "preview_data": df.head(100),
                "figure": None
            }

        # Ограничим чтобы не сжечь все токены
        max_rows = 300
        display_df = df[available_cols].head(max_rows).copy()
    
        # Обрежем длинные строки
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].str.slice(0, 80)

        # Превращаем в текст
        data_text = display_df.to_string(index=False)

        # Формируем промпт из шаблона
        prompt = TASK_FALLBACK_ANALYST.format(
            rows_shown=min(len(df), max_rows),
            total_rows=len(df),
            data_text=data_text,
            question=question
        )

        try:
            response = self.llm.ask(
                user_prompt=prompt,
                system_prompt=SYSTEM_FALLBACK_ANALYST,
                temperature=0.0
            )
            logger.info(f"Прямой LLM-ответ получен, длина: {len(response)} символов")
            return {
                "answer": response,
                "preview_data": pd.DataFrame(),
                "figure": None
            }
        except Exception as e:
            logger.error(f"Ошибка при прямом обращении к LLM: {e}")
            return {
                "answer": "Не удалось обработать запрос. Попробуйте переформулировать вопрос.",
                "preview_data": pd.DataFrame(),
                "figure": None
            }

    # =========================================================================
    # ВЫБОР ИНСТРУМЕНТА ЧЕРЕЗ LLM
    # =========================================================================

    def _ask_llm_to_choose_tool(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Отправляет вопрос в LLM с системным промптом tool_selector.
        Ожидает ответ в формате JSON: {"tool": "...", "arguments": {...}}
        """
        try:
            # Используем шаблон SYSTEM_TOOL_SELECTOR с плейсхолдером {question}
            prompt = SYSTEM_TOOL_SELECTOR.replace("{question}", question)
            
            response = self.llm.ask(
                user_prompt=prompt,
                system_role="tool_selector",
                temperature=0.0
            )
            logger.debug(f"Сырой ответ LLM (tool_selector): {response}")

            # Очистка от markdown-обёртки ```json ... ```
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            if not response or response == "{}":
                logger.warning("LLM вернул пустой JSON")
                return None

            data = json.loads(response)
            if not data or "tool" not in data:
                logger.warning(f"LLM вернул JSON без ключа 'tool': {data}")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от LLM: {e}\nОтвет: {response}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при вызове LLM для выбора инструмента: {e}")
            return None

    # =========================================================================
    # FALLBACK: ВЫБОР ИНСТРУМЕНТА ПО КЛЮЧЕВЫМ СЛОВАМ
    # =========================================================================

    def _fallback_tool_selection(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Запасной метод выбора инструмента на основе ключевых слов.
        Используется, если LLM не смогла определить инструмент.
        """
        q = question.lower()

        # --- Топ клиентов ---
        if any(w in q for w in ["кто чаще", "топ клиент", "самые частые клиенты", "клиенты чаще всего"]):
            n_match = re.search(r'топ[-\s]*(\d+)', q)
            n = int(n_match.group(1)) if n_match else 20
            return {"tool": "get_top_clients", "arguments": {"n": n}}

        # --- Топ точек ---
        if any(w in q for w in ["топ точек", "какие точки", "точки чаще всего", 
                                "активные точки", "топ магазин"]):
            n_match = re.search(r'топ[-\s]*(\d+)', q)
            n = int(n_match.group(1)) if n_match else 20
            return {"tool": "get_top_points", "arguments": {"n": n}}

        # --- Топ проблем ---
        if any(w in q for w in ["проблем", "ошибок", "сбоев", "частые проблемы", "топ проблем"]):
            n_match = re.search(r'топ[-\s]*(\d+)', q)
            n = int(n_match.group(1)) if n_match else 20
            return {"tool": "get_top_problems", "arguments": {"n": n}}

        # --- Клиент + проблема (комбинированный поиск) ---
        # Пытаемся извлечь клиента и ключевые слова проблемы
        client_match = re.search(
            r'(?:клиент[а]?|у|для)\s+([^\s]+(?:\s+[^\s]+){0,2})', q
        )
        problem_words = []
        for word in ["проблем", "ошибк", "сканер", "касс", "егаис", "утм", "честный знак"]:
            if word in q:
                # Извлекаем контекст вокруг проблемного слова
                problem_match = re.search(rf'\b\w*{word}\w*\b', q)
                if problem_match:
                    problem_words.append(problem_match.group())
        
        if client_match and problem_words:
            return {
                "tool": "search_combined",
                "arguments": {
                    "client_query": client_match.group(1).strip(),
                    "problem_query": " ".join(problem_words)
                }
            }

        # --- Конкретный клиент (без проблемы) ---
        if client_match and not problem_words:
            return {
                "tool": "search_client",
                "arguments": {"client_name": client_match.group(1).strip()}
            }

        # --- Конкретная точка ---
        point_match = re.search(
            r'(?:точк[аиеу]|магазин[ае]?)\s+([^\s]+(?:\s+[^\s]+){0,3})', q
        )
        if point_match:
            return {
                "tool": "search_point",
                "arguments": {"point_query": point_match.group(1).strip()}
            }

        # --- Проблема (без клиента/точки) ---
        if problem_words:
            return {
                "tool": "search_problem",
                "arguments": {"problem_query": " ".join(problem_words)}
            }

        logger.info(f"Fallback не смог подобрать инструмент для вопроса: «{question}»")
        return None