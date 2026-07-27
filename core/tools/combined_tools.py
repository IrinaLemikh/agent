
"""
Комбинированный инструмент для поиска по клиенту/точке + проблеме.
"""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger
from .utils import get_preview_columns, format_answer
from .client_tools import search_client
from .point_tools import search_point
from .problem_tools import _filter_by_problem_query
from core.llm.client import DeepSeekClient


def search_combined(df: pd.DataFrame,
                    client_query: Optional[str] = None,
                    point_query: Optional[str] = None,
                    problem_query: Optional[str] = None,
                    llm: Optional[DeepSeekClient] = None) -> Dict[str, Any]:
    """
    Комбинированный поиск: сначала сужаем по клиенту или точке (pandas),
    затем на результате ищем проблему через LLM.
    """
    if llm is None:
        return format_answer(summary="LLM недоступен.", answer="Ошибка: LLM не инициализирован.")

    if not client_query and not point_query:
        return format_answer(summary="Укажите клиента или торговую точку.", answer="Не указан клиент или точка.")
    if not problem_query:
        return format_answer(summary="Укажите проблему для поиска.", answer="Не указана проблема.")

    # Шаг 1: сужение по клиенту или точке
    if point_query:
        # Используем search_point, но нам нужен только DataFrame, без переименования
        # Быстрый вариант: повторить фильтрацию
        if '|' in point_query:
            parts = point_query.split('|')
            client_part = parts[0].strip()
            address_part = parts[1].strip() if len(parts) > 1 else ""
            mask_client = df['client_normalized'].str.contains(client_part, case=False, na=False) if 'client_normalized' in df.columns else pd.Series(False, index=df.index)
            mask_address = df['address_normalized'].str.contains(address_part, case=False, na=False) if 'address_normalized' in df.columns else pd.Series(False, index=df.index)
            mask = mask_client & mask_address
            entity_desc = f"{client_part} | {address_part}"
        else:
            mask = df['point_key'].str.contains(point_query.strip(), case=False, na=False) if 'point_key' in df.columns else pd.Series(False, index=df.index)
            entity_desc = point_query.strip()
        narrowed_df = df[mask].copy()
        if narrowed_df.empty:
            return format_answer(
                summary=f"Точка '{entity_desc}' не найдена.",
                answer=f"Торговая точка '{entity_desc}' не найдена."
            )
        entity_type = "точка"
    else:
        # Поиск по клиенту
        mask = df['client_normalized'].str.contains(client_query.strip(), case=False, na=False) if 'client_normalized' in df.columns else pd.Series(False, index=df.index)
        narrowed_df = df[mask].copy()
        if narrowed_df.empty:
            return format_answer(
                summary=f"Клиент '{client_query}' не найден. Попробуйте перефразировать запрос.",
                answer=f"Клиент '{client_query}' не найден. Попробуйте перефразировать запрос."
            )
        entity_desc = client_query.strip()
        entity_type = "клиент"

    # Шаг 2: фильтрация по проблеме через LLM на narrowed_df
    prob_filtered = _filter_by_problem_query(narrowed_df, problem_query, llm)
    if prob_filtered.empty:
        return format_answer(
            summary=f"У {entity_type} '{entity_desc}' не найдено обращений по проблеме '{problem_query}'. Попробуйте перефразировать запрос.",
            answer=f"У {entity_type} '{entity_desc}' не найдено обращений по проблеме '{problem_query}'. Попробуйте перефразировать запрос."
        )

    total = len(prob_filtered)
    last_date = prob_filtered['date'].max() if 'date' in prob_filtered.columns else "н/д"
    summary = f"{entity_type} '{entity_desc}', проблема '{problem_query}': {total} обращений"
    answer = f"Обращения {entity_type} '{entity_desc}' с проблемой '{problem_query}':\nВсего: {total}, последнее: {last_date}"

    avail_cols, ru_names = get_preview_columns(prob_filtered)
    if avail_cols:
        preview = prob_filtered[avail_cols].sort_values('date', ascending=False).reset_index(drop=True)
        preview.columns = [ru_names.get(col, col) for col in avail_cols]
    else:
        preview = pd.DataFrame()

    return format_answer(
        summary=summary,
        answer=answer,
        table=preview
    )