# /root/agent/core/tools/point_tools.py
"""
Инструменты для анализа по торговым точкам (point_key = клиент | адрес).
"""

import pandas as pd
from typing import Dict, Any, Optional
from .utils import get_preview_columns, format_answer, filter_by_date


def get_top_points(df: pd.DataFrame,
                   n: int = 20,
                   min_tickets: int = 2) -> Dict[str, Any]:
    """
    Топ N торговых точек по количеству обращений ИЛИ все точки с > min_tickets обращений.
    Если n > 0: топ-N (по умолчанию 20).
    Если n = 0: все точки с количеством обращений > min_tickets (по умолчанию 2).
    """
    if df.empty or 'point_key' not in df.columns:
        return format_answer(
            summary="Нет данных о торговых точках.",
            answer="Нет данных о торговых точках."
        )

    counts = df['point_key'].value_counts().reset_index()
    counts.columns = ['Торговая точка', 'Количество обращений']

    if n > 0:
        result = counts.head(n)
        mode_desc = f"Топ {min(n, len(result))} торговых точек"
    else:
        result = counts[counts['Количество обращений'] > min_tickets]
        mode_desc = f"Торговые точки с более чем {min_tickets} обращениями"

    if result.empty:
        return format_answer(
            summary=f"{mode_desc}: не найдено. Попробуйте перефразировать запрос.",
            answer=f"{mode_desc}: не найдено. Попробуйте перефразировать запрос."
        )

    result.insert(0, '№', range(1, len(result) + 1))
    top_list = [f"{row['№']}. {row['Торговая точка']} — {row['Количество обращений']} обр."
                for _, row in result.iterrows()]
    answer = f"{mode_desc} по количеству обращений:\n" + "\n".join(top_list)
    summary = f"{mode_desc}: {result.iloc[0]['Торговая точка']} ({result.iloc[0]['Количество обращений']} обр.)..."

    return format_answer(
        summary=summary,
        answer=answer,
        table=result
    )


def search_point(df: pd.DataFrame,
                 point_query: str) -> Dict[str, Any]:
    """
    Поиск всех обращений по торговой точке.
    Поддерживает форматы:
    - "Пивко | Хрустальная 37" (точное указание клиента и адреса)
    - "Пивко" (поиск по point_key содержит)
    - "Хрустальная" (поиск по point_key содержит)
    """
    if df.empty or 'point_key' not in df.columns:
        return format_answer(
            summary="Нет данных о торговых точках.",
            answer="Нет данных о торговых точках."
        )
    if not point_query.strip():
        return format_answer(
            summary="Не указан запрос для поиска точки.",
            answer="Укажите название торговой точки или 'Клиент | Адрес'."
        )

    query = point_query.strip()
    if '|' in query:
        parts = query.split('|')
        client_part = parts[0].strip()
        address_part = parts[1].strip() if len(parts) > 1 else ""
        mask_client = df['client_normalized'].str.contains(client_part, case=False, na=False) \
            if 'client_normalized' in df.columns else pd.Series(False, index=df.index)
        mask_address = df['address_normalized'].str.contains(address_part, case=False, na=False) \
            if 'address_normalized' in df.columns else pd.Series(False, index=df.index)
        mask = mask_client & mask_address
        point_desc = f"{client_part} | {address_part}"
    else:
        mask = df['point_key'].str.contains(query, case=False, na=False)
        point_desc = query

    point_df = df[mask].copy()
    if point_df.empty:
        return format_answer(
            summary=f"Торговая точка '{point_desc}' не найдена. Попробуйте перефразировать запрос.",
            answer=f"Торговая точка '{point_desc}' не найдена. Попробуйте перефразировать запрос."
        )

    total = len(point_df)
    last_date = point_df['date'].max() if 'date' in point_df.columns else "н/д"
    summary = f"Торговая точка '{point_desc}': {total} обращений, последнее {last_date}"
    answer = f"Обращения по торговой точке '{point_desc}':\nВсего: {total}, последнее: {last_date}"

    avail_cols, ru_names = get_preview_columns(point_df)
    if avail_cols:
        preview = point_df[avail_cols].sort_values('date', ascending=False).reset_index(drop=True)
        preview.columns = [ru_names.get(col, col) for col in avail_cols]
    else:
        preview = pd.DataFrame()

    return format_answer(
        summary=summary,
        answer=answer,
        table=preview
    )


def search_point_by_date(df: pd.DataFrame,
                         point_query: str,
                         date_from: Optional[str] = None,
                         date_to: Optional[str] = None,
                         last_n_days: Optional[int] = None,
                         consecutive_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Все обращения торговой точки с фильтром по датам.
    """
    if df.empty or 'point_key' not in df.columns:
        return format_answer(summary="Нет данных о торговых точках.", answer="Нет данных о торговых точках.")
    if not point_query.strip():
        return format_answer(summary="Не указан запрос для поиска точки.", answer="Укажите название торговой точки.")

    # Сначала получаем обращения точки (как в search_point)
    query = point_query.strip()
    if '|' in query:
        parts = query.split('|')
        client_part = parts[0].strip()
        address_part = parts[1].strip() if len(parts) > 1 else ""
        mask_client = df['client_normalized'].str.contains(client_part, case=False, na=False) \
            if 'client_normalized' in df.columns else pd.Series(False, index=df.index)
        mask_address = df['address_normalized'].str.contains(address_part, case=False, na=False) \
            if 'address_normalized' in df.columns else pd.Series(False, index=df.index)
        mask = mask_client & mask_address
        point_desc = f"{client_part} | {address_part}"
    else:
        mask = df['point_key'].str.contains(query, case=False, na=False)
        point_desc = query

    point_df = df[mask].copy()
    if point_df.empty:
        return format_answer(
            summary=f"Торговая точка '{point_desc}' не найдена.",
            answer=f"Торговая точка '{point_desc}' не найдена."
        )

    # Применяем фильтр по дате
    filtered_df = filter_by_date(point_df, date_from, date_to, last_n_days, consecutive_days)
    if filtered_df.empty:
        return format_answer(
            summary=f"Обращения точки '{point_desc}' за указанный период не найдены. Попробуйте перефразировать запрос.",
            answer=f"Обращения точки '{point_desc}' за указанный период не найдены. Попробуйте перефразировать запрос."
        )

    total = len(filtered_df)
    last_date = filtered_df['date'].max() if 'date' in filtered_df.columns else "н/д"
    period_desc = ""
    if last_n_days:
        period_desc = f"за последние {last_n_days} дней"
    elif date_from and date_to:
        period_desc = f"с {date_from} по {date_to}"
    elif consecutive_days:
        period_desc = f"за {consecutive_days} дней подряд"

    summary = f"Точка '{point_desc}' {period_desc}: {total} обращений"
    answer = f"Обращения точки '{point_desc}' {period_desc}:\nВсего: {total}, последнее {last_date}"

    avail_cols, ru_names = get_preview_columns(filtered_df)
    if avail_cols:
        preview = filtered_df[avail_cols].sort_values('date', ascending=False).reset_index(drop=True)
        preview.columns = [ru_names.get(col, col) for col in avail_cols]
    else:
        preview = pd.DataFrame()

    return format_answer(
        summary=summary,
        answer=answer,
        table=preview
    )