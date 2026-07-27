# /root/agent/core/tools/client_tools.py
"""
Инструменты для анализа по клиентам (client_normalized).
"""

import pandas as pd
from typing import Dict, Any, Optional
from .utils import get_preview_columns, format_answer, filter_by_date


def get_top_clients(df: pd.DataFrame,
                    n: int = 20,
                    min_tickets: int = 2) -> Dict[str, Any]:
    """
    Топ N клиентов по количеству обращений ИЛИ все клиенты с > min_tickets обращений.
    Если n > 0: топ-N (по умолчанию 20).
    Если n = 0: все клиенты с количеством обращений > min_tickets (по умолчанию 2).
    """
    if df.empty or 'client_normalized' not in df.columns:
        return format_answer(
            summary="Нет данных о клиентах.",
            answer="Нет данных о клиентах."
        )

    counts = df['client_normalized'].value_counts().reset_index()
    counts.columns = ['Клиент', 'Количество обращений']

    if n > 0:
        result = counts.head(n)
        mode_desc = f"Топ {min(n, len(result))} клиентов"
    else:
        result = counts[counts['Количество обращений'] > min_tickets]
        mode_desc = f"Клиенты с более чем {min_tickets} обращениями"

    if result.empty:
        return format_answer(
            summary=f"{mode_desc}: не найдено.",
            answer=f"{mode_desc}: не найдено."
        )

    # Добавляем порядковый номер
    result.insert(0, '№', range(1, len(result) + 1))
    top_list = [f"{row['№']}. {row['Клиент']} — {row['Количество обращений']} обр."
                for _, row in result.iterrows()]
    answer = f"{mode_desc} по количеству обращений:\n" + "\n".join(top_list)
    summary = f"{mode_desc}: {result.iloc[0]['Клиент']} ({result.iloc[0]['Количество обращений']} обр.)..."

    return format_answer(
        summary=summary,
        answer=answer,
        table=result
    )


def search_client(df: pd.DataFrame,
                  client_name: str) -> Dict[str, Any]:
    """
    Поиск всех обращений клиента по ключевому слову в client_normalized.
    """
    if df.empty or 'client_normalized' not in df.columns:
        return format_answer(
            summary="Нет данных о клиентах.",
            answer="Нет данных о клиентах."
        )
    if not client_name.strip():
        return format_answer(
            summary="Не указано название клиента.",
            answer="Укажите название клиента для поиска."
        )

    mask = df['client_normalized'].str.contains(client_name.strip(), case=False, na=False)
    client_df = df[mask].copy()

    if client_df.empty:
        return format_answer(
            summary=f"Клиент '{client_name}' не найден. Попробуйте перефразировать запрос.",
            answer=f"Клиент '{client_name}' не найден. Попробуйте перефразировать запрос."
        )

    total = len(client_df)
    last_date = client_df['date'].max() if 'date' in client_df.columns else "н/д"
    # Частые проблемы
    problems_str = ""
    if 'problem_normalized' in client_df.columns:
        top_problems = client_df['problem_normalized'].value_counts().head(5)
        if not top_problems.empty:
            problems_lines = [f"{i}. {prob} — {cnt} раз(а)" for i, (prob, cnt) in enumerate(top_problems.items(), 1)]
            problems_str = "Частые проблемы:\n" + "\n".join(problems_lines)

    summary = f"Клиент '{client_name}': {total} обращений, последнее {last_date}"
    answer = f"Все обращения клиента '{client_name}': {total} шт.\nПоследнее: {last_date}\n\n{problems_str}"

    # Формируем таблицу
    avail_cols, ru_names = get_preview_columns(client_df)
    if not avail_cols:
        return format_answer(summary=summary, answer=answer)

    preview = client_df[avail_cols].sort_values('date', ascending=False).reset_index(drop=True)
    preview.columns = [ru_names.get(col, col) for col in avail_cols]

    return format_answer(
        summary=summary,
        answer=answer,
        table=preview
    )


def search_client_by_date(df: pd.DataFrame,
                          client_name: str,
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None,
                          last_n_days: Optional[int] = None,
                          consecutive_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Все обращения клиента с дополнительным фильтром по датам.
    Сначала фильтр по клиенту, затем по дате.
    """
    # Сначала получаем обращения клиента
    if df.empty or 'client_normalized' not in df.columns:
        return format_answer(summary="Нет данных о клиентах.", answer="Нет данных о клиентах.")
    if not client_name.strip():
        return format_answer(summary="Не указано название клиента.", answer="Укажите название клиента.")

    mask = df['client_normalized'].str.contains(client_name.strip(), case=False, na=False)
    client_df = df[mask].copy()

    if client_df.empty:
        return format_answer(
            summary=f"Клиент '{client_name}' не найден. Попробуйте перефразировать запрос.",
            answer=f"Клиент '{client_name}' не найден. Попробуйте перефразировать запрос."
        )

    # Применяем фильтр по дате
    filtered_df = filter_by_date(client_df, date_from, date_to, last_n_days, consecutive_days)
    if filtered_df.empty:
        return format_answer(
            summary=f"Обращения клиента '{client_name}' за указанный период не найдены.",
            answer=f"Обращения клиента '{client_name}' за указанный период не найдены."
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

    summary = f"Клиент '{client_name}' {period_desc}: {total} обращений"
    answer = f"Обращения клиента '{client_name}' {period_desc}:\nВсего: {total}, последнее {last_date}"

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