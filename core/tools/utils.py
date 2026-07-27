# /root/agent/core/tools/utils.py
"""
Общие утилиты для инструментов.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger


def parse_date(date_str: str) -> pd.Timestamp:
    """
    Преобразует строку даты в pandas.Timestamp.
    Поддерживает форматы: '2025-11-08 16:34:39', '2025-11-08'
    """
    try:
        return pd.to_datetime(date_str)
    except Exception as e:
        logger.error(f"Ошибка парсинга даты '{date_str}': {e}")
        return pd.NaT


def get_preview_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Возвращает список колонок, доступных в df для отображения в таблице,
    и словарь русских названий.
    """
    standard_cols = [
        'date', 'ticket_id', 'client_normalized', 'address_normalized',
        'problem_normalized', 'status'
    ]
    ru_names = {
        'date': 'Дата',
        'ticket_id': 'Номер тикета',
        'client_normalized': 'Клиент',
        'address_normalized': 'Адрес',
        'problem_normalized': 'Проблема',
        'status': 'Статус'
    }
    available = [col for col in standard_cols if col in df.columns]
    return available, ru_names


def format_answer(
    summary: str = "",
    answer: str = "",
    table: Optional[pd.DataFrame] = None,
    figure: Any = None
) -> Dict[str, Any]:
    """
    Собирает стандартный словарь ответа инструмента.
    """
    if table is not None and not table.empty:
        # Убираем существующую колонку '№', если есть (избегаем дублирования)
        if '№' in table.columns:
            table = table.drop(columns=['№'])
        # Вставляем порядковый номер первой колонкой
        table.insert(0, '№', range(1, len(table) + 1))

    result = {
        "summary": summary,
        "answer": answer,
        "preview_data": table if table is not None else pd.DataFrame(),
        "figure": figure
    }
    return result


def filter_by_date(
    df: pd.DataFrame,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    last_n_days: Optional[int] = None,
    consecutive_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Фильтрует DataFrame по одному из трёх временных режимов.
    Режимы взаимоисключающие, приоритет:
    1. consecutive_days
    2. last_n_days
    3. date_from + date_to
    """
    if 'date' not in df.columns:
        logger.warning("Колонка 'date' отсутствует, фильтрация по дате невозможна")
        return df

    # Копируем и парсим даты
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_parsed'])
    if df.empty:
        return df

    if consecutive_days is not None:
        # Режим: найти непрерывный период длиной >= consecutive_days дней с обращениями
        # Сортируем уникальные даты
        unique_dates = sorted(df['date_parsed'].dt.date.unique())
        if not unique_dates:
            return pd.DataFrame()  # пустой DataFrame

        # Ищем непрерывные последовательности
        best_start = None
        best_end = None
        best_len = 0
        current_start = unique_dates[0]
        current_len = 1

        for i in range(1, len(unique_dates)):
            prev_date = unique_dates[i-1]
            curr_date = unique_dates[i]
            if (curr_date - prev_date).days == 1:
                current_len += 1
            else:
                # конец последовательности, проверяем
                if current_len >= consecutive_days and current_len > best_len:
                    best_start = current_start
                    best_end = unique_dates[i-1]
                    best_len = current_len
                # начинаем новую
                current_start = curr_date
                current_len = 1

        # Проверяем последнюю последовательность
        if current_len >= consecutive_days and current_len > best_len:
            best_start = current_start
            best_end = unique_dates[-1]
            best_len = current_len

        if best_start is None:
            logger.info(f"Не найдено непрерывного периода >= {consecutive_days} дней")
            return pd.DataFrame()  # пусто

        logger.info(f"Найден непрерывный период {best_start} - {best_end} ({best_len} дней)")
        # фильтруем df по датам
        mask = (df['date_parsed'] >= pd.Timestamp(best_start)) & \
               (df['date_parsed'] <= pd.Timestamp(best_end) + pd.Timedelta(days=1))
        return df[mask].drop(columns=['date_parsed'])

    if last_n_days is not None:
        cutoff = datetime.now() - timedelta(days=last_n_days)
        mask = df['date_parsed'] >= pd.Timestamp(cutoff)
        logger.info(f"Фильтр за последние {last_n_days} дней, начиная с {cutoff}")
        return df[mask].drop(columns=['date_parsed'])

    if date_from is not None and date_to is not None:
        try:
            from_dt = pd.Timestamp(date_from)
            to_dt = pd.Timestamp(date_to) + pd.Timedelta(days=1)
            mask = (df['date_parsed'] >= from_dt) & (df['date_parsed'] < to_dt)
            logger.info(f"Фильтр с {date_from} по {date_to}")
            return df[mask].drop(columns=['date_parsed'])
        except Exception as e:
            logger.error(f"Ошибка в датах фильтра: {e}")
            return df.drop(columns=['date_parsed'])

    # Если ничего не задано – вернуть без фильтра
    return df.drop(columns=['date_parsed'])