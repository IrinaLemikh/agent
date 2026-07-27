"""
Инструменты для анализа проблем (problem_normalized).
Используют LLM для семантической группировки и поиска.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger
from .utils import get_preview_columns, format_answer, filter_by_date
from core.llm.client import DeepSeekClient
from core.llm.prompts import TASK_GROUP_PROBLEMS, TASK_SELECT_PROBLEMS


def _parse_llm_json(response_text: str) -> Optional[Any]:
    """Пытается извлечь JSON из ответа LLM, даже если есть лишний текст."""
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        else:
            return json.loads(response_text.strip())
    except Exception as e:
        logger.error(f"Ошибка парсинга JSON: {e}, ответ: {response_text[:200]}")
        return None


def _filter_by_problem_query(df: pd.DataFrame,
                             problem_query: str,
                             llm: DeepSeekClient) -> pd.DataFrame:
    """
    Возвращает DataFrame строк, чьи problem_normalized семантически соответствуют запросу.
    Использует LLM для выбора подходящих формулировок из уникальных значений.
    """
    unique_problems = df['problem_normalized'].dropna().unique().tolist()
    if not unique_problems:
        return pd.DataFrame()

    MAX_ITEMS = 3000
    if len(unique_problems) > MAX_ITEMS:
        top_problems = df['problem_normalized'].value_counts().head(MAX_ITEMS).index.tolist()
    else:
        top_problems = unique_problems

    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(top_problems))

    prompt = TASK_SELECT_PROBLEMS.format(
        problem_query=problem_query,
        numbered=numbered
    )

    response = llm.ask(user_prompt=prompt, system_role="grouper", temperature=0.1)
    selected = _parse_llm_json(response)

    if selected is None or not isinstance(selected, list):
        logger.warning("LLM не вернул список формулировок, выполняем резервный поиск по подстроке")
        keyword_mask = pd.Series(False, index=df.index)
        for word in problem_query.strip().split():
            keyword_mask |= df['problem_normalized'].str.contains(word, case=False, na=False)
        return df[keyword_mask].copy()
    else:
        return df[df['problem_normalized'].isin(selected)].copy()


def get_top_problems(df: pd.DataFrame,
                     n: int = 20,
                     min_tickets: int = 2,
                     llm: Optional[DeepSeekClient] = None) -> Dict[str, Any]:
    """
    Топ N проблем (сгруппированных семантически) ИЛИ все проблемы > min_tickets.
    Использует LLM для группировки похожих формулировок из problem_normalized.

    Если n > 0: топ-N проблем (по умолчанию 20).
    Если n = 0: все проблемы с количеством > min_tickets (по умолчанию 2).
    """
    if df.empty or 'problem_normalized' not in df.columns:
        return format_answer(summary="Нет данных о проблемах.", answer="Нет данных о проблемах.")
    if llm is None:
        return format_answer(summary="LLM недоступен.", answer="Ошибка: LLM не инициализирован.")

    counts = df['problem_normalized'].value_counts().reset_index()
    counts.columns = ['problem', 'count']
    if counts.empty:
        return format_answer(summary="Нет проблем.", answer="Нет проблем для анализа.")

    items_list = [f"{row['problem']} ({row['count']})" for _, row in counts.iterrows()]
    numbered_items = [f"{i+1}. {item}" for i, item in enumerate(items_list)]
    items_text = "\n".join(numbered_items)

    prompt = TASK_GROUP_PROBLEMS.format(items_text=items_text)
    
    response = llm.ask(user_prompt=prompt, system_role="grouper", temperature=0.1)
    
    parsed = _parse_llm_json(response)

    if parsed is None:
        logger.warning("LLM не вернул валидный JSON, возвращаем несгруппированный топ")
        if n > 0:
            top = counts.head(n)
        else:
            top = counts[counts['count'] > min_tickets]
        if top.empty:
            return format_answer(summary="Проблем не найдено.", answer="Проблем не найдено.")
        top.insert(0, '№', range(1, len(top) + 1))
        answer_lines = [f"{row['№']}. {row['problem']} — {row['count']} раз(а)" for _, row in top.iterrows()]
        return format_answer(
            summary=f"Топ проблем (без группировки): {top.iloc[0]['problem']}...",
            answer="ВНИМАНИЕ! Не удалось сгруппировать проблемы через LLM. Точность ответа низкая. Попробуйте сузить дипапзон данных для поиска (например: запросить топ 5, вместо топ 10 или уменьшить кол-во листов).\n" + "\n".join(answer_lines),
            table=top.rename(columns={'problem': 'Проблема', 'count': 'Количество'})
        )

    groups = []
    for item in parsed:
        # Достаём ID и подтягиваем текст сами
        example_ids = item.get('example_ids', [])
        # items_list имеет вид ["проблема (кол-во)", ...], вытаскиваем текст до скобки
        example_texts = []
        for eid in example_ids:
            try:
                eid = int(eid)
            except (ValueError, TypeError):
                continue
            
            if 1 <= eid <= len(items_list):
                # Берём текст проблемы (всё до последней скобки с числом)
                full_text = items_list[eid - 1]
                # "Не работает касса (42)" -> "Не работает касса"
                problem_text = full_text.rsplit(' (', 1)[0]
                example_texts.append(problem_text)

        groups.append({
            'Группа': item.get('group', ''),
            'Количество': item.get('total_count', 0),
            'Примеры': ', '.join(example_texts) if example_texts else '—'
        })

    result_df = pd.DataFrame(groups)
    if result_df.empty:
        return format_answer(summary="Проблемы не найдены.", answer="Проблемы не найдены.")

    if n > 0:
        result_df = result_df.head(n)
        mode_desc = f"Топ {min(n, len(result_df))} проблем"
    else:
        result_df = result_df[result_df['Количество'] > min_tickets]
        mode_desc = f"Проблемы с более чем {min_tickets} обращениями"

    if result_df.empty:
        return format_answer(summary=f"{mode_desc}: не найдено.", answer=f"{mode_desc}: не найдено.")

    result_df.insert(0, '№', range(1, len(result_df) + 1))
    answer_lines = [f"{row['№']}. {row['Группа']} — {row['Количество']} обр. (примеры: {row['Примеры']})"
                    for _, row in result_df.iterrows()]
    answer = f"{mode_desc}:\n" + "\n".join(answer_lines)
    summary = f"{mode_desc}: {result_df.iloc[0]['Группа']} ({result_df.iloc[0]['Количество']} обр.)..."

    return format_answer(
        summary=summary,
        answer=answer,
        table=result_df
    )


def search_problem(df: pd.DataFrame,
                   problem_query: str,
                   llm: Optional[DeepSeekClient] = None) -> Dict[str, Any]:
    """
    Поиск ВСЕХ обращений, семантически связанных с problem_query.
    LLM выбирает из уникальных значений problem_normalized те, которые относятся к запросу.
    """
    if df.empty or 'problem_normalized' not in df.columns:
        return format_answer(summary="Нет данных о проблемах.", answer="Нет данных о проблемах.")
    if llm is None:
        return format_answer(summary="LLM недоступен.", answer="Ошибка: LLM не инициализирован.")
    if not problem_query.strip():
        return format_answer(summary="Не указан поисковый запрос.", answer="Укажите ключевые слова для поиска проблемы.")

    filtered_df = _filter_by_problem_query(df, problem_query, llm)

    if filtered_df.empty:
        return format_answer(
            summary=f"По запросу '{problem_query}' ничего не найдено. Попробуйте перефразировать запрос.",
            answer=f"По запросу '{problem_query}' обращений не найдено."
        )

    total = len(filtered_df)
    clients_count = filtered_df['client_normalized'].nunique() if 'client_normalized' in filtered_df.columns else 0
    summary = f"Проблема '{problem_query}': {total} обращений, {clients_count} клиентов"
    answer = f"Обращения по проблеме '{problem_query}':\nВсего обращений: {total}\nКлиентов: {clients_count}"

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


def search_problem_by_date(df: pd.DataFrame,
                           problem_query: str,
                           llm: Optional[DeepSeekClient] = None,
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None,
                           last_n_days: Optional[int] = None,
                           consecutive_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Поиск по проблеме + фильтр по датам.
    Сначала ищем проблему через LLM, потом применяем дата-фильтр.
    """
    if llm is None:
        return format_answer(summary="LLM недоступен.", answer="Ошибка: LLM не инициализирован.")
    if not problem_query.strip():
        return format_answer(summary="Не указан поисковый запрос.", answer="Укажите ключевые слова для поиска проблемы.")

    prob_df = _filter_by_problem_query(df, problem_query, llm)
    if prob_df.empty:
        return format_answer(
            summary=f"По запросу '{problem_query}' ничего не найдено. Попробуйте перефразировать запрос.",
            answer=f"По запросу '{problem_query}' обращений не найдено."
        )

    filtered_df = filter_by_date(prob_df, date_from, date_to, last_n_days, consecutive_days)
    if filtered_df.empty:
        return format_answer(
            summary=f"Обращения по проблеме '{problem_query}' за указанный период не найдены. Попробуйте перефразировать запрос.",
            answer=f"Обращения по проблеме '{problem_query}' за указанный период не найдены. Попробуйте перефразировать запрос."
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

    summary = f"Проблема '{problem_query}' {period_desc}: {total} обращений"
    answer = f"Обращения по проблеме '{problem_query}' {period_desc}:\nВсего: {total}, последнее {last_date}"

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