"""
Инструменты для анализа проблем (problem_normalized / problem_tags).

=============================================================================
ИЗМЕНЕНО: get_top_problems() больше НЕ вызывает LLM для группировки.
=============================================================================
Раньше группировка была нужна, потому что problem_normalized — это просто
сырые нормализованные строки без структуры, и единственный способ понять,
что "отпал сканер" и "сканер не работает" — про одно и то же, был семантический
LLM-вызов (TASK_GROUP_PROBLEMS).

Теперь каждая проблема уже размечена тегами категорий (problem_tags) на
этапе нормализации в fetcher.py (слой 1 fuzzy + слой 2 LLM с function
calling, см. core/data/fetcher.py). Категория — это и есть группа, поэтому
топ проблем можно посчитать простым value_counts() по тегам, без LLM,
без риска, что модель вернёт невалидный JSON, и без затрат на токены.

ВАЖНО — многотегово: одно обращение может иметь 1-3 тега (problem_tags —
список). Значит, сумма счётчиков по всем категориям МОЖЕТ ПРЕВЫШАТЬ
количество обращений — одно обращение с двумя тегами учитывается в обеих
категориях. Это осознанное решение (см. чат): точнее отражает
многогранные проблемы (например, "обновление Фронтол и прошивка ККТ"
относится и к Фронтолу, и к ККТ), но означает, что "Топ проблем" — это
не разбиение на непересекающиеся группы, а скорее срез "сколько обращений
затронули эту категорию".

TASK_GROUP_PROBLEMS в prompts.py удалён (ты решила снести легаси) — этот
файл больше не импортирует и не использует его.
=============================================================================
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger
from .utils import get_preview_columns, format_answer, filter_by_date
from core.llm.client import DeepSeekClient
from core.llm.prompts import TASK_SELECT_PROBLEMS


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

    ПРИМЕЧАНИЕ: этот метод не тронут в этой итерации — он используется
    search_problem/search_problem_by_date, где полнотекстовый семантический
    поиск по формулировке всё ещё осмыслен (запрос пользователя произвольный,
    не обязательно совпадает с названием категории). Обсудим отдельно,
    стоит ли тут тоже задействовать problem_tags.
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


# =============================================================================
# ПЕРЕПИСАНО: get_top_problems теперь считает по problem_tags, без LLM.
# =============================================================================
def get_top_problems(df: pd.DataFrame,
                     n: int = 20,
                     min_tickets: int = 2,
                     llm: Optional[DeepSeekClient] = None) -> Dict[str, Any]:
    """
    Топ N категорий проблем ИЛИ все категории с количеством > min_tickets.

    Считает напрямую по problem_tags (категории, присвоенные на этапе
    нормализации в fetcher.py) — без обращения к LLM. Параметр llm оставлен
    в сигнатуре только для обратной совместимости с вызывающим кодом
    (например, tool_selector может передавать его по привычке) — фактически
    не используется и функция больше не требует его наличия.

    Если n > 0: топ-N категорий (по умолчанию 20).
    Если n = 0: все категории с количеством > min_tickets (по умолчанию 2).

    ВАЖНО: обращение может иметь 1-3 тега, поэтому сумма количеств по всем
    категориям может превышать общее число обращений — одно обращение
    с несколькими тегами учитывается в каждой из своих категорий.
    """
    if df.empty or 'problem_tags' not in df.columns:
        return format_answer(summary="Нет данных о проблемах.", answer="Нет данных о проблемах.")

    # Оставляем только строки, где problem_tags — непустой список.
    # (пустой список — например, problem_raw было пустым при загрузке)
    # ВАЖНО (баг-фикс): изначально здесь была проверка isinstance(x, list),
    # которая работает для списков в памяти сразу после fetcher.py, НО НЕ
    # работает после round-trip через parquet — pandas/pyarrow при чтении
    # list-колонки обратно возвращает значения как numpy.ndarray, а не как
    # Python list. isinstance(ndarray, list) всегда False, из-за чего ВСЕ
    # строки ошибочно считались "без тегов", хотя данные были полностью
    # корректны (подтверждено диагностикой на реальном датасете: 17298 из
    # 17298 строк содержали теги, просто как ndarray). hasattr(x, '__len__')
    # одинаково работает и для list, и для ndarray, а None/NaN его не имеют
    # и корректно отфильтровываются.
    has_tags_mask = df['problem_tags'].apply(lambda x: hasattr(x, '__len__') and len(x) > 0)
    tagged_df = df[has_tags_mask]

    if tagged_df.empty:
        return format_answer(
            summary="Нет обращений с присвоенными категориями.",
            answer="Нет обращений с присвоенными категориями (problem_tags пуст у всех строк)."
        )

    # explode: одно обращение с 2-3 тегами превращается в 2-3 строки —
    # ровно то, что нам и нужно для подсчёта "сколько обращений затронули
    # каждую категорию" (см. предупреждение о многотегово в шапке файла)
    exploded = tagged_df.explode('problem_tags')

    counts = exploded['problem_tags'].value_counts().reset_index()
    counts.columns = ['category', 'count']

    if counts.empty:
        return format_answer(summary="Категорий не найдено.", answer="Категорий не найдено.")

    if n > 0:
        top = counts.head(n)
        mode_desc = f"Топ {min(n, len(top))} категорий проблем"
    else:
        top = counts[counts['count'] > min_tickets]
        mode_desc = f"Категории проблем с более чем {min_tickets} обращениями"

    if top.empty:
        return format_answer(summary=f"{mode_desc}: не найдено.", answer=f"{mode_desc}: не найдено.")

    # Примеры для каждой категории — берём несколько уникальных
    # нормализованных формулировок, которые реально попали в эту категорию
    rows = []
    for _, row in top.iterrows():
        category = row['category']
        count = row['count']
        examples = (
            exploded.loc[exploded['problem_tags'] == category, 'problem_normalized']
            .dropna()
            .unique()[:3]
            .tolist()
        )
        rows.append({
            'Категория': category,
            'Количество': count,
            'Примеры': ', '.join(examples) if examples else '—'
        })

    result_df = pd.DataFrame(rows)
    result_df.insert(0, '№', range(1, len(result_df) + 1))

    answer_lines = [
        f"{row['№']}. {row['Категория']} — {row['Количество']} обр. (примеры: {row['Примеры']})"
        for _, row in result_df.iterrows()
    ]
    answer = f"{mode_desc}:\n" + "\n".join(answer_lines)
    summary = f"{mode_desc}: {result_df.iloc[0]['Категория']} ({result_df.iloc[0]['Количество']} обр.)..."

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

    ПРИМЕЧАНИЕ: не тронут в этой итерации, см. комментарий у _filter_by_problem_query.
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

    ПРИМЕЧАНИЕ: не тронут в этой итерации, см. комментарий у _filter_by_problem_query.
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