"""
Реконсиляция клиентов/адресов после нормализации — сведение вариантов
написания одной и той же реальной сущности через общий якорь (адрес для
клиента, клиент для адреса), без обращения к LLM.

Мотивация (см. обсуждение в чате): построчная нормализация client_raw/
address_raw независима по каждому уникальному сырому значению — один и тот
же реальный клиент/точка может получить разные normalized-варианты, если
сырой текст отличался (например "Иванов ИИ" и "ИП Иванов Иван Иванович"),
из-за чего point_key дробится на несколько записей для одной физической
точки. Кэш это не лечит — он хранит перевод по значению, без знания о
совместной встречаемости client_raw/address_raw в одной строке.

Реконсиляция работает НАД уже собранным DataFrame (после нормализации,
до сохранения в parquet), полностью детерминированно, без LLM:
  1. reconcile_clients  — группировка по "отпечатку" адреса (город
     отброшен, т.к. может отсутствовать), слияние клиентов внутри группы,
     если совпадение уверенное (высокий fuzzy score).
  2. reconcile_addresses — группировка по (уже сведённому) клиенту,
     подтягивание города там, где он есть у той же связки клиент+дом.
  3. recompute_point_key — пересчёт point_key одним вложенным проходом
     после того как оба поля выше уже финальны.

Кэш (core/data/fetcher.py: self.cache) НЕ модифицируется и не читается
этим модулем — реконсиляция стейтлесс и пересчитывается заново при каждом
полном прогоне fetch_all(), поэтому старые записи "самоисцеляются", когда
в новых данных появляется более полный/убедительный вариант написания.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
from loguru import logger

# Порог уверенности для слияния клиентов внутри одной адресной группы.
# Группа уже отфильтрована по совпадающему дому — это сильный априорный
# сигнал, поэтому порог ниже, чем для слоя категорий (там сравнение шло
# без такого якоря вообще).
CLIENT_MERGE_THRESHOLD = 85

# Доп. защита от двух классов ложных слияний, найденных на реальных данных
# (см. обсуждение в чате):
# 1) "ПивКо" сливалось с "Пивко Инвест"/"Пивко Франшиза"/любым другим именем,
#    содержащим слово "пивко" — token_set_ratio даёт 100, если короткая
#    строка целиком содержится в длинной, независимо от того, один ли это
#    реальный клиент. MIN_LENGTH_RATIO отсекает слияние, если одна из строк
#    существенно короче другой (общее/брендовое имя без остальных деталей).
# 2) "Кирьянова Валентина Николаевна..." сливалось с "Кирьянова Галина
#    Михайловна..." — score 87, потому что все слова кроме имени совпадают,
#    а token_set_ratio не различает "одно расходящееся слово" и "просто
#    более полная запись". CONTENT_WORD_MIN_LEN — если после вычитания
#    общих токенов с ОБЕИХ сторон остаётся значимое слово (не короткая
#    метка вроде "проф"/"бывш") — это, вероятно, разные люди, не сливаем.
MIN_LENGTH_RATIO = 0.5
CONTENT_WORD_MIN_LEN = 5

_TOKEN_RE = re.compile(r'[^0-9a-zа-яё\s]')


def _tokenize(s: str) -> set:
    return set(_TOKEN_RE.sub(' ', s.lower()).split())


def _safe_to_merge(a: str, b: str) -> bool:
    """Доп. проверки поверх token_set_ratio — см. комментарий у констант выше."""
    length_ratio = min(len(a), len(b)) / max(len(a), len(b))
    if length_ratio < MIN_LENGTH_RATIO:
        return False

    only_a = _tokenize(a) - _tokenize(b)
    only_b = _tokenize(b) - _tokenize(a)
    two_way_diff = bool(only_a) and bool(only_b)
    if two_way_diff:
        content_conflict = (
            any(len(w) >= CONTENT_WORD_MIN_LEN for w in only_a)
            and any(len(w) >= CONTENT_WORD_MIN_LEN for w in only_b)
        )
        if content_conflict:
            return False

    return True

STREET_MARKER_RE = re.compile(
    r'(?i)\b(ул\.?|улица|пр-?кт|проспект|мкр\.?|мкрн|переулок|пер\.|шоссе|бульвар|наб\.|тракт)\b|^\d'
)


def has_city_prefix(address_normalized: str) -> bool:
    """
    Эвристика: первый сегмент до запятой похож на улицу (содержит маркер
    типа "ул."/"пр-кт" или начинается с цифры) -> города в адресе нет.
    Иначе считаем, что первый сегмент - это город/населённый пункт.
    """
    if not isinstance(address_normalized, str) or ',' not in address_normalized:
        return False
    first_seg = address_normalized.split(',')[0].strip()
    if not first_seg:
        return False
    return not STREET_MARKER_RE.search(first_seg)


def address_fingerprint(address_normalized: str) -> str:
    """
    "Отпечаток" адреса без города: улица+дом, приведённые к нижнему
    регистру, без пунктуации/пробелов. Один и тот же дом с городом и без
    города даёт один и тот же отпечаток - это и есть якорь для связки.
    Пустая строка - если исходное значение пустое или это не похоже на
    адрес (например "офис").
    """
    if not isinstance(address_normalized, str) or not address_normalized.strip():
        return ""

    parts = [p.strip() for p in address_normalized.split(',')]
    if has_city_prefix(address_normalized) and len(parts) > 1:
        rest = ','.join(parts[1:])
    else:
        rest = address_normalized

    fp = re.sub(r'[^0-9a-zа-яё]', '', rest.lower())
    return fp


def _cluster_by_similarity(names: List[str], threshold: int) -> List[List[str]]:
    """
    Группирует имена в кластеры через связные компоненты графа, где ребро
    между двумя именами есть, если token_set_ratio >= threshold. Простая
    union-find на маленьких группах (внутри одной адресной группы имён
    единицы, не тысячи - O(n^2) тут не проблема).
    """
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if fuzz.token_set_ratio(names[i], names[j]) >= threshold and _safe_to_merge(names[i], names[j]):
                union(i, j)

    clusters: Dict[int, List[str]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(names[i])
    return list(clusters.values())


def reconcile_clients(
    df: pd.DataFrame,
    threshold: int = CLIENT_MERGE_THRESHOLD,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """
    Группирует строки по отпечатку адреса, внутри группы сливает похожие
    варианты client_normalized к одному каноническому (самому длинному —
    как правило, более полная форма содержит больше информации, чем
    сокращённая). Возвращает обновлённый df и лог слияний для отчёта.
    """
    if 'address_normalized' not in df.columns or 'client_normalized' not in df.columns:
        return df, []

    df = df.copy()
    df['_addr_fp'] = df['address_normalized'].apply(address_fingerprint)

    merge_log: List[Dict[str, object]] = []
    rename_map: Dict[str, str] = {}

    grouped = df[df['_addr_fp'] != ''].groupby('_addr_fp')['client_normalized']
    for fp, series in grouped:
        distinct = sorted(set(v for v in series.dropna().unique() if v))
        if len(distinct) < 2:
            continue

        clusters = _cluster_by_similarity(distinct, threshold)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = max(cluster, key=len)
            for name in cluster:
                if name != canonical:
                    rename_map[name] = canonical
            merge_log.append({
                'addr_fingerprint': fp,
                'variants': cluster,
                'canonical': canonical,
            })

    if rename_map:
        df['client_normalized'] = df['client_normalized'].replace(rename_map)
        logger.info(f"🔗 Реконсиляция клиентов: слито групп — {len(merge_log)}, "
                    f"переименовано вариантов — {len(rename_map)}")

    df = df.drop(columns=['_addr_fp'])
    return df, merge_log


def reconcile_addresses(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """
    Группирует строки по (уже сведённому) client_normalized и отпечатку
    адреса. Если внутри такой группы есть хотя бы один адрес с городом и
    хотя бы один без — подтягивает город с city-варианта на без-city
    варианты. Работает только внутри одного клиента и одного дома — не
    трогает случаи, где отпечаток совпал у РАЗНЫХ клиентов (см. пример
    Аделя Кутуя: разные дома у разных клиентов на одной улице отпечаток
    не совпадёт, а если бы совпал - разный клиент не даст скрестить город).
    """
    required = {'address_normalized', 'client_normalized'}
    if not required.issubset(df.columns):
        return df, []

    df = df.copy()
    df['_addr_fp'] = df['address_normalized'].apply(address_fingerprint)
    df['_has_city'] = df['address_normalized'].apply(has_city_prefix)

    backfill_log: List[Dict[str, object]] = []
    value_map: Dict[Tuple, str] = {}

    key_cols = ['client_normalized', '_addr_fp']
    grouped = df[df['_addr_fp'] != ''].groupby(key_cols)

    for (client, fp), group in grouped:
        with_city = group.loc[group['_has_city'], 'address_normalized']
        without_city = group.loc[~group['_has_city'], 'address_normalized']
        if with_city.empty or without_city.empty:
            continue

        canonical_addr = with_city.mode().iloc[0]  # самый частый city-вариант в группе
        for raw_variant in without_city.unique():
            value_map[(client, raw_variant)] = canonical_addr
            backfill_log.append({
                'client': client,
                'from': raw_variant,
                'to': canonical_addr,
            })

    if value_map:
        def _apply(row):
            key = (row['client_normalized'], row['address_normalized'])
            return value_map.get(key, row['address_normalized'])

        df['address_normalized'] = df.apply(_apply, axis=1)
        logger.info(f"🔗 Реконсиляция адресов: подтянут город для {len(backfill_log)} вариантов")

    df = df.drop(columns=['_addr_fp', '_has_city'])
    return df, backfill_log


def recompute_point_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Векторизованный пересчёт point_key после реконсиляции client_normalized
    и address_normalized. Логика повторяет прежнюю построчную версию из
    fetcher._process_sheet (client | address, с фоллбэком на одно из полей).
    """
    df = df.copy()
    client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('')
    addr = df.get('address_normalized', pd.Series('', index=df.index)).fillna('')

    both = (client != '') & (addr != '')
    only_client = (client != '') & (addr == '')
    only_addr = (client == '') & (addr != '')

    point_key = pd.Series(None, index=df.index, dtype=object)
    point_key[both] = client[both] + ' | ' + addr[both]
    point_key[only_client] = client[only_client]
    point_key[only_addr] = addr[only_addr]

    df['point_key'] = point_key
    return df


def reconcile(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Точка входа: прогоняет все три шага и собирает отчёт для логов/отладки."""
    df, client_merges = reconcile_clients(df)
    df, address_backfills = reconcile_addresses(df)
    df = recompute_point_key(df)

    report = {
        'client_merges': client_merges,
        'address_backfills': address_backfills,
    }
    return df, report
