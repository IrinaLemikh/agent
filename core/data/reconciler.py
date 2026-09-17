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
  3. reconcile_point_names — подтягивание названия точки от соседа по дому
     и сведение написаний к самому частому.
  4. canonicalize_point_fields — внутри одной точки одно написание клиента
     и адреса, чтобы разнобой регистра не рвал её на несколько ключей.
  5. recompute_point_key — пересчёт point_key одним вложенным проходом
     после того как все поля выше уже финальны.

Кэш (core/data/fetcher.py: self.cache) НЕ модифицируется и не читается
этим модулем — реконсиляция стейтлесс и пересчитывается заново при каждом
полном прогоне fetch_all(), поэтому старые записи "самоисцеляются", когда
в новых данных появляется более полный/убедительный вариант написания.
"""

import collections
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
# 4, а не 5: на пороге 5 сливались "ИП Гольдин Никита Александрович" и
# "ИП Гольдин Юрий Александрович" — разные люди по одному адресу, потому что
# "Юрий" короче пяти букв и не считался значимым словом. Проверено на данных:
# понижение до 4 убирает это слияние и не отменяет ни одного правильного.
CONTENT_WORD_MIN_LEN = 4

_TOKEN_RE = re.compile(r'[^0-9a-zа-яё\s]')


def _tokenize(s: str) -> set:
    return set(_TOKEN_RE.sub(' ', s.lower()).split())


_LEGAL_FORM_RE = re.compile(r'(?i)\b(ип|ооо|зао|оао|пао|ао|нко|ано)\b')
_LEGAL_FORM_FRONT_RE = re.compile(r'(?i)^\s*(ип|ооо|зао|оао|пао|ао|нко|ано)\b')


def _legal_forms(s: str) -> set:
    return set(m.lower() for m in _LEGAL_FORM_RE.findall(s))


def _pick_canonical(cluster: List[str]) -> str:
    """
    Канонический вариант из кластера: сначала предпочитаем записи, где
    правовая форма стоит ПЕРЕД названием ("ООО ПВ Ритейл", а не
    "ПВ РИТЕЙЛ ООО") — одна позиция на весь датасет; среди них берём самый
    длинный, как более полную запись.
    """
    front = [name for name in cluster if _LEGAL_FORM_FRONT_RE.match(name)]
    return max(front or cluster, key=len)


def _safe_to_merge(a: str, b: str) -> bool:
    """Доп. проверки поверх token_set_ratio — см. комментарий у констант выше."""
    # Разные правовые формы — разные юрлица, как бы похожи ни были названия.
    # Найдено на реальных данных: "ООО БирЛайт (Тагильское пиво)" сливалось
    # с "ИП Голосова Наталья Витальевна (Тагильское пиво БирЛайт)", и 52
    # обращения уезжали на чужое юрлицо. _safe_to_merge это не ловил, потому
    # что "ООО" короче CONTENT_WORD_MIN_LEN.
    forms_a, forms_b = _legal_forms(a), _legal_forms(b)
    if forms_a and forms_b and forms_a != forms_b:
        return False

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

# Город последнего рубежа: ставится только если город не удалось взять ни у
# соседа по тому же дому, ни у того же клиента (см. reconcile_addresses).
DEFAULT_CITY = "Екатеринбург"


def _norm_key(s: str) -> str:
    """
    Ключ сравнения строк: без регистра, пунктуации и пробелов.

    "ё" приводится к "е". В источнике одно и то же название приходит и так и
    так — "Артёмовский" в 22 строках, "Артемовский" в 78, — а для опознания
    это одна и та же буква. Ключ нигде не показывается, поэтому на отчёты это
    не влияет: в них по-прежнему побеждает самое частое написание.
    """
    return re.sub(r'[^0-9a-zа-я]', '', str(s).lower().replace('ё', 'е'))


# Слова-обозначения выбрасываются при сравнении адресов — где бы они ни
# стояли. Только для сопоставления: в самой строке адреса они остаются, это
# невидимый ключ.
#
# Зачем: один и тот же микрорайон в данных записан семью способами — "мкр",
# "мкр.", "мкрн", "мкрн.", "м-н", "мик-н", "микрорайон", причём 424 раза тип
# стоит перед названием и 254 раза после. Плюс часть адресов обходится вовсе
# без типа ("Екатеринбург, Светлый 7"). Промпт просит единую форму, но
# полагаться только на него нельзя: отпечаток не должен зависеть от того,
# послушалась ли модель. Выбросив тип, "мкр Светлый, д 2", "Светлый м-н 2" и
# "Светлый 2" дают один ключ.
#
# Потеря смысла здесь не страшна ("4 мкрн" превращается в "4"), потому что
# ключ никому не показывается, а сравнивается всегда внутри одного клиента и
# одного города.
_ADDRESS_TYPE_WORDS = {
    'ул', 'улица', 'мкр', 'мкрн', 'микрорайон', 'мн', 'микн', 'квл', 'квартал',
    'пер', 'переулок', 'пркт', 'просп', 'проспект', 'ш', 'шоссе', 'бр', 'бульвар',
    'наб', 'набережная', 'проезд', 'тракт', 'аллея', 'линия',
    'д', 'дом', 'влд', 'двлд', 'зд', 'здание', 'уч', 'участок',
    'корп', 'корпус', 'стр', 'строение',
    'кв', 'квартира', 'пом', 'помещ', 'помещение', 'оф', 'офис', 'эт', 'этаж',
}


def address_fingerprint(address_normalized: str, city: str = "") -> str:
    """
    "Отпечаток" адреса без города (улица+дом) - якорь "тот же дом". Один и
    тот же дом, записанный с городом и без, даёт один отпечаток.

    Город приходит отдельной колонкой address_city и отрезается буквально.
    Раньше здесь была эвристика has_city_prefix, которая гадала по виду
    строки, город перед нами или улица: "Агрономическая, 28" она принимала
    за город и схлопывала отпечаток до "28", склеивая разные улицы.
    """
    if not isinstance(address_normalized, str) or not address_normalized.strip():
        return ""

    rest = address_normalized.strip()
    if city:
        prefix = f"{city},"
        if rest.lower().startswith(prefix.lower()):
            rest = rest[len(prefix):]

    parts = [_norm_key(token) for token in re.split(r'[\s,]+', rest)]
    return ''.join(p for p in parts if p and p not in _ADDRESS_TYPE_WORDS)


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
    city_col = df.get('address_city', pd.Series('', index=df.index)).fillna('').astype(str)
    df['_addr_fp'] = [
        address_fingerprint(a, c)
        for a, c in zip(df['address_normalized'].fillna('').astype(str), city_col)
    ]

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
            canonical = _pick_canonical(cluster)
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
    Делает две вещи над адресами.

    1. ЗАПОЛНЯЕТ ГОРОД там, где его нет, четырьмя ступенями по убыванию
       надёжности: сосед по тому же дому у того же клиента -> единственный
       город этого клиента -> город, упомянутый в названии точки ->
       DEFAULT_CITY. Дефолт стоит последним осознанно: раньше он
       подставлялся ещё на этапе промпта и перебивал настоящие города
       (точка во Владивостоке становилась екатеринбургской), а теперь
       применяется только когда взять город больше неоткуда.

       Ступень с названием точки — именно ЗАПАСНАЯ, и только при пустом
       адресе. Проверено на данных: там, где город известен и из адреса, и
       из названия, они расходятся в 39 случаях из 157 — "ПРОФ Пивчик
       Сургут" стоит в Когалыме, "Проф Краснодар" в Яблоновском. Название
       точки — это имя территории франшизы, а не место точки, поэтому
       перебивать им настоящий адрес нельзя. Но когда адреса нет вовсе,
       "Севастополь" из названия всё равно лучше слепого Екатеринбурга.

       Адреса без цифр (плейсхолдеры вроде "офис") города НЕ получают —
       "Екатеринбург, офис" был бы выдумкой, а "офис" остаётся точкой как есть.

    2. ПРИВОДИТ НАПИСАНИЕ к одному виду внутри связки (клиент, дом, город):
       "Трактовая, 1Ж" и "Трактовая, 1ж", "Дм. Неаполитанова" и
       "Дм.Неаполитанова", "ОФИС"/"Офис"/"офис" — это одна точка.
       Город входит в ключ группировки намеренно: одна улица и дом в разных
       городах (Приданниково и Тюмень, ул. Пограничников, 1) — разные точки,
       сливать их нельзя.
    """
    required = {'address_normalized', 'client_normalized'}
    if not required.issubset(df.columns):
        return df, []

    df = df.copy()
    addr = df['address_normalized'].fillna('').astype(str).str.strip()
    client = df['client_normalized'].fillna('').astype(str).str.strip()
    city = df.get('address_city', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()

    # ---- шаг 0: единое написание города ----
    # "Ростов на Дону" (70 строк) и "Ростов-на-Дону" (217) — один город, но
    # город входит в ключ группировки, и без сведения одна точка разъезжается
    # на две. Сравниваем без регистра и пунктуации, показываем самое частое.
    variants: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for value in city:
        if value:
            variants[_norm_key(value)][value] += 1
    canonical_city = {key: counter.most_common(1)[0][0] for key, counter in variants.items()}

    cities_unified = 0
    for idx in df.index:
        old = city[idx]
        if not old:
            continue
        new = canonical_city.get(_norm_key(old), old)
        if new != old:
            if addr[idx].startswith(old):
                addr[idx] = new + addr[idx][len(old):]
            city[idx] = new
            cities_unified += 1

    fp = pd.Series(
        [address_fingerprint(a, c) for a, c in zip(addr, city)],
        index=df.index,
    )

    log: List[Dict[str, object]] = []

    # ---- шаг 1: заполнение города ----
    known = pd.DataFrame({'client': client, 'fp': fp, 'city': city})
    known = known[known['city'] != '']
    by_house = (
        known.groupby(['client', 'fp'])['city']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    by_client = known.groupby('client')['city'].unique().to_dict()

    # Справочник городов собираем из самих данных — всё, что хоть раз
    # встретилось городом в адресе. Хардкодить список городов не нужно.
    city_vocab = {_norm_key(c): c for c in known['city'].unique()}
    point_names = df.get('point_name', pd.Series('', index=df.index)).fillna('').astype(str)

    def city_from_point_name(name: str):
        for token in name.split():
            found = city_vocab.get(_norm_key(token))
            if found:
                return found
        return None

    need_city = (city == '') & (addr != '') & addr.str.contains(r'\d', regex=True)

    for idx in df.index[need_city]:
        c, f = client[idx], fp[idx]

        source, origin = by_house.get((c, f)), 'дом'
        if source is None:
            candidates = by_client.get(c, [])
            source, origin = (candidates[0], 'клиент') if len(candidates) == 1 else (None, '')
        if source is None:
            source, origin = city_from_point_name(point_names[idx]), 'название'
        if source is None:
            source, origin = DEFAULT_CITY, 'дефолт'

        city[idx] = source
        addr[idx] = f"{source}, {addr[idx]}"
        log.append({'client': c, 'address': addr[idx], 'city': source, 'origin': origin})

    # Сведение написания адреса раньше жило здесь и группировало по точному
    # тексту клиента. Переехало ниже, в canonicalize_point_fields: там ключ
    # уже окончательный (после реконсиляции названий точек), поэтому видны и
    # строки с пустым клиентом, где опознаётся point_name, и разнобой в самом
    # клиенте.
    df['address_normalized'] = addr
    df['address_city'] = city

    by_origin = collections.Counter(item['origin'] for item in log)
    logger.info(
        f"🔗 Реконсиляция адресов: город проставлен для {len(log)} строк "
        f"(от соседа по дому — {by_origin['дом']}, от клиента — {by_origin['клиент']}, "
        f"из названия точки — {by_origin['название']}, дефолт — {by_origin['дефолт']}); "
        f"написание города сведено у {cities_unified} строк"
    )
    return df, log


def reconcile_point_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """
    Делает две вещи, каждая из которых требует ВСЕГО набора строк (потому и
    живёт здесь, а не в client._expand_point_name, где виден один батч).

    1. ПОДТЯГИВАЕТ НАЗВАНИЕ от соседа: если у пары (клиент, дом) в одних
       строках точка названа, а в других пусто — ставим самое частое имя из
       этой же пары. Тот же приём, что с городом. Особенно важно для строк
       с пустым клиентом: там именно point_name служит опознавательной
       частью point_key, и без названия точка сливается в голый адрес.

    2. СВОДИТ НАПИСАНИЕ к самому частому варианту среди отличающихся только
       регистром и пробелами: "КиноДомино" и "Кино Домино", "БЕЛОРУССКИЕ
       ПРОДУКТЫ" и "Белорусские продукты".
    """
    if 'point_name' not in df.columns:
        return df, []

    df = df.copy()
    before = df['point_name'].fillna('').astype(str).str.strip()
    names = before.copy()

    # ---- шаг 1: подтягивание от соседа по (клиент, дом) ----
    if 'address_normalized' in df.columns:
        client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
        addr = df['address_normalized'].fillna('').astype(str).str.strip()
        city = df.get('address_city', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
        fp = pd.Series([address_fingerprint(a, c) for a, c in zip(addr, city)], index=df.index)

        known = pd.DataFrame({'client': client, 'fp': fp, 'name': names})
        known = known[(known['name'] != '') & (known['fp'] != '')]
        by_house = (
            known.groupby(['client', 'fp'])['name']
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
        for idx in df.index[(names == '') & (fp != '')]:
            source = by_house.get((client[idx], fp[idx]))
            if source:
                names[idx] = source

    # ---- шаг 2: единое написание ----
    frame = pd.DataFrame({'name': names, 'key': names.apply(_norm_key)})
    best = (
        frame[frame['name'] != '']
        .groupby('key')['name']
        .agg(lambda s: s.value_counts().idxmax())
    )
    after = frame.apply(lambda row: best.get(row['key'], row['name']), axis=1)

    changes = [
        {'from': b, 'to': a}
        for b, a in zip(before, after) if b != a
    ]
    df['point_name'] = after

    if changes:
        logger.info(f"🔗 Реконсиляция названий точек: приведено к канону {len(changes)} строк")
    return df, changes


def reconcile_client_point_roles(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Сводит случаи, когда одна и та же контора записана то именем владельца,
    то вывеской — в разных строках по-разному.

    В данных это выглядит так (см. обсуждение в чате):

        ООО БирЛайт        + точка "Тагильское пиво"   299 строк
        "Тагильское пиво"  как сам клиент                5 строк

    Для ключа это два разных хозяина, и один магазин попадает в отчёт
    дважды. Приём тот же, что и везде: спросить у данных, но только когда
    ответ однозначен. Условия срабатывания:

      • название точки где-то в данных стоит клиентом само по себе;
      • у этого названия среди строк с заполненным клиентом ровно ОДИН
        владелец (иначе это сеть — "ПивКо" стоит точкой у сотни клиентов,
        и приравнивать их друг к другу нельзя).

    Канон выбирается по частоте, как и остальные написания: побеждает тот
    вариант, которым чаще подписан клиент. Это заодно развязывает взаимные
    пары — "Скутин" стоит точкой у клиента "САДКО", а "САДКО" точкой у
    клиента "Скутин"; обе пары приводят к одному и тому же победителю.

    Прежнее написание не теряется: оно уходит в point_name, если тот пуст.
    """
    df = df.copy()
    client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    point_name = df.get('point_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()

    named = pd.DataFrame({'client': client, 'point_name': point_name})
    named = named[(named['client'] != '') & (named['point_name'] != '')]
    owners = named.groupby('point_name')['client'].agg(lambda s: {_norm_key(x): x for x in s})

    as_client = collections.Counter(_norm_key(c) for c in client if c)

    mapping: Dict[str, str] = {}
    for name, owner_map in owners.items():
        if len(owner_map) != 1:
            continue  # у названия несколько владельцев — это сеть, не псевдоним
        owner = next(iter(owner_map.values()))
        name_key, owner_key = _norm_key(name), _norm_key(owner)
        if name_key == owner_key or name_key not in as_client:
            continue  # название нигде не выступает клиентом само по себе
        loser, winner = ((owner, name) if as_client[name_key] >= as_client[owner_key]
                         else (name, owner))
        mapping[_norm_key(loser)] = winner

    changes: List[Dict[str, str]] = []
    for idx in df.index[client != '']:
        winner = mapping.get(_norm_key(client[idx]))
        if not winner or winner == client[idx]:
            continue
        changes.append({'from': client[idx], 'to': winner})
        if not point_name[idx]:
            df.at[idx, 'point_name'] = client[idx]
        df.at[idx, 'client_normalized'] = winner

    if changes:
        pairs = sorted({(c['from'], c['to']) for c in changes})
        logger.info(f"🔗 Клиент и точка поменялись ролями: сведено {len(changes)} строк, "
                    f"пар — {len(pairs)}: {'; '.join(f'{a} -> {b}' for a, b in pairs[:5])}")
    return df, changes


def backfill_identity(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Подписывает строки, где не заполнено НИ клиента, ни названия точки, но
    адрес известен и у этого дома в остальных строках ровно один хозяин.

    Зачем (см. обсуждение в чате): оператор заполняет клиента не всегда, а
    адрес пишет тем же текстом. В сырых данных это видно буквально:

        (пусто)                       | г. Ростов на Дону, ул. Каскадная 164   7 строк
        ИП Белокопытов А.Н. (Проф ...)| г. Ростов на Дону, ул. Каскадная 164  51 строка

    Без подписи такие строки становятся отдельной точкой "голый адрес", и
    один магазин попадает в отчёт дважды. Замер: 272 строки, ключей
    3168 -> 3015.

    Это тот же приём, что уже применён к городу и к названию точки: спросить
    у соседа по дому, но только если сосед ОДИН. Там, где у дома несколько
    разных хозяев (203 строки — в одном здании правда сидят разные
    арендаторы), не трогаем ничего: угадывать, к кому из них относится
    обращение, нечем.

    Дом опознаётся по (город, отпечаток адреса) — тому же ключу, что и
    везде, поэтому "Каскадная 164" и "ул. Каскадная, д. 164" считаются одним
    домом, а одинаковые улицы в разных городах не путаются.
    """
    df = df.copy()
    client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    point_name = df.get('point_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    addr = df.get('address_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    city = df.get('address_city', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()

    identity = client.where(client != '', point_name)
    house = pd.Series(
        [f'{_norm_key(c)}|{address_fingerprint(a, c)}' for c, a in zip(city, addr)],
        index=df.index,
    )

    # хозяева дома: только строки, где адрес есть и подпись есть
    named = pd.DataFrame({'house': house, 'identity': identity,
                          'client': client, 'point_name': point_name})
    named = named[(named['identity'] != '') & (addr != '')]
    owners = named.groupby('house')['identity'].agg(lambda s: set(s))
    # у соседей по дому подпись стоит в клиенте или только в названии точки —
    # в ту же колонку пишем и мы, иначе точка опознавалась бы иначе, чем соседи
    by_client = named.groupby('house')['client'].agg(lambda s: (s != '').all())

    log: List[Dict[str, str]] = []
    for idx in df.index[(identity == '') & (addr != '')]:
        candidates = owners.get(house[idx])
        if not candidates or len(candidates) != 1:
            continue  # хозяина нет вовсе или их несколько — не гадаем
        owner = next(iter(candidates))
        column = 'client_normalized' if by_client[house[idx]] else 'point_name'
        df.at[idx, column] = owner
        log.append({'address': addr[idx], 'owner': owner})

    if log:
        logger.info(f"🔗 Подпись безымянных строк: {len(log)} строк отнесено к "
                    f"единственному хозяину дома")
    return df, log


def canonicalize_point_fields(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Последний шаг перед сборкой ключа: внутри одной физической точки даёт всем
    строкам ОДНО написание клиента и адреса.

    Зачем (см. обсуждение в чате): point_key склеивается из отображаемых строк
    дословно, поэтому любой разнобой написания рвёт точку на несколько ключей —
    и это видно в отчётах, а не только внутри. Замер на реальных данных: 14
    точек разбито надвое, например "ПивКо | Березовский, Мира, 2А" (5 строк) и
    "ПивКо | Березовский, Мира, 2а" (15 строк) — две строки в топе точек вместо
    одной. Канон выбирается по частоте: тот же приём, что уже применён к городу
    и названию точки.

    Ключ группировки — (личность, город, отпечаток адреса). Про каждую часть:

    • ЛИЧНОСТЬ, а не клиент: у строк с пустым клиентом опознавательная часть
      ключа — point_name, ровно как в recompute_point_key ниже.

    • ГОРОД обязателен, хотя отпечаток и так не зависит от написания улицы.
      Отпечаток отрезает город буквально, и без него "Ленина, 33А" в Кашино
      слилась бы с "Ленина, 33а" в Хабаровске — улица Ленина есть везде.
      Замер: без города ложно слиплись бы 10 групп. Пустой город при этом не
      мешает: настоящий адрес его всегда получает (последняя ступень лестницы
      в reconcile_addresses — дефолт), а пустым он остаётся только у строк
      вроде "офис", которым и правильно группироваться между собой.

    ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕ ДЕЛАЕТСЯ: пустой адрес не заполняется от соседей.
    Проверено на данных — у таких строк пуст сам address_raw, так что подтянуть
    им "офис" значило бы выдумать. Прочерк должен означать "адрес неизвестен",
    а не "наверное, офис": иначе в топе точек одной строкой смешаются
    обращения из головного офиса и обращения непонятно откуда.
    """
    df = df.copy()
    client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    point_name = df.get('point_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    addr = df.get('address_normalized', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    city = df.get('address_city', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()

    def _canonical(values: pd.Series, keys: pd.Series) -> pd.Series:
        """Каждому непустому значению — самый частый вариант с тем же ключом.
        Пустые не участвуют ни как кандидаты, ни как получатели."""
        frame = pd.DataFrame({'value': values, 'key': keys})
        filled = frame[frame['value'] != '']
        best = filled.groupby('key')['value'].agg(lambda s: s.value_counts().idxmax())
        return frame.apply(
            lambda row: best.get(row['key'], row['value']) if row['value'] else row['value'],
            axis=1,
        )

    # ---- 1. клиент: варианты, отличающиеся только регистром и пунктуацией ----
    # 'ООО ИНТЕР МК-УРАЛ' и 'ООО "ИНТЕР МК-УРАЛ"', 'БЕЛОРУССКИЕ ПРОДУКТЫ' и
    # 'Белорусские продукты'. Слияние непохожих имён — задача reconcile_clients,
    # здесь только написание.
    new_client = _canonical(client, client.map(_norm_key))

    # ---- 2. адрес: варианты внутри одной точки ----
    identity = new_client.where(new_client != '', point_name)
    group_key = pd.Series(
        [f'{_norm_key(i)}|{_norm_key(c)}|{address_fingerprint(a, c)}'
         for i, c, a in zip(identity, city, addr)],
        index=df.index,
    )
    new_addr = _canonical(addr, group_key)

    stats = {
        'clients': int((new_client != client).sum()),
        'addresses': int((new_addr != addr).sum()),
    }
    df['client_normalized'] = new_client
    df['address_normalized'] = new_addr

    if stats['clients'] or stats['addresses']:
        logger.info(
            f"🔗 Канонизация перед ключом: написание клиента сведено у "
            f"{stats['clients']} строк, адреса — у {stats['addresses']}"
        )
    return df, stats


def recompute_point_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Векторизованный пересчёт point_key после реконсиляции client_normalized
    и address_normalized. Логика повторяет прежнюю построчную версию из
    fetcher._process_sheet (client | address, с фоллбэком на одно из полей).

    НОВОЕ (см. обсуждение в чате): если client_normalized пуст (стандартный
    случай для точек без юрлица в сырых данных, например "Пивко Франшиза" —
    это point_name, не клиент), в качестве "идентифицирующей" части ключа
    подставляется point_name вместо client_normalized. Иначе такие точки
    схлопывались бы в голый адрес и переставали различаться по бренду.
    Колонки client_normalized/point_name при этом НЕ модифицируются — это
    чистая сборка производной строки, а не реконсиляция/слияние.
    """
    df = df.copy()
    client = df.get('client_normalized', pd.Series('', index=df.index)).fillna('')
    point_name = df.get('point_name', pd.Series('', index=df.index)).fillna('')
    addr = df.get('address_normalized', pd.Series('', index=df.index)).fillna('')

    identity = client.where(client != '', point_name)

    both = (identity != '') & (addr != '')
    only_identity = (identity != '') & (addr == '')
    only_addr = (identity == '') & (addr != '')

    point_key = pd.Series(None, index=df.index, dtype=object)
    point_key[both] = identity[both] + ' | ' + addr[both]
    point_key[only_identity] = identity[only_identity]
    point_key[only_addr] = addr[only_addr]

    df['point_key'] = point_key
    return df


def reconcile(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Точка входа: прогоняет все шаги и собирает отчёт для логов/отладки."""
    df, client_merges = reconcile_clients(df)
    df, role_swaps = reconcile_client_point_roles(df)
    df, address_backfills = reconcile_addresses(df)
    df, identity_backfills = backfill_identity(df)
    df, point_name_changes = reconcile_point_names(df)
    df, canonical_stats = canonicalize_point_fields(df)
    df = recompute_point_key(df)

    report = {
        'client_merges': client_merges,
        'role_swaps': role_swaps,
        'address_backfills': address_backfills,
        'identity_backfills': identity_backfills,
        'canonical_stats': canonical_stats,
        'point_name_changes': point_name_changes,
    }
    return df, report
