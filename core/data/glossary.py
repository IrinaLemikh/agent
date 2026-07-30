"""
Словарь терминов и типовых кейсов техподдержки.

ОСНОВНОЙ режим использования (утверждено в архитектурном обсуждении):
  0. to_prompt_text()   — весь словарь целиком, плоским текстом, вставляется
                           в промпт LLM ВСЕГДА, вне зависимости от того,
                           понадобится ли он для конкретного обращения.
                           Модель не может надёжно решить сама, что ей нужно
                           искать в словаре — например, увидев "пробитие
                           маркировок" в тексте обращения, она не обязательно
                           сообразит, что нужно посмотреть термин "РР". Раз
                           в LLM попадает только небольшой "трудный остаток"
                           (не весь батч сразу), а сам словарь маленький
                           (порядка ~100 строк), раздувания контекста не будет.

Оставлены ПРО ЗАПАС (в текущей архитектуре НЕ используются — пригодятся,
если словарь вырастет настолько, что вставлять его целиком в промпт станет
неразумно, и придётся вернуться к модели, которая сама решает, что искать):
  1. lookup(query)     — точечный fuzzy-поиск по конкретному термину/фразе.
  2. list_terms()       — лёгкий "индекс": только названия терминов, без полных
                           определений. Нужен, когда модель встретила обращение,
                           которое в целом непонятно, но не может выделить
                           конкретное слово для точечного поиска — сначала
                           смотрит на список названий, потом делает lookup()
                           на то, что показалось релевантным.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from fuzzywuzzy import fuzz
from loguru import logger


DEFAULT_GLOSSARY_PATH = Path("/app/config/glossary.json")


class Glossary:
    """
    Словарь терминов и типовых кейсов с fuzzy-поиском по запросу.

    Формат записи в glossary.json:
    {
        "<термин или короткая фраза-ключ>": {
            "meaning": "краткая расшифровка/перевод термина",
            "case_description": "опционально: развёрнутое описание типового
                                  случая — контекст, который помогает не
                                  сделать неверный вывод",
            "aliases": ["опционально: синонимы/варианты написания для
                         улучшения fuzzy-совпадения"]
        }
    }

    Категории сюда намеренно НЕ входят — они живут в categories.json
    (см. core/data/categories.py), чтобы не дублировать одно и то же
    правило в двух местах.
    """

    def __init__(self, path: Path = DEFAULT_GLOSSARY_PATH):
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, Any]] = self._load()
        self._search_index: List[tuple] = self._build_search_index()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            logger.warning(
                f"⚠️ Файл словаря не найден: {self.path}. "
                f"Работаем с пустым словарём (LLM будет полагаться только на свои знания)."
            )
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"📖 Словарь загружен: {len(data)} записей из {self.path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка чтения словаря {self.path}: {e}")
            return {}

    def _build_search_index(self) -> List[tuple]:
        index = []
        for term, entry in self.entries.items():
            index.append((term, term))
            for alias in entry.get("aliases", []):
                index.append((alias, term))
        return index

    def reload(self):
        """Перечитать словарь с диска (например, после правки через веб-морду)."""
        self.entries = self._load()
        self._search_index = self._build_search_index()

    # =========================================================================
    # НОВОЕ: основной способ подачи словаря модели — весь текст сразу
    # =========================================================================
    def to_prompt_text(self) -> str:
        """
        Весь словарь целиком, плоским текстом, готовым для вставки в промпт
        LLM. Это ОСНОВНОЙ способ подачи словаря модели (см. шапку файла) —
        lookup()/list_terms()/тул-схемы ниже в этом сценарии не участвуют,
        они оставлены про запас.

        Формат одной строки:
            "- термин — значение (также: алиас1, алиас2). Уточнение: ..."
        Части "(также: ...)" и "Уточнение: ..." опускаются, если у записи
        нет aliases / case_description соответственно.

        Термины сортируются по названию — ради стабильности текста между
        запусками (полезно, если провайдер кэширует одинаковые префиксы
        промпта, и просто для воспроизводимости при отладке).
        """
        if not self.entries:
            return ""

        lines = []
        for term in sorted(self.entries.keys()):
            entry = self.entries[term]
            meaning = entry.get("meaning", "")
            line = f"- {term} — {meaning}" if meaning else f"- {term}"

            aliases = entry.get("aliases") or []
            if aliases:
                line += f" (также: {', '.join(aliases)})"

            case_description = entry.get("case_description")
            if case_description:
                line += f". Уточнение: {case_description}"

            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # ПРО ЗАПАС: точечный fuzzy-поиск и облегчённый индекс — не используются
    # в текущей архитектуре (см. шапку файла), не трогаем логику.
    # =========================================================================
    def lookup(self, query: str, top_k: int = 5, min_score: int = 50) -> List[Dict[str, Any]]:
        """
        Точечный поиск: ищет top_k записей, релевантных конкретному запросу.
        Используется, когда модель может назвать конкретное непонятное слово/фразу.

        Возвращает список словарей: term, meaning, case_description, score.
        Пустой список — если ничего не найдено выше min_score; в этом случае
        модели стоит попробовать list_terms() вместо повторного гадания.
        """
        if not query or not query.strip() or not self._search_index:
            return []

        query_clean = query.strip().lower()
        scored = []
        seen_terms = set()

        for search_key, original_term in self._search_index:
            score = fuzz.token_sort_ratio(query_clean, search_key.lower())
            if search_key.lower() in query_clean or query_clean in search_key.lower():
                score = max(score, 85)
            if score >= min_score:
                scored.append((score, original_term))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, term in scored:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            entry = self.entries[term]
            results.append({
                "term": term,
                "meaning": entry.get("meaning", ""),
                "case_description": entry.get("case_description"),
                "score": score,
            })
            if len(results) >= top_k:
                break

        if results:
            logger.debug(f"🔍 lookup('{query}') -> топ: {results[0]['term']} ({results[0]['score']})")
        else:
            logger.debug(f"🔍 lookup('{query}') -> ничего не найдено (min_score={min_score})")

        return results

    def lookup_many(self, queries: List[str], top_k: int = 5, min_score: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Batched-версия lookup — несколько терминов из одного батча за один tool call."""
        return {q: self.lookup(q, top_k=top_k, min_score=min_score) for q in queries}

    def list_terms(self) -> List[str]:
        """
        Лёгкий обзор: только названия терминов (без определений).
        Используется, когда обращение в целом непонятно, но нет конкретного
        слова для lookup() — модель смотрит на список названий и решает,
        по какому термину сделать точечный запрос.
        """
        terms = sorted(self.entries.keys())
        logger.debug(f"📋 list_terms() -> {len(terms)} терминов")
        return terms


# =============================================================================
# JSON-схемы тулов для function calling — ПРО ЗАПАС, не используются в
# текущей архитектуре (см. шапку файла)
# =============================================================================

LOOKUP_TERM_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_term",
        "description": (
            "Ищет значение конкретного термина, сокращения или типового случая "
            "в словаре техподдержки. Используй ВСЕГДА, когда встретил незнакомое "
            "сокращение/жаргон, ИЛИ когда не уверен, к какой категории отнести "
            "обращение, даже если все слова по отдельности понятны — в словаре "
            "могут быть уточнения по неочевидным случаям. НЕ угадывай значение "
            "самостоятельно, если есть хоть малейшее сомнение."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Один или несколько терминов/фраз для поиска, "
                        "например: ['дя', 'csi', 'английский']"
                    ),
                }
            },
            "required": ["queries"],
        },
    },
}

LIST_GLOSSARY_TERMS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_glossary_terms",
        "description": (
            "Возвращает список ВСЕХ названий терминов в словаре (без определений). "
            "Используй, если обращение в целом непонятно, но ты не можешь выделить "
            "конкретное слово для lookup_term — после просмотра списка вызови "
            "lookup_term на тот термин, который кажется релевантным."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


if __name__ == "__main__":
    import sys
    test_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GLOSSARY_PATH
    g = Glossary(path=test_path)
    print("Все термины:", g.list_terms())
    print("\n--- to_prompt_text() ---")
    print(g.to_prompt_text())
    for q in ["бс", "англ буквы", "не откр дя", "csi"]:
        print(f"\nЗапрос: '{q}'")
        for r in g.lookup(q):
            print(f"  [{r['score']}] {r['term']}: {r['meaning']}")