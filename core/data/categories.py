"""
Менеджер категорий обращений (config/categories.json).

Отвечает только за категории — не знает ничего про сленг/аббревиатуры
(этим занимается Glossary) и ничего не знает про LLM-вызовы (этим
занимается DeepSeekClient). Используется:
- fetcher.py — слой 1 (fuzzy-матч сырого текста с названиями категорий)
  и подготовка данных для слоя 2 (текст категорий + список названий для enum)
- в перспективе — Streamlit-UI для редактирования categories.json

Формат categories.json:
{
  "categories": [
    {"name": "Название категории", "description": "Пояснение (опционально)"},
    ...
  ]
}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


# Путь захардкожен по /app/, как и у остальных файлов core/data/
# (fetcher.py, indexer.py, loader.py, glossary.py) — это реальная раскладка
# внутри Docker-контейнера (WORKDIR /app, config смонтирован в /app/config,
# см. docker-compose.yml), а не относительный путь от текущей директории.
DEFAULT_CATEGORIES_PATH = Path("/app/config/categories.json")


class CategoriesManager:
    """Читает categories.json и готовит данные для слоя 1 и слоя 2."""

    def __init__(self, path: Path = DEFAULT_CATEGORIES_PATH):
        self.path = Path(path)
        self.categories: List[Dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        """
        Перечитывает categories.json с диска. Вынесено в отдельный метод
        (по аналогии с Glossary.reload()) — понадобится для будущего UI
        редактирования категорий без перезапуска процесса.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Файл категорий не найден: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"Файл {self.path} не содержит непустого списка 'categories'")

        # Лёгкая валидация + проверка на дубликаты имён (важно: имя категории
        # используется и как enum-значение в function calling, и как значение
        # тега в данных — дубликаты незаметно всё сломают)
        seen_names = set()
        validated = []
        for entry in categories:
            name = entry.get("name")
            if not name or not isinstance(name, str):
                logger.warning(f"⚠️ Пропущена категория без корректного 'name': {entry}")
                continue
            if name in seen_names:
                logger.warning(f"⚠️ Дублирующееся имя категории в categories.json: '{name}'")
                continue
            seen_names.add(name)
            validated.append({
                "name": name,
                "description": entry.get("description", "").strip() if entry.get("description") else ""
            })

        self.categories = validated
        logger.info(f"📦 Загружено категорий: {len(self.categories)} из {self.path}")

    def get_category_names(self) -> List[str]:
        """
        Список названий категорий — используется:
        - в слое 1 (fuzzy-сравнение сырого текста обращения с названиями)
        - как enum для function calling в слое 2 (модель не может вернуть
          значение, которого нет в этом списке)
        """
        return [c["name"] for c in self.categories]

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Полный список категорий (name + description) — пригодится для
        будущего Streamlit-UI (st.data_editor и т.п.), без него пришлось бы
        читать JSON заново в UI-коде.
        """
        return list(self.categories)

    def get_description(self, name: str) -> Optional[str]:
        """Пояснение к конкретной категории по точному имени, если нужно отдельно."""
        for c in self.categories:
            if c["name"] == name:
                return c["description"] or None
        return None

    def to_prompt_text(self) -> str:
        """
        Развёрнутый текст всех категорий с пояснениями — вставляется в промпт
        LLM в слое 2 (description инструмента function calling), чтобы модель
        могла дизамбигуировать похожие категории (например "Сканер" vs
        "Разрешительный режим"). Сам enum (что можно вернуть) задаётся
        отдельно через get_category_names() — этот текст только поясняет,
        какую категорию когда выбирать.

        Категории без description выводятся без пояснения.
        """
        lines = []
        for c in self.categories:
            if c["description"]:
                lines.append(f"- {c['name']} — {c['description']}")
            else:
                lines.append(f"- {c['name']}")
        return "\n".join(lines)