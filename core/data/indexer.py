"""
Индекс листов: хранит только названия таблиц и листов для UI.
При каждом обновлении полностью перезаписывается.
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass, asdict
import pandas as pd
from loguru import logger


@dataclass
class SheetMetadata:
    """Метаданные одного листа — только для отображения в UI."""
    table_name: str = ""
    sheet_name: str = ""
    row_count: int = 0
    has_data: bool = False

    def key(self) -> str:
        return f"{self.table_name}_{self.sheet_name}"

    def update_from_data(self, df: pd.DataFrame):
        self.row_count = len(df)
        self.has_data = self.row_count > 0


class SheetIndex:
    """
    Тупой хранитель названий листов для мультиселекта.
    При каждом update_from_data() — ПОЛНАЯ ПЕРЕЗАПИСЬ.
    """

    def __init__(self, index_path: str = "/app/cache/sheet_index.json"):
        self.index_path = index_path
        self.sheets: Dict[str, SheetMetadata] = {}

    # -----------------------------------------------------------------
    # Сохранение / Загрузка
    # -----------------------------------------------------------------
    def save(self) -> None:
        data = {"sheets": {k: asdict(v) for k, v in self.sheets.items()}}
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Индекс сохранён: {len(self.sheets)} листов")

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            return False
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Берём только те поля, которые есть в новом SheetMetadata
        self.sheets = {}
        for k, v in data["sheets"].items():
            filtered = {field: v.get(field, "") for field in SheetMetadata.__dataclass_fields__}
            self.sheets[k] = SheetMetadata(**filtered)
        logger.info(f"📂 Индекс загружен: {len(self.sheets)} листов")
        return True

    # -----------------------------------------------------------------
    # Обновление
    # -----------------------------------------------------------------
    def clear(self) -> None:
        """Очищает ВСЕ данные — вызывается перед обновлением."""
        self.sheets.clear()
        logger.info("🧹 Индекс очищен")

    def update_from_data(self, df: pd.DataFrame, table_name: str, sheet_name: str):
        """Добавляет/обновляет ОДИН лист."""
        key = f"{table_name}_{sheet_name}"
        if key not in self.sheets:
            self.sheets[key] = SheetMetadata(
                table_name=table_name,
                sheet_name=sheet_name
            )
        self.sheets[key].update_from_data(df)

    def get_all_sheets(self) -> List[Dict]:
        return [asdict(s) for s in self.sheets.values()]