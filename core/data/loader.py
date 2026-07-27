"""
Модуль для загрузки данных из Parquet по списку листов.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


class DataLoader:
    """Загружает данные из current.parquet, фильтруя по листам."""
    
    def __init__(self, parquet_path: Optional[Path] = None):
        if parquet_path is None:
            parquet_path = Path("/app/core/data/current.parquet")
        self.parquet_path = parquet_path
        self._df: Optional[pd.DataFrame] = None
    
    def _load_if_needed(self):
        """Ленивая загрузка: читаем Parquet только при первом запросе."""
        if self._df is None:
            logger.debug(f"Чтение {self.parquet_path}")
            self._df = pd.read_parquet(self.parquet_path)
            logger.info(f"Загружено {len(self._df)} записей")
    
    def get_sheets(self, sheets: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Возвращает данные для указанных листов.
        
        Args:
            sheets: список вида [{"table_name": "имя", "sheet_name": "имя"}, ...]
        
        Returns:
            DataFrame с отфильтрованными данными
        """
        if not sheets:
            logger.warning("Список листов пуст")
            return pd.DataFrame()
        
        self._load_if_needed()
        
        # Строим условие: (table_name == X AND sheet_name == Y) OR ...
        mask = pd.Series(False, index=self._df.index)
        for s in sheets:
            table = s.get('table_name')
            sheet = s.get('sheet_name')
            if table and sheet:
                mask |= (self._df['_table_name'] == table) & \
                        (self._df['_sheet_name'] == sheet)
        
        result = self._df[mask].copy()
        logger.info(f"Найдено {len(result)} записей для {len(sheets)} листов")
        return result
    
    def get_all(self) -> pd.DataFrame:
        """Возвращает все данные (осторожно, может быть много!)."""
        self._load_if_needed()
        return self._df.copy()