"""Модули для работы с данными: загрузка, индексация, нормализация."""
from .fetcher import Fetcher
from .indexer import SheetIndex
from .loader import DataLoader

__all__ = ['Fetcher', 'SheetIndex', 'DataLoader']