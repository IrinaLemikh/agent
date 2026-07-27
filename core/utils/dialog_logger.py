"""
Логгер диалогов с реакциями пользователей.
Пишет диалоги в читаемом текстовом формате с ротацией по дате.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class DialogLogger:
    """
    Менеджер логирования диалогов агент-пользователь.
    - Автоматическая ротация по дате
    - Человекочитаемый текстовый формат
    - Сохранение реакций (👍/👎)
    - Автоочистка при превышении 10 МБ
    """
    
    def __init__(self, log_dir: str = "logs/dialogs", max_size_mb: int = 10):
        """
        Args:
            log_dir: директория для хранения логов диалогов
            max_size_mb: максимальный размер всех файлов диалогов в МБ
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        
    def _get_log_file(self) -> Path:
        """Возвращает путь к файлу логов на текущую дату."""
        return self.log_dir / f"dialogs_{self._current_date}.log"
    
    def _check_rotation(self):
        """Проверяет и выполняет ротацию при смене даты."""
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self._current_date:
            self._current_date = new_date
            logger.info(f"Ротация лога диалогов: создан файл на {new_date}")
    
    def _cleanup_old_logs(self):
        """Автоочистка старых логов при превышении лимита."""
        total_size = sum(
            f.stat().st_size 
            for f in self.log_dir.glob("dialogs_*.log") 
            if f.is_file()
        )
        
        if total_size > self.max_size_bytes:
            logger.warning(
                f"Общий размер логов диалогов ({total_size / 1024 / 1024:.1f} МБ) "
                f"превышает лимит ({self.max_size_bytes / 1024 / 1024:.0f} МБ). "
                f"Выполняю очистку..."
            )
            
            # Получаем все файлы логов, сортируем по дате (старые первее)
            log_files = sorted(
                self.log_dir.glob("dialogs_*.log"),
                key=lambda p: p.stat().st_mtime
            )
            
            # Удаляем самые старые файлы, пока размер не станет ниже лимита
            for old_file in log_files[:-1]:  # Оставляем последний (текущий)
                if total_size <= self.max_size_bytes:
                    break
                    
                file_size = old_file.stat().st_size
                old_file.unlink()
                total_size -= file_size
                logger.info(f"Удалён старый лог диалогов: {old_file.name} ({file_size / 1024:.1f} КБ)")
    
    def _format_dialog_entry(
        self,
        question: str,
        answer: str,
        selected_sheets: list,
        reaction: Optional[str] = None,
        session_id: str = "unknown",
        response_time: Optional[float] = None,
        is_reaction_only: bool = False
    ) -> str:
        """
        Форматирует запись диалога в читаемый текст.
        
        Args:
            question: вопрос пользователя
            answer: ответ агента
            selected_sheets: список выбранных листов
            reaction: реакция пользователя
            session_id: ID сессии
            is_reaction_only: True если это только реакция (без повторения всего диалога)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheets_info = ", ".join([
            f"{s['table_name']}/{s['sheet_name']}" 
            for s in selected_sheets
        ]) if selected_sheets else "нет"
        
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"[{timestamp}] Session: {session_id}")

        if response_time is not None:
            lines.append(f"Response time: {response_time} сек")
        
        if is_reaction_only:
            # Только реакция на предыдущий ответ
            lines.append(f"User question: {question}")
            lines.append(f"Agent answer (first 200 chars): {answer[:200]}{'...' if len(answer) > 200 else ''}")
        else:
            # Полный диалог
            lines.append(f"Sheets used: {sheets_info}")
            lines.append(f"")
            lines.append(f"User: {question}")
            lines.append(f"")
            lines.append(f"Agent: {answer}")
        
        if reaction:
            reaction_text = {
                "👍": "👍 Полезно",
                "👎": "👎 Можно лучше"
            }.get(reaction, f"Реакция: {reaction}")
            lines.append(f"")
            lines.append(f"Reaction: {reaction_text}")
        
        lines.append(f"{'='*80}")
        lines.append("")  # пустая строка между записями
        
        return "\n".join(lines)
    
    def log_dialog(
        self,
        question: str,
        answer: str,
        selected_sheets: list,
        reaction: Optional[str] = None,
        session_id: Optional[str] = None,
        response_time: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ):
        """
        Записывает диалог в лог.
        
        Args:
            question: вопрос пользователя
            answer: ответ агента
            selected_sheets: список выбранных листов [{"table_name": ..., "sheet_name": ...}]
            reaction: реакция пользователя (👍, 👎 или None)
            session_id: ID сессии (если не указан — авто из st.session_state)
            extra_meta: дополнительные метаданные
        """
        self._check_rotation()
        
        # Форматируем запись
        entry = self._format_dialog_entry(
            question=question,
            answer=answer,
            selected_sheets=selected_sheets,
            reaction=reaction,
            session_id=session_id or "unknown",
            response_time=response_time,
            is_reaction_only=False
        )
        
        # Пишем в файл
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        # Периодическая очистка
        if hash(question) % 10 == 0:
            self._cleanup_old_logs()
        
        reaction_str = f" (реакция: {reaction})" if reaction else ""
        logger.debug(
            f"Диалог записан: {len(answer)} символов ответа, "
            f"{len(selected_sheets)} листов{reaction_str}"
        )
    
    def log_reaction(
        self,
        question: str,
        answer: str,
        selected_sheets: list,
        reaction: str,
        session_id: Optional[str] = None,
        response_time: Optional[float] = None
    ):
        """
        Логирует реакцию пользователя на ответ.
        Записывает в сокращённом формате (только вопрос/ответ кратко + реакция).
        """
        self._check_rotation()
        
        # Форматируем запись только с реакцией
        entry = self._format_dialog_entry(
            question=question,
            answer=answer,
            selected_sheets=selected_sheets,
            reaction=reaction,
            session_id=session_id or "unknown",
            response_time=response_time,
            is_reaction_only=True
        )
        
        # Пишем в файл
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        logger.debug(f"Реакция записана: {reaction} на ответ длиной {len(answer)} символов")


# Глобальный экземпляр для использования во всём приложении
dialog_logger = DialogLogger()