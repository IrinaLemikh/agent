import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime

# Добавляем корень проекта в пути поиска Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.loader import DataLoader
from core.data.indexer import SheetIndex
from core.llm.client import DeepSeekClient
from core.dispatcher import AgentDispatcher
from core.data.fetcher import Fetcher
from core.utils.dialog_logger import dialog_logger
from core.utils.docs import WHATSNEW, ABOUT_SYSTEM

# Настройка страницы
st.set_page_config(
    page_title="Агент по обращениям в техподдержку", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Кастомный CSS
st.markdown("""
<style>
    h1 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
        font-weight: 500 !important;
        color: #0f0f0f !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #FF8C00, #FF4500) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(255, 69, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 69, 0, 0.4) !important;
    }
    .data-preview {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background-color: white;
        margin: 1rem 0;
    }
    .row-info {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stMultiSelect {
        margin-bottom: 1rem;
    }
    .stMultiSelect div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
    /* Стили для сообщений чата */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stChatMessage pre {
        white-space: pre-wrap;
        font-family: inherit;
        font-size: inherit;
        margin: 0;
        padding: 0;
        background: none;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("Агент по обращениям в техподдержку")

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Инициализируем индекс и пробуем загрузить с диска
if 'indexer' not in st.session_state:
    st.session_state.indexer = SheetIndex()
    st.session_state.indexer.load()  # пробуем загрузить, если есть

if 'llm_client' not in st.session_state:
    st.session_state.llm_client = DeepSeekClient()

if 'show_whatsnew' not in st.session_state:
    st.session_state.show_whatsnew = False

# Состояние для превью
if 'preview_by_selection' not in st.session_state:
    st.session_state.preview_by_selection = pd.DataFrame()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Состояние для чата
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Хранение таблиц и графиков для каждого ответа
if 'message_tables' not in st.session_state:
    st.session_state.message_tables = {}  # ключ — индекс сообщения
if 'message_figures' not in st.session_state:
    st.session_state.message_figures = {}  # ключ — индекс сообщения

# ID сессии для логов
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Хранение реакций пользователя
if 'message_reactions' not in st.session_state:
    st.session_state.message_reactions = {}  # ключ — индекс сообщения

# --- КНОПКИ УПРАВЛЕНИЯ ---
col1, col2, col_empty = st.columns([2.2, 1.3, 8.5])

with col1:
    if st.button("Обновить данные из Google Sheets", type="primary", use_container_width=True):
        with st.spinner("Обновление таблиц и загрузка данных..."):
            try:
                fetcher = Fetcher()
                fetcher.fetch_all()
                st.session_state.indexer.load()
                if st.session_state.indexer.get_all_sheets():
                    st.session_state.data_loaded = True
                    st.session_state.last_update = datetime.now().strftime("%d.%m.%Y %H:%M")
                    st.session_state.preview_by_selection = pd.DataFrame()
                    st.session_state.messages = []
                    st.session_state.message_tables = {}
                    st.session_state.message_figures = {}
                    st.session_state.message_reactions = {}
                    st.success("Данные успешно обновлены!")
                else:
                    st.warning("Индекс загружен, но не содержит листов. Возможно, нет данных.")
            except Exception as e:
                st.error(f"Ошибка при обновлении данных: {e}")

with col2:
    if st.button("✨ Что нового", use_container_width=True):
        st.session_state.show_whatsnew = not st.session_state.get('show_whatsnew', False)

# Показываем "Что нового" если нажато
if st.session_state.get('show_whatsnew', False):
    with st.expander("✨ Что нового", expanded=True):
        st.markdown(WHATSNEW)
        st.markdown(ABOUT_SYSTEM)

# Если есть время последнего обновления - показываем
if st.session_state.last_update:
    st.caption(f"Последнее обновление: {st.session_state.last_update}")

# --- ОСНОВНОЙ ИНТЕРФЕЙС (только если данные загружены) ---
if st.session_state['data_loaded']:
    
    # Получаем все листы из индексера (локально, без походов в Google API)
    all_sheets = st.session_state.indexer.get_all_sheets()
    
    if not all_sheets:
        st.warning("В индексе нет данных. Нажмите кнопку обновления выше.")
    else:
        # Уникальные имена таблиц
        table_names = sorted(list(set(sheet['table_name'] for sheet in all_sheets)))
        
        # --- ВЫБОР ТАБЛИЦ ---
        st.markdown("### Таблицы")
        
        selected_table_names = st.multiselect(
            "Выберите таблицы для анализа",
            options=table_names,
            default=table_names if table_names else None,
            placeholder="Выберите таблицы..."
        )
        
        # --- ВЫБОР ЛИСТОВ ---
        st.markdown("### Листы")
        
        selected_sheets_info = []
        
        if selected_table_names:
            # Фильтруем листы по выбранным таблицам
            available_sheets = [
                sheet for sheet in all_sheets 
                if sheet['table_name'] in selected_table_names
            ]
            
            # Группируем по таблицам для отображения
            sheets_by_table = {}
            for sheet in available_sheets:
                table = sheet['table_name']
                if table not in sheets_by_table:
                    sheets_by_table[table] = []
                sheets_by_table[table].append(sheet['sheet_name'])
            
            for table, sheet_names in sheets_by_table.items():
                selected = st.multiselect(
                    f"Листы из таблицы **{table}**",
                    options=sorted(sheet_names, key=lambda x: x.lower()),
                    default=sorted(sheet_names),
                    placeholder="Выберите листы...",
                    key=f"sheets_{table}"
                )
                for sheet in selected:
                    selected_sheets_info.append({
                        "table_name": table,
                        "sheet_name": sheet
                    })
            
            if selected_sheets_info:
                st.caption(f"Выбрано {len(selected_sheets_info)} листов из {len(selected_table_names)} таблиц")
            else:
                st.warning("Выберите хотя бы один лист")
        else:
            st.warning("Сначала выберите таблицы")
        
        # --- ЗАГРУЗКА ДАННЫХ ПО ВЫБРАННЫМ ЛИСТАМ ДЛЯ ПРЕВЬЮ ---
        if selected_sheets_info:
            loader = DataLoader()
            df_selected = loader.get_sheets(selected_sheets_info)
            st.session_state.preview_by_selection = df_selected
        else:
            st.session_state.preview_by_selection = pd.DataFrame()
            
        
        # --- ПРЕВЬЮ ДАННЫХ (просто образец из выбранных листов) ---
        st.markdown("### Превью данных")
        
        if not st.session_state.preview_by_selection.empty:
            preview_cols = ['client_raw', 'address_raw', 'problem_raw', 'date', 'status', '_table_name', '_sheet_name']
            available_cols = [col for col in preview_cols if col in st.session_state.preview_by_selection.columns]
            
            if available_cols:
                preview_data = st.session_state.preview_by_selection[available_cols].head(100).copy()
                
                rename_map = {
                    'client_raw': 'Клиент',
                    'address_raw': 'Адрес',
                    'problem_raw': 'Проблема',
                    'date': 'Дата',
                    'status': 'Статус',
                    '_table_name': 'Таблица',
                    '_sheet_name': 'Лист'
                }
                preview_data.columns = [rename_map.get(col, col) for col in preview_data.columns]
                
                st.dataframe(
                    preview_data,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                st.markdown(
                    f"<div class='row-info'>Показано {len(preview_data)} из {len(st.session_state.preview_by_selection)} строк, {len(preview_data.columns)} колонок</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.info("В данных нет доступных для отображения колонок")
        else:
            st.info("В выбранных листах нет данных")
        
        # --- ЧАТ С АГЕНТОМ ---
        st.markdown("### 💬 Задайте вопрос агенту")
        
        # Отображаем историю чата
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                content = message["content"]
                # Конвертируем • в markdown-список
                if '•' in content:
                    lines = content.split('•')
                    items = [f"- {line.strip()}" for line in lines if line.strip()]
                    content = '\n'.join(items)
                st.markdown(content)

                # Если это ответ ассистента — показываем сохранённую таблицу и график
                if message["role"] == "assistant":
                    msg_index = message.get("msg_index", i)
                    
                    # Показываем график
                    if msg_index in st.session_state.message_figures:
                        st.plotly_chart(
                            st.session_state.message_figures[msg_index], 
                            use_container_width=True
                        )
                    
                    # Показываем таблицу
                    if msg_index in st.session_state.message_tables:
                        preview_df = st.session_state.message_tables[msg_index]
                        if "не найден" not in content.lower():
                            st.markdown("### Детальные данные")
                            st.dataframe(
                                preview_df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "№": st.column_config.NumberColumn(width="small")
                                }
                            )
                    
                    # --- КНОПКИ РЕАКЦИЙ ---
                    current_reaction = st.session_state.message_reactions.get(msg_index)
                    
                    if current_reaction is None:
                        col1, col2, col3 = st.columns([0.5, 0.5, 9])
                        
                        with col1:
                            if st.button("👍", key=f"like_{msg_index}", help="Ответ полезен!"):
                                st.session_state.message_reactions[msg_index] = "👍"
                                user_question = ""
                                for j in range(i-1, -1, -1):
                                    if st.session_state.messages[j]["role"] == "user":
                                        user_question = st.session_state.messages[j]["content"]
                                        break
                                
                                dialog_logger.log_reaction(
                                    question=user_question,
                                    answer=message["content"],
                                    selected_sheets=selected_sheets_info,
                                    reaction="👍",
                                    session_id=st.session_state.session_id
                                )
                                st.toast("👍 Спасибо за обратную связь!", icon="👍")
                                
                        
                        with col2:
                            if st.button("👎", key=f"dislike_{msg_index}", help="Нужна доработка"):
                                st.session_state.message_reactions[msg_index] = "👎"
                                user_question = ""
                                for j in range(i-1, -1, -1):
                                    if st.session_state.messages[j]["role"] == "user":
                                        user_question = st.session_state.messages[j]["content"]
                                        break
                                
                                dialog_logger.log_reaction(
                                    question=user_question,
                                    answer=message["content"],
                                    selected_sheets=selected_sheets_info,
                                    reaction="👎",
                                    session_id=st.session_state.session_id
                                )
                                st.toast("👎 Спасибо, агент будет наказан!", icon="👎")
                                
                    else:
                        st.caption(f"Вы оценили этот ответ: {current_reaction}")
                
        # Поле ввода нового вопроса
        if question := st.chat_input("Спросите о повторных обращениях, частых проблемах..."):
            # Добавляем вопрос пользователя в историю
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Получаем ответ от агента
            with st.chat_message("assistant"):
                with st.spinner("Анализирую данные..."):
                    try:
                        dispatcher = AgentDispatcher(
                            indexer=st.session_state.indexer,
                            llm_client=st.session_state.llm_client
                        )
                        
                        result = dispatcher.process(
                            question=question,
                            selected_sheets=selected_sheets_info
                        )
                        
                        answer = result.get('answer', 'Не удалось обработать запрос')
                        preview_df = result.get('preview_data', pd.DataFrame())
                        figure = result.get('figure')

                        # Отображаем график, если есть
                        if figure is not None:
                            st.plotly_chart(figure, use_container_width=True)
                        
                        # Отображаем ответ с сохранением форматирования
                        if answer:
                            # Конвертируем • в markdown-список
                            if '•' in answer:
                                lines = answer.split('•')
                                items = [f"- {line.strip()}" for line in lines if line.strip()]
                                answer = '\n'.join(items)
                            st.markdown(answer)
                        
                        # Отображаем детальные данные, если есть
                        if not preview_df.empty and "данные отсутствуют" not in answer.lower():
                            st.markdown("### Детальные данные")
                            st.dataframe(
                                preview_df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "№": st.column_config.NumberColumn(width="small")
                                }
                            )

                        # Сохраняем ответ в историю
                        msg_index = len(st.session_state.messages)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "msg_index": msg_index
                        })

                        if not preview_df.empty:
                            st.session_state.message_tables[msg_index] = preview_df
                        if figure is not None:
                            st.session_state.message_figures[msg_index] = figure

                        # --- КНОПКИ РЕАКЦИЙ ДЛЯ НОВОГО ОТВЕТА ---
                        col1, col2, col3 = st.columns([0.5, 0.5, 9])
                        
                        with col1:
                            if st.button("👍", key=f"like_{msg_index}", help="Ответ полезен!"):
                                st.session_state.message_reactions[msg_index] = "👍"
                                dialog_logger.log_reaction(
                                    question=question,
                                    answer=answer,
                                    selected_sheets=selected_sheets_info,
                                    reaction="👍",
                                    session_id=st.session_state.session_id
                                )
                                st.toast("👍 Спасибо за обратную связь!", icon="👍")
                        
                        with col2:
                            if st.button("👎", key=f"dislike_{msg_index}", help="Нужна доработка"):
                                st.session_state.message_reactions[msg_index] = "👎"
                                dialog_logger.log_reaction(
                                    question=question,
                                    answer=answer,
                                    selected_sheets=selected_sheets_info,
                                    reaction="👎",
                                    session_id=st.session_state.session_id
                                )
                                st.toast("👎 Спасибо, агент будет наказан!", icon="👎")
                                

                    except Exception as e:
                        error_msg = f"Ошибка при обработке запроса: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})