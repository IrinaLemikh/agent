# /root/agent/core/tools/registry.py
"""
Реестр доступных инструментов с описаниями для LLM.
"""

from typing import Dict, Any
from .client_tools import get_top_clients, search_client, search_client_by_date
from .point_tools import get_top_points, search_point, search_point_by_date
from .problem_tools import get_top_problems, search_problem, search_problem_by_date
from .combined_tools import search_combined


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ===== КЛИЕНТЫ: ТОП =====
    "get_top_clients": {
        "func": get_top_clients,
        "description": "Топ N клиентов по количеству обращений ИЛИ все клиенты с количеством обращений > min_tickets. "
                       "Для запросов: 'топ клиентов', 'самые частые клиенты', 'клиенты с более чем 5 обращениями'.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Количество клиентов в топе. 0 — показать всех с > min_tickets обращений (по умолчанию 20)."
                },
                "min_tickets": {
                    "type": "integer",
                    "description": "Минимальное количество обращений (используется только при n=0, по умолчанию 2)."
                }
            },
            "required": []
        }
    },

    # ===== КЛИЕНТЫ: ПОИСК =====
    "search_client": {
        "func": search_client,
        "description": "Поиск ВСЕХ обращений клиента по ключевому слову в названии. Показывает таблицу обращений и топ-5 частых проблем. "
                       "Для запросов: 'обращения пивко', 'покажи тикеты чайкина', 'с чем обращался магнит'.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Название клиента или его часть (обязательно). Пример: 'пивко', 'чайкина'."
                }
            },
            "required": ["client_name"]
        }
    },

    "search_client_by_date": {
        "func": search_client_by_date,
        "description": "Все обращения клиента с дополнительным фильтром по датам. Сначала фильтр по клиенту, затем по дате. "
                       "Для запросов: 'обращения пивко за последние 7 дней', 'тикеты магнита с 1 по 15 мая'.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Название клиента или его часть (обязательно)."
                },
                "date_from": {
                    "type": "string",
                    "description": "Начало периода (YYYY-MM-DD). Используется вместе с date_to."
                },
                "date_to": {
                    "type": "string",
                    "description": "Конец периода (YYYY-MM-DD). Используется вместе с date_from."
                },
                "last_n_days": {
                    "type": "integer",
                    "description": "Показать обращения за последние N дней (например, 7)."
                },
                "consecutive_days": {
                    "type": "integer",
                    "description": "Найти непрерывный период длиной >= N дней с обращениями клиента."
                }
            },
            "required": ["client_name"]
        }
    },

    # ===== ТОРГОВЫЕ ТОЧКИ: ТОП =====
    "get_top_points": {
        "func": get_top_points,
        "description": "Топ N торговых точек по количеству обращений ИЛИ все точки с > min_tickets обращений. "
                       "Торговая точка = комбинация 'клиент | адрес'. "
                       "Для запросов: 'топ точек', 'кто обращается чаще всего', 'магазины с более чем 3 обращениями'.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Количество точек в топе. 0 — показать все с > min_tickets (по умолчанию 20)."
                },
                "min_tickets": {
                    "type": "integer",
                    "description": "Минимальное количество обращений (при n=0, по умолчанию 2)."
                }
            },
            "required": []
        }
    },

    # ===== ТОРГОВЫЕ ТОЧКИ: ПОИСК =====
    "search_point": {
        "func": search_point,
        "description": "Поиск всех обращений по торговой точке. Поддерживает форматы: 'Пивко | Хрустальная 37' (точное указание), "
                       "'Пивко' или 'Хрустальная' (поиск по части point_key). "
                       "Для запросов: 'обращения пивко хрустальная', 'покажи точку магнит казань'.",
        "parameters": {
            "type": "object",
            "properties": {
                "point_query": {
                    "type": "string",
                    "description": "Запрос для поиска точки: название, адрес или 'Клиент | Адрес' (обязательно)."
                }
            },
            "required": ["point_query"]
        }
    },

    "search_point_by_date": {
        "func": search_point_by_date,
        "description": "Все обращения торговой точки с фильтром по датам. "
                       "Для запросов: 'обращения пивко за последнюю неделю', 'тикеты точки магнит с 1 мая'.",
        "parameters": {
            "type": "object",
            "properties": {
                "point_query": {
                    "type": "string",
                    "description": "Запрос для поиска точки (обязательно)."
                },
                "date_from": {
                    "type": "string",
                    "description": "Начало периода (YYYY-MM-DD)."
                },
                "date_to": {
                    "type": "string",
                    "description": "Конец периода (YYYY-MM-DD)."
                },
                "last_n_days": {
                    "type": "integer",
                    "description": "Показать за последние N дней."
                },
                "consecutive_days": {
                    "type": "integer",
                    "description": "Непрерывный период >= N дней с обращениями точки."
                }
            },
            "required": ["point_query"]
        }
    },

    # ===== ПРОБЛЕМЫ: ТОП (с LLM-группировкой) =====
    "get_top_problems": {
        "func": get_top_problems,
        "description": "Топ N проблем, сгруппированных семантически через LLM, ИЛИ все группы проблем с > min_tickets обращений. "
                       "Для запросов: 'топ проблем', 'самые частые проблемы', 'с чем чаще всего обращаются'.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Количество групп проблем в топе. 0 — показать все с > min_tickets (по умолчанию 20)."
                },
                "min_tickets": {
                    "type": "integer",
                    "description": "Минимальное количество обращений в группе (при n=0, по умолчанию 2)."
                }
            },
            "required": []
        }
    },

    # ===== ПРОБЛЕМЫ: ПОИСК (с LLM-семантикой) =====
    "search_problem": {
        "func": search_problem,
        "description": "Поиск ВСЕХ обращений, семантически связанных с problem_query. LLM выбирает релевантные формулировки из уникальных значений. "
                       "Для запросов: 'проблемы со сканером', 'обращения по кассе', 'ошибки егаис'.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_query": {
                    "type": "string",
                    "description": "Ключевые слова или описание проблемы (обязательно). Пример: 'сканер', 'не работает касса'."
                }
            },
            "required": ["problem_query"]
        }
    },

    "search_problem_by_date": {
        "func": search_problem_by_date,
        "description": "Поиск обращений по проблеме (семантический, через LLM) + фильтр по датам. "
                       "Для запросов: 'проблемы со сканером за последние 7 дней', 'ошибки кассы с 1 по 15 мая'.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_query": {
                    "type": "string",
                    "description": "Ключевые слова проблемы (обязательно)."
                },
                "date_from": {
                    "type": "string",
                    "description": "Начало периода (YYYY-MM-DD)."
                },
                "date_to": {
                    "type": "string",
                    "description": "Конец периода (YYYY-MM-DD)."
                },
                "last_n_days": {
                    "type": "integer",
                    "description": "Показать за последние N дней."
                },
                "consecutive_days": {
                    "type": "integer",
                    "description": "Непрерывный период >= N дней."
                }
            },
            "required": ["problem_query"]
        }
    },

    # ===== КОМБИНИРОВАННЫЙ ПОИСК (клиент/точка + проблема) =====
    "search_combined": {
        "func": search_combined,
        "description": "Комбинированный поиск: сначала сужение по клиенту ИЛИ точке (pandas), затем фильтрация по проблеме через LLM. "
                       "Для запросов: 'у пивко проблемы со сканером', 'обращения магнита по кассе', "
                       "'проблемы с егаис в точке пивко | хрустальная'. "
                       "Обязательно укажи client_query ИЛИ point_query, И problem_query.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_query": {
                    "type": "string",
                    "description": "Название клиента или его часть. Взаимоисключающе с point_query."
                },
                "point_query": {
                    "type": "string",
                    "description": "Запрос для поиска точки ('Клиент | Адрес' или часть). Взаимоисключающе с client_query."
                },
                "problem_query": {
                    "type": "string",
                    "description": "Ключевые слова проблемы (обязательно)."
                }
            },
            "required": ["problem_query"]
        }
    },
}


def execute_tool(tool_name: str, args: Dict[str, Any], df, llm=None):
    """
    Выполняет инструмент по имени с переданными аргументами.
    
    Args:
        tool_name: имя инструмента из TOOL_REGISTRY
        args: словарь аргументов для функции
        df: основной DataFrame с данными
        llm: экземпляр DeepSeekClient (опционально, нужно для problem-инструментов)
    
    Returns:
        Результат, который вернула функция инструмента.
    """
    tool_info = TOOL_REGISTRY.get(tool_name)
    if not tool_info:
        raise ValueError(f"Инструмент '{tool_name}' не найден в реестре. Доступны: {list(TOOL_REGISTRY.keys())}")

    func = tool_info["func"]
    
    # Инструменты, которым нужен llm
    llm_tools = {"get_top_problems", "search_problem", "search_problem_by_date", "search_combined"}
    
    if tool_name in llm_tools:
        return func(df, **args, llm=llm)
    else:
        return func(df, **args)