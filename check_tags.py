import pandas as pd

df = pd.read_parquet("/app/core/data/current.parquet")
print("Всего строк в parquet:", len(df))
print("Тип значения в первой ячейке problem_tags:", type(df["problem_tags"].iloc[0]))
print()

non_empty = df["problem_tags"].apply(
    lambda x: hasattr(x, "__len__") and len(x) > 0
)
print("Строк с непустыми тегами (по ВСЕМУ датасету):", non_empty.sum(), "из", len(df))
print()

if non_empty.sum() > 0:
    print("Пример строки с тегами:")
    print(df[non_empty][["problem_raw", "problem_normalized", "problem_tags"]].iloc[0])
else:
    print("Ни у одной строки во ВСЁМ датасете нет тегов - проблема не в конкретном листе")

print()
print("--- Листы, которые реально есть в данных ---")
print("Уникальные значения _sheet_name:", df["_sheet_name"].unique()[:10])
