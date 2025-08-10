# README.md
# Streamlit Trainer & Backtester with DNA Contract

Цель: два автономных Streamlit-приложения (Trainer и Backtester), работающих на локальных данных (CSV/XLSX/JSON), без прямых API-вызовов. Единый контракт "ДНК стратегии" (DNA) позволяет быстро подключать/обучать/тестировать любые стратегии. Поддержаны ансамбли моделей, кэширование фич/идей, сохранение артефактов обучения и бэктестов.

## Структура проекта

- strategies/
  - Example_DNA_v1.py           — пример ДНК-стратегии
- data/
  - raw/                        — исходные выгрузки
  - prepared/                   — кэш подготовленных датасетов
- models/                       — сохранённые модели и метаданные
- reports/
  - training/                   — отчёты обучения
  - backtests/                  — отчёты бэктестов
- libs/
  - utils.py                    — утилиты: логгер, таймер, кэш-ключи, GPU-настройки
  - dna_contract.py             — загрузка/валидация ДНК, снапшот исходника
  - data_io.py                  — загрузка, merge, валидация, кэш
  - features.py                 — общие фичи (ATR, RSI, SMA/EMA, волатильность)
  - training.py                 — пайплайн обучения, метрики, сохранение бандла
  - backtesting.py              — инференс/оркестр, симуляция, метрики, отчёты
- apps/
  - trainer_app.py              — Streamlit обучатор
  - backtester_app.py           — Streamlit тестер
- requirements.txt
- README.md

## Установка

1) Создайте окружение:
   - Python 3.10+ рекомендуется
   - Linux/Mac/Windows

2) Установка зависимостей:
   - python -m venv .venv
   - (Windows) .venv\Scripts\activate
   - (Unix) source .venv/bin/activate
   - pip install -r requirements.txt

Опционально для GPU:
- Установите CUDA-совместимый xgboost (pip install xgboost>=2.0.0 с CUDA-сборкой) и совместимую CUDA runtime.
- При отсутствии CUDA скрипты автоматически перейдут на CPU.

## Формат входных данных

Ожидаются колонки:
- open_time: datetime64[ns, UTC] (для JSON допустимы миллисекунды от эпохи — конвертируются)
- open, high, low, close, volume: числовые

Поддерживаемые форматы:
- CSV (заголовки обязательны)
- XLSX (первый лист)
- JSON (records/list; open_time может быть ms или ISO-строкой)

Требования:
- Без дубликатов по open_time
- Сортировка по времени по возрастанию
- Без NaN в обязательных колонках (строгая проверка)

## Запуск приложений

- Обучатор:
  - streamlit run apps/trainer_app.py

- Тестер:
  - streamlit run apps/backtester_app.py

Оба приложения имеют sidebar с путями (data/models/reports/strategies), уровнем логирования и флагами. Все шаги реализованы как формы с кнопками, состояние хранится в session_state, чтобы исключить произвольные rerun.

## Контракт ДНК (strategies/*.py)

Обязательные экспорты:
- STRATEGY_NAME: str
- EXIT_POLICIES: list[dict] (минимум поле "name")
- TASKS: dict — описание задач, например:
  - "entry_action": {"type": "classification", "classes": ["SKIP","BUY","SELL"]}
  - "policy_choice": {"type": "classification", "classes": [имена политик]}
  - "level_quality": {"type": "classification", "classes": [0,1]}

Обязательные функции:
- calculate_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame
- generate_ideas(df_with_features: pd.DataFrame, meta: dict) -> pd.DataFrame
- build_labels(df_with_ideas: pd.DataFrame, meta: dict) -> dict[str, pd.Series]
- feature_columns(df_with_features: pd.DataFrame) -> list[str]
- inference_inputs(df_with_features: pd.DataFrame, feats: list[str]) -> pd.DataFrame

Строгие правила:
- Никакого look-ahead/future leakage
- Никаких внешних API — только входные DataFrame
- Формат времени/колонок как в выгрузке
- Все параметры через meta (atr_period и т.п.)

## Обучение (apps/trainer_app.py)

Шаги:
1) Загрузка данных из data/raw; batch-merge, валидация, предпросмотр.
2) Выбор DNA из strategies/, динамическая загрузка, вывод STRATEGY_NAME/TASKS.
3) Настройка meta (периоды, окна), выбор задачи и алгоритма, гиперпараметры, флаг GPU.
4) Обучение: pipeline calculate_features → generate_ideas → build_labels → X/y → time split → fit → метрики.
5) Сохранение: models/{name}.joblib, {name}.meta.json, {name}.strategy.py; отчёты в reports/training/{name}/.

Метрики:
- Классификация: accuracy, F1 macro, ROC-AUC (бинарная)
- Регрессия: MAE/MSE/R2
- Артефакты: confusion_matrix.png, feature_importances.csv (если применимо)

## Бэктест (apps/backtester_app.py)

Шаги:
1) Загрузка данных.
2) Загрузка модели (.joblib) и метаданных (.meta.json); автоподключение копии DNA из .strategy.py или ручной выбор DNA. Настройка ансамбля.
3) Сбор фич/идей, формирование X строго по feature_columns из меты модели, инференс одной/нескольких моделей (голосование hard/soft/weighted).
4) Симуляция: комиссия, спред, плечо, риск (fixed cash/% equity), стартовый капитал; исполнение по рыночным ценам (close±spread), SL/TP по сигналам.
5) Сохранение отчёта: trades.csv, equity_curve.csv, metrics.csv/json, charts/*.png, config.json в reports/backtests/{run_id}/.

Метрики:
- total_pnl, winrate, avg_win/avg_loss, profit_factor, max_dd, sharpe, sqn

## Производительность

- Векторизация Pandas/NumPy
- Кэш подготовленных фич/идей (data/prepared) — ключ включает файлы/ДНК/meta/версию
- GPU XGBoost: device="cuda", tree_method="hist"; CPU fallback при недоступности
- Логи в reports/.../ui_logs.txt и дружественные сообщения об ошибках

## Пример: стратегии/Example_DNA_v1.py

- Простые фичи (ATR, RSI, SMA/EMA, волатильность), эвристические сигналы без look-ahead.
- Три задачи: entry_action, policy_choice, level_quality.
- Предварительный расчёт sl_price/tp_price на основании политики.

## Советы по данным

- Если JSON хранит миллисекунды — используйте встроенную загрузку (load_json) из приложения.
- Следите за NaN: валидация отклонит набор с пропусками в ключевых колонках.
- Разделяйте обучающий/валидационный период по времени (без shuffle).

## Пресеты ансамблей

- В тестере можно добавить модели для policy_choice и level_quality
- Логику агрегации (например, торговать только если level_quality=1 и entry_action!=SKIP) можно доработать в DNA или в оркестраторе при необходимости.

## Версионирование

- Версия инструмента: хранится в libs/utils.py (TOOL_VERSION)
- Версии библиотек/среды сохраняются в метаданных модели.

## Запуск в продакшн

- Модели и DNA файлы из models/ совместимы с будущим live-приложением (feature_columns, inference_inputs, сигналы).
- Для ускорения инференса на больших объёмах рассмотрите сохранение фич/идей в data/prepared и повторное использование.

## Тестирование (опционально)

- Рекомендуется добавить unit-тесты для libs/dna_contract.py, libs/data_io.py и критичных частей backtesting.py.

## Лицензия

- Укажите лицензию по вашему проекту.

