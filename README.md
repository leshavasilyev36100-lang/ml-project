# Customer Churn ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Воспроизводимый модульный пайплайн машинного обучения для задачи классификации оттока клиентов. Проект построен по стандартам инженерной разработки (Production/MLOps): зависимости изолированы через `uv`, параметры вынесены в `YAML`, а исполнение стадий оркестрируется единым Bash-скриптом.

---
Архитектура

```text
ml-project/
├── configs/
│   └── config.yaml          # Гиперпараметры и пути к данным/артефактам
├── data/
│   ├── raw/                 # Исходный сырой датасет
│   └── processed/           # Разделенные выборки (train.csv, test.csv)
├── models/                  # Сохраненная сериализованная модель (.joblib)
├── outputs/                 # Метрики качества (metrics.json) и предсказания (predictions.csv)
├── src/
│   ├── data.py              # Генерация и разбиение выборки
│   ├── train.py             # Обучение GradientBoostingClassifier
│   └── predict.py           # Инференс модели на отложенных данных
├── .gitignore               # Исключение артефактов и окружения из Git
├── pyproject.toml           # Декларативное описание зависимостей проекта
├── run.sh                   # Bash-оркестратор стадий пайплайна
└── uv.lock                  # Зафиксированные версии пакетов
