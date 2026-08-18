FROM python:3.10-slim

WORKDIR /app

# Зависимости отдельным слоем: пересборка при изменении кода не тянет pip заново.
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY scripts/ ./scripts/
COPY service/ ./service/

# Артефакты кладутся в образ на этапе сборки; перед сборкой выполнить:
#   python -m scripts.export_model
COPY models/ ./models/

EXPOSE 8000

# Подключение к Postgres задаётся переменными окружения DB_DESTINATION_*
# при запуске контейнера (docker run --env-file .env ...).
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
