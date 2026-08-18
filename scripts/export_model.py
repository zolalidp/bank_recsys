"""Выгрузка production-модели из MLflow Model Registry в локальный каталог.

Запускается перед сборкой Docker-образа: сервис в рантайме читает файлы из
models/ и не ходит в MLflow. Обновление модели = повторный экспорт + пересборка.

    python -m scripts.export_model [--alias champion] [--out models]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Окружение настраивается ДО импорта mlflow. Без MLFLOW_TRACKING_URI клиент
# не идёт на наш сервер, а молча создаёт пустую локальную базу (mlflow.db в
# текущем каталоге) и падает с "Registered Model not found". Ключи из .env
# нужны для скачивания артефактов из S3 (сервер поднят с --no-serve-artifacts,
# клиент ходит в бакет сам).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")

import mlflow

REGISTERED_NAME = "bank_recsys"


def export(alias: str = "champion", out: str | Path = "models") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_NAME, alias)
    print(f"{REGISTERED_NAME}@{alias} -> version {version.version} (run {version.run_id})")

    model_dir = mlflow.artifacts.download_artifacts(
        f"models:/{REGISTERED_NAME}@{alias}", dst_path=str(out / "_mlflow_model")
    )
    cb = next(Path(model_dir).rglob("*.cb"))
    shutil.copy2(cb, out / "model.cb")
    shutil.rmtree(out / "_mlflow_model")

    mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="feature_artifacts", dst_path=str(out)
    )

    print(f"готово: {out / 'model.cb'} + {out / 'feature_artifacts'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    export(args.alias, args.out)
