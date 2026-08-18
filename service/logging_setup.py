"""Логирование сервиса.

Один формат для прикладных логов и uvicorn, уровень задаётся переменной
LOG_LEVEL (по умолчанию INFO). Логи пишутся в stdout — в Docker их собирает
`docker logs` / драйвер логирования, файлы внутри контейнера не нужны.
"""

from __future__ import annotations

import logging
import os
import sys

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # uvicorn ставит свои хендлеры со своим форматом — приводим к общему,
    # иначе в docker logs два разных формата вперемешку.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
