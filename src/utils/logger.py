import logging

from pathlib import Path
from logging.handlers import RotatingFileHandler


def setting_logger(logger_dir: Path) -> None:
    logger_dir.mkdir(parents=True, exist_ok=True)

    json_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    file = RotatingFileHandler(
        filename=logger_dir / "app.log",
        encoding="utf-8",
        maxBytes=10 * 1024 * 1024,
        backupCount=3
    )
    file.setLevel(logging.DEBUG)

    console.setFormatter(json_format)
    file.setFormatter(json_format)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file)

    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("onnxscript").setLevel(logging.WARNING)
    logging.getLogger("onnx_ir").setLevel(logging.WARNING)
