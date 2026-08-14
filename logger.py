import logging
from logging.handlers import RotatingFileHandler
import os

logger = logging.getLogger("index_tts2_service")
logger.setLevel(logging.INFO)

log_path = os.environ.get("INDEXTTS_NEW_LOG_PATH", "./indextts_service.log")

# 每个文件最大 20MB，最多保留 3 个文件（当前 + 2 个备份）
handler = RotatingFileHandler(
    log_path,
    mode="a",
    maxBytes=20 * 1024 * 1024,  # 20MB
    backupCount=2,               # 2 个备份 + 1 个当前 = 3 个
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
# 避免重复添加 handler（模块被多次 import 时）
if not logger.handlers:
    logger.addHandler(handler)
