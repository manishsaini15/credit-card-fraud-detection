import logging
import os

# Create logs folder

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(
    LOG_DIR,
    "running_logs.log"
)

logging.basicConfig(

    filename=LOG_FILE_PATH,

    format=(
        "[ %(asctime)s ] "
        "%(lineno)d "
        "%(name)s "
        "- %(levelname)s "
        "- %(message)s"
    ),

    level=logging.INFO
)

logger = logging.getLogger("FraudDetection")