import os
from pathlib import Path
import logging

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model parameters for diabetes prediction
SEQ_LENGTH = 13  # 13 hourly readings (6 AM to 6 PM)
LATENT_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100

# Updated features for diabetes prediction
FEATURES = ['rbs_value']  # Time series feature - RBS values
TABULAR_FEATURES = ['age', 'bmi', 'average_rbs', 'hba1c', 'hypertension', 'respiratory_rate', 'heart_rate', 'spo2']
COND_FEATURES = ['age', 'bmi', 'hba1c']  # Conditioning features
TARGET_VARIABLES = ['diabetes', 'bp_status']  # What we're predicting

# ============ GITHUB-ONLY DATASET PATHS (NO LOCAL FALLBACK) ============
# Replace with your actual GitHub repository owner/repo/branch
GITHUB_OWNER = "eswaroy"
GITHUB_REPO = "synthieis"
GITHUB_BRANCH = "main"

# HARDCODED GitHub RAW URLs (MUST use raw.githubusercontent.com)
DEFAULT_TIME_SERIES_PATH = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/ml_service/datasets/enhanced_diabetes_timeseries_dataset.csv"
DEFAULT_TABULAR_PATH = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/ml_service/datasets/enhanced_diabetes_tabular_dataset.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "synthetic_data"
MODEL_DIR = PROJECT_ROOT / "trained_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Convert output paths to strings
OUTPUT_DIR = str(OUTPUT_DIR)
MODEL_DIR = str(MODEL_DIR)
LOGS_DIR = str(LOGS_DIR)

# Logging configuration with UTF-8 encoding for cross-platform compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("[OK] Using GitHub dataset URLs for training and generation")
logger.info(f"[OK] Time Series: {DEFAULT_TIME_SERIES_PATH}")
logger.info(f"[OK] Tabular: {DEFAULT_TABULAR_PATH}")
logger.info("=" * 80)
