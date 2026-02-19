"""Configuration settings for the system."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "llama3.2"
TEMPERATURE = 0
MAX_ITERATIONS = 10

# Pipeline models
PARSER_MODEL = "mistral"
EXPLAINER_MODEL = "llama3.2"

# =============================================================================
# DATA PATHS — Change these when switching to new data files
# =============================================================================

# Transaction data
TRANSACTIONS_FILE = os.path.join(_PROJECT_ROOT, "data", "rgs.csv")
TRANSACTIONS_DELIMITER = "\t"

# Bureau DPD tradeline data
BUREAU_DPD_FILE = os.path.join(_PROJECT_ROOT, "dpd_data.csv")
BUREAU_DPD_DELIMITER = "\t"

# Pre-computed tradeline features
TL_FEATURES_FILE = os.path.join(_PROJECT_ROOT, "tl_features.csv")
TL_FEATURES_DELIMITER = "\t"

LOG_DIR = "logs"

# =============================================================================
# SETTINGS
# =============================================================================

VERBOSE_MODE = True
STREAMING_ENABLED = True
USE_LLM_EXPLAINER = True
