from dotenv import load_dotenv
import os

load_dotenv("/kaggle/.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 1536
STRIDE = 128
OUTPUT_DIR = "/kaggle/output/deberta3large"

BATCH_SIZE = 1
ACC_STEPS = 2
EPOCHS = 3
LR = 2.5e-5

arch_suffix = "deberta_large_966"

name = f"ex_ec2_{arch_suffix}_return_overflowing_tokens"