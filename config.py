
from pathlib import Path

BASE_DIR = Path("/home/student/e17/e175739/_p/BERT_chu/chABSA") #環境に応じて変更
VOCAB_FILE = BASE_DIR / "vocab/vocab.txt"
BERT_CONFIG = BASE_DIR / "weights/bert_config.json"
model_file = BASE_DIR / "weights/pytorch_model.bin"
MODEL_FILE = BASE_DIR / "data/bert_fine_tuning_chABSA_22epoch.pth"
PKL_FILE = BASE_DIR / "data/text.pkl"
DATA_PATH = BASE_DIR / "/home/student/e17/e175739/_p/BERT_chu/BERT_slot2/chABSA/data"
max_length = 256
