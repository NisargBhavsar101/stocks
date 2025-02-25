import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models/trained_models/")
STOCK_LIST_FILE = os.path.join(BASE_DIR, "models/trained_models.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static/")
DATA_DIR = os.path.join(BASE_DIR, "data/")
STOCK_INFO_FILE = os.path.join(DATA_DIR, "stock_info.json")
