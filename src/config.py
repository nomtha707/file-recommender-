# src/config.py
from pathlib import Path

# --- Core Paths ---
# Define project root relative to this config file
PROJECT_ROOT = Path(__file__).parent.parent 
DB_FOLDER = PROJECT_ROOT / "db"
DOCUMENTS_FOLDER = PROJECT_ROOT / "documents_to_monitor"

# --- Watcher Settings ---
# IMPORTANT: Update this to the main directory you want to watch
WATCH_PATH = r"D:/College_IIITDWD" 
# Folders to exclude from indexing
EXCLUDED_DIRS = ['.venv', 'site-packages', '__pycache__', '.git', '.vscode', 'node_modules', '$RECYCLE.BIN']
# File types to index
VALID_EXTENSIONS = ('.txt', '.md', '.py', '.csv', '.docx', '.pdf') 
# Skip files larger than this (in bytes) - e.g., 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024 

# --- Database Paths ---
SQLITE_DB_PATH = DB_FOLDER / "metadata.db"
CHROMA_DB_PATH = str(DB_FOLDER / "chroma_db") # Chroma needs a string path

# --- Embedding Model ---
# Using the same model as Project 1 for consistency
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cpu" # Force embedding to CPU to save GPU VRAM

# --- LLM Model ---
LLM_MODEL_NAME = "gemma:2b-cpu" # Use the CPU version
LLM_REQUEST_TIMEOUT = 600.0 # 10 minutes

# --- Retrieval Settings ---
# How many candidates to fetch initially during semantic search
SEMANTIC_TOP_K = 20
# How many final results to show the user
DISPLAY_TOP_K = 5