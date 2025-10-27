import os
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from src import config

# --- Project Imports ---
# Use absolute imports based on the 'src' folder being the root
from src import config  # Import our unified config
from src.database.metadata_db import MetadataDB
from src.watcher.extractor import extract_text  # Use the extractor from Project 2

# --- LlamaIndex Imports ---
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Initialize Embedding Model (CPU) ---
# We initialize it once here for the watcher process
# Ensure this matches the settings in main.py
embed_model = HuggingFaceEmbedding(
    model_name=config.EMBED_MODEL_NAME,
    device=config.EMBED_DEVICE
)
print(
    f"Watcher: Embedding model {config.EMBED_MODEL_NAME} loaded on {config.EMBED_DEVICE}.")

def file_metadata(path: str):
    """Gets file metadata, converting timestamps to ISO format."""
    try:
        # Ensure path is a string for os.stat
        path_str = str(path)
        st = os.stat(path_str)
        return {
            'path': path_str,  # Store the string path
            'name': os.path.basename(path_str),
            'size': st.st_size,
            'created_at': datetime.fromtimestamp(st.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(st.st_mtime).isoformat(),
            'accessed_at': datetime.fromtimestamp(st.st_atime).isoformat(),
            # Defaults for behavioral data (will be updated elsewhere if needed)
            'access_count': 0,
            'total_time_spent_hrs': 0.0,
            'extra_json': '{}'
        }
    except Exception as e:
        print(f'Watcher Error: Could not get metadata for {path}: {e}')
        return None


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, db: MetadataDB, index: VectorStoreIndex):
        self.db = db
        self.index = index  # We'll use the LlamaIndex index object directly
        self.excluded_dirs = config.EXCLUDED_DIRS
        self.valid_extensions = config.VALID_EXTENSIONS
        print("Watcher: FileChangeHandler initialized.")

    def _is_path_excluded(self, path):
        if not path or not isinstance(path, str):
            return True
        # Check against normalized path components
        try:
            path_obj = Path(path)
            # Check if any part of the path is in excluded_dirs or starts with '.'
            if any(part in self.excluded_dirs or part.startswith('.') for part in path_obj.parts):
                return True
        except Exception:  # Handle potential path errors
            return True  # Exclude if path is invalid

        # Check file extension
        if not path.lower().endswith(self.valid_extensions):
            return True
        return False

    def process_file(self, path: str, check_modified_time: bool = False):
        """Processes a file for indexing into SQLite and ChromaDB via LlamaIndex."""
        # Ensure path is a string before checks
        path_str = str(path)

        if self._is_path_excluded(path_str):
            # print(f"Watcher Debug: Skipping excluded path: {path_str}") # Optional debug
            return False  # Indicate skipped

        if not os.path.exists(path_str):
            print(
                f"Watcher Warning: Path does not exist during processing: {path_str}")
            return False  # Indicate skipped

        try:
            current_meta = file_metadata(path_str)
            if not current_meta:
                return False  # Skip if metadata failed

            if check_modified_time:
                stored_mod_time_str = self.db.get_modified_time(path_str)
                if stored_mod_time_str and current_meta['modified_at'] <= stored_mod_time_str:
                    # print(f"Watcher Debug: Skipping unchanged file: {path_str}") # Optional debug
                    return False  # File hasn't changed, skip processing

            # Skip large files
            if current_meta['size'] > config.MAX_FILE_SIZE:
                print(
                    f"Watcher Info: Skipping large file ({current_meta['size'] / (1024*1024):.2f} MB): {path_str}")
                # Mark as inactive in DB maybe? Or just skip indexing? For now, just skip.
                # self.db.mark_deleted(path_str) # This might be too aggressive
                return False

            print(f"Watcher: Processing {path_str}")
            # print(f"  Extracting text...") # Verbose logging
            text = extract_text(path_str)
            # print(f"  Text extracted (length: {len(text)}).") # Verbose logging

            # Create a LlamaIndex Document
            # We use the path as doc_id for easy updates/deletions
            document = Document(
                text=text,
                doc_id=path_str,
                metadata={
                    # Include key metadata for potential filtering in LlamaIndex queries
                    'path': path_str,
                    'name': current_meta['name'],
                    'created_at': current_meta['created_at'],
                    'modified_at': current_meta['modified_at'],
                    'accessed_at': current_meta['accessed_at'],
                    'size': current_meta['size'],
                }
                # Note: Behavioral data (access_count, time_spent) lives primarily in SQLite
                # but could be added here if needed for direct LlamaIndex filtering.
            )

            # --- Update ChromaDB using LlamaIndex ---
            # print(f"  Upserting document to VectorStore via LlamaIndex...") # Verbose logging
            # Use insert to let LlamaIndex handle embedding and storage
            # It implicitly handles updates based on doc_id
            self.index.insert(document)
            # print(f"  Document upserted to VectorStore.") # Verbose logging

            # --- Update SQLite DB ---
            # Fetch existing behavioral data to preserve it during upsert
            existing_behavioral = self.db.get_behavioral_data(path_str)
            if existing_behavioral:
                current_meta['access_count'] = existing_behavioral[0]
                current_meta['total_time_spent_hrs'] = existing_behavioral[1]

            # print(f"  Upserting metadata to SQLite DB...") # Verbose logging
            self.db.upsert(current_meta)
            # print(f"  Metadata upserted to SQLite DB.") # Verbose logging

            print(f"Watcher: Indexed {path_str}")
            return True  # Indicate processed

        except Exception as e:
            print(f"Watcher ❌ Error processing {path_str}: {e}")
            return False  # Indicate error

    def on_created(self, event):
        if event.is_directory:
            return
        self.process_file(event.src_path, check_modified_time=False)

    def on_modified(self, event):
        if event.is_directory:
            return
        self.process_file(event.src_path, check_modified_time=False)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path_str = str(event.src_path)
        if not self._is_path_excluded(path_str):
            print(f"Watcher: Deleting {path_str} from index...")
            # Mark as deleted in SQLite
            self.db.mark_deleted(path_str)
            # Delete from ChromaDB via LlamaIndex
            try:
                self.index.delete_ref_doc(path_str, delete_from_docstore=True)
                print(f"Watcher: Deleted {path_str} successfully.")
            except Exception as e:
                # Need robust error handling (e.g., if doc_id not found)
                print(
                    f"Watcher Warning: Error deleting {path_str} from vector store: {e}")

# --- Main Watcher Loop (intended to be run in a separate thread/process) ---


def perform_initial_scan(db: MetadataDB, index: VectorStoreIndex, event_handler: FileChangeHandler):
    """Performs the initial scan, processing new/modified files."""
    path_to_watch = str(config.WATCH_PATH)
    print(
        f"Watcher: Performing initial scan of {path_to_watch} (processing new/modified files)...")
    processed_count = 0
    skipped_count = 0
    start_time = time.time()  # Start timing

    if os.path.exists(path_to_watch):
        for root, dirs, files in os.walk(path_to_watch, topdown=True):
            dirs[:] = [
                d for d in dirs if d not in config.EXCLUDED_DIRS and not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]

            try:
                is_excluded_root = any(part in config.EXCLUDED_DIRS or part.startswith(
                    '.') for part in Path(root).parts)
            except Exception:
                is_excluded_root = True

            if is_excluded_root:
                continue

            for filename in files:
                # Add a KeyboardInterrupt check inside the loop
                try:
                    # Check for interruption periodically
                    if time.time() % 10 == 0:  # Check every ~10 seconds
                        pass  # Simple check to allow interrupt handling

                    file_path = os.path.join(root, filename)
                    was_processed = event_handler.process_file(
                        file_path, check_modified_time=True)
                    if was_processed is True:
                        processed_count += 1
                    elif was_processed is False:
                        skipped_count += 1
                except KeyboardInterrupt:
                    print("\nWatcher: Initial scan interrupted by user.")
                    raise  # Re-raise the exception to stop the scan
                except Exception as e:
                    print(
                        f"Watcher Error during initial scan for {filename}: {e}")
                    skipped_count += 1

        end_time = time.time()  # End timing
        print(
            f"Watcher: Initial scan complete. Processed: {processed_count}, Skipped/Unchanged: {skipped_count}. Duration: {end_time - start_time:.2f} seconds.")
    else:
        print(f"Watcher Error: Watch path '{path_to_watch}' does not exist.")
        raise FileNotFoundError(f"Watch path does not exist: {path_to_watch}")
    

def run_observer_loop(db: MetadataDB, index: VectorStoreIndex, event_handler: FileChangeHandler):
    """Starts the file watcher observer loop (no initial scan)."""
    path_to_watch = str(config.WATCH_PATH)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    print(f'Watcher: Started watching {path_to_watch} for changes...')

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nWatcher: Stopping observer...")
        observer.stop()
        # No need to join here, main thread handles shutdown
    # observer.join() # Join is usually blocking, handle in main thread if needed
    print("Watcher: Observer stopped.")

# def run_watcher_service(db: MetadataDB, index: VectorStoreIndex):
#     """Starts the file watcher service."""
#     path_to_watch = str(config.WATCH_PATH)  # Ensure it's a string
#     event_handler = FileChangeHandler(db, index)

#     print(
#         f"Watcher: Performing initial scan of {path_to_watch} (processing new/modified files)...")
#     processed_count = 0
#     skipped_count = 0
#     if os.path.exists(path_to_watch):
#         for root, dirs, files in os.walk(path_to_watch, topdown=True):
#             dirs[:] = [
#                 d for d in dirs if d not in config.EXCLUDED_DIRS and not d.startswith('.')]
#             files = [f for f in files if not f.startswith('.')]

#             # Check root exclusion more carefully
#             try:
#                 is_excluded_root = any(part in config.EXCLUDED_DIRS or part.startswith(
#                     '.') for part in Path(root).parts)
#             except Exception:
#                 is_excluded_root = True  # Exclude if path is invalid

#             if is_excluded_root:
#                 continue

#             for filename in files:
#                 try:
#                     file_path = os.path.join(root, filename)
#                     # Pass check_modified_time=True for the scan
#                     was_processed = event_handler.process_file(
#                         file_path, check_modified_time=True)
#                     if was_processed is True:  # Explicit check for True
#                         processed_count += 1
#                     # Explicit check for False (Skipped or Error)
#                     elif was_processed is False:
#                         skipped_count += 1
#                     # None return might indicate path issue before processing attempt
#                 except Exception as e:
#                     print(
#                         f"Watcher Error during initial scan for {filename}: {e}")
#                     skipped_count += 1
#         print(
#             f"Watcher: Initial scan complete. Processed: {processed_count}, Skipped/Unchanged: {skipped_count}")
#     else:
#         print(
#             f"Watcher Error: Watch path '{path_to_watch}' does not exist. Please check config.py.")
#         return  # Stop if path is invalid

#     observer = Observer()
#     observer.schedule(event_handler, path_to_watch, recursive=True)
#     observer.start()
#     print(f'Watcher: Started watching {path_to_watch} for changes...')

#     try:
#         while True:
#             time.sleep(5)  # Reduce CPU usage slightly by sleeping longer
#     except KeyboardInterrupt:
#         print("\nWatcher: Stopping observer...")
#         observer.stop()
#     observer.join()
#     print("Watcher: Observer stopped.")

# Example of how this might be run (e.g., in main.py)
# if __name__ == '__main__':
#     # This part would typically be in your main application setup
#     db_instance = MetadataDB(config.SQLITE_DB_PATH)
#
#     # Initialize ChromaDB vector store for LlamaIndex
#     chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
#     chroma_collection = chroma_client.get_or_create_collection("fused_recommender_store")
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
#     # Create the main LlamaIndex VectorStoreIndex object
#     llama_index_instance = VectorStoreIndex.from_documents(
#         [], storage_context=storage_context, embed_model=embed_model
#     )
#
#     # Run the watcher service (maybe in a thread)
#     run_watcher_service(db_instance, llama_index_instance)
