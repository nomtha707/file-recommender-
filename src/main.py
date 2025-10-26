import threading
import asyncio
import chromadb
import os 

# --- Project Imports ---
from src import config
from src.database.metadata_db import MetadataDB
# Import the watcher service function
from src.watcher.file_watcher import run_watcher_service

# --- LlamaIndex Imports ---
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine

# Import the UPDATED retrieval components
from src.retrieval_service import CustomHybridRetriever, BehavioralReRanker

# --- Background Simulator Task (using SQLite) ---


async def background_activity_simulator(db: MetadataDB):
    """
    Simulates user activity by updating the SQLite database.
    """
    print("[Simulator] Starting background activity simulation...")
    while True:
        await asyncio.sleep(60)  # Simulate every 60 seconds

        # Get a list of active files from DB to simulate activity on
        # Use to_thread for the synchronous DB call
        active_files = await asyncio.to_thread(db.fetch_all_active)
        if not active_files:
            # print("[Simulator] No active files found to simulate activity on.") # Optional Debug
            continue

        # Simulate activity on the first active file found
        # In a real scenario, you'd pick based on focus, randomness, etc.
        # fetch_all_active returns tuples, path is first element
        target_path = active_files[0][0]

        print(f"\n[Simulator] 📈 Simulating activity on: {target_path}")
        try:
            # Update SQLite DB - run synchronous DB method in a thread
            await asyncio.to_thread(db.increment_access, target_path, time_increment_hrs=0.5)
            print(f"[Simulator]    └── Updated DB for: {target_path}")

            # --- Trigger Watcher ---
            # "Touch" the file to make the watcher re-index its metadata (esp. modified/accessed time)
            # This ensures ChromaDB metadata stays somewhat fresh, though SQLite is the source of truth
            # Run the blocking file I/O in a thread
            await asyncio.to_thread(touch_file, target_path)
            print(f"[Simulator]    └── Triggered file modification for watcher.")

        except Exception as e:
            print(
                f"[Simulator] Error during simulation for {target_path}: {e}")


def touch_file(path: str):
    """Helper function to update file modification time."""
    try:
        with open(path, 'a'):
            os.utime(path, None)  # Update timestamps
    except Exception as e:
        print(f"[touch_file] Error touching {path}: {e}")

# --- User Chat Loop Task ---


async def user_chat_loop(query_engine: RetrieverQueryEngine):
    """
    Waits for user input and queries the engine.
    """
    print("\n--- Type your query. Enter 'exit' to quit the session. ---")
    while True:
        try:
            user_query = await asyncio.to_thread(input, "Query: ")

            if user_query.lower() == "exit":
                print("Shutting down chat...")
                break  # Exit the loop

            if not user_query:
                continue

            print(f"\nThinking...")

            # --- Use the Query Engine ---
            # Run the synchronous query method in a thread
            response = await asyncio.to_thread(query_engine.query, user_query)

            print("\n💡 Response:")
            print(response.response)  # Display the LLM's synthesized answer
            # print("\nSource Nodes:") # Optional: Print source nodes for debugging
            # for node in response.source_nodes:
            #     print(f"- {node.metadata.get('path')} (Score: {node.score:.4f})")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Consider adding more robust error handling or loop continuation

# --- Main Application Logic ---


async def main():
    print("--- 🚀 Initializing Fused Document Recommender ---")

    # 1. Initialize SQLite Database
    db_instance = MetadataDB(str(config.SQLITE_DB_PATH))  # Ensure path is string
    print("SQLite DB initialized.")

    # 2. Configure LlamaIndex Settings
    Settings.llm = Ollama(model=config.LLM_MODEL_NAME, request_timeout=config.LLM_REQUEST_TIMEOUT)
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)
    Settings.embed_model = embed_model
    print(
        f"LLM: {config.LLM_MODEL_NAME} | Embed: {config.EMBED_MODEL_NAME} ({config.EMBED_DEVICE})")

    # 3. Initialize ChromaDB & LlamaIndex VectorStoreIndex
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(
        "fused_recommender_store")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        # Pass embed_model explicitly
        [], storage_context=storage_context, embed_model=embed_model
    )
    print("ChromaDB and LlamaIndex VectorStoreIndex initialized.")

    # 4. Start the Watcher Service in a separate thread
    # Pass the DB instance and the LlamaIndex index object
    watcher_thread = threading.Thread(
        target=run_watcher_service,
        args=(db_instance, index),
        daemon=True  # Allows main program to exit even if watcher is running
    )
    watcher_thread.start()
    print("File watcher service started in background thread.")
    # Give the watcher a moment to potentially start its initial scan
    await asyncio.sleep(5)

    # 5. Initialize Retrieval Components
    custom_retriever = CustomHybridRetriever(
        index=index, metadata_db=db_instance, top_k=config.SEMANTIC_TOP_K)
    reranker = BehavioralReRanker()
    print("Custom retriever and re-ranker initialized.")

    # 6. Create the LlamaIndex Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=custom_retriever,
        node_postprocessors=[reranker],
        llm=Settings.llm  # Use LLM to synthesize final answers
        # streaming=True # Optional: for streaming responses
    )
    print("Retriever Query Engine initialized.")
    print("--- ✅ System is ready ---")

    # 7. Run Chat Loop and Background Simulator Concurrently
    print("Starting chat loop and background simulator...")

    # Create coroutine objects first
    chat_coro = user_chat_loop(query_engine)
    simulator_coro = background_activity_simulator(db_instance)

    # Explicitly create Task objects
    # Giving them names helps with debugging if needed
    chat_task = asyncio.create_task(chat_coro, name="ChatLoopTask")
    simulator_task = asyncio.create_task(simulator_coro, name="SimulatorTask")

    # Pass the set of Task objects to asyncio.wait
    done, pending = await asyncio.wait(
        {chat_task, simulator_task},  # Use a set { } or list [ ] of Tasks
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the pending task (likely the simulator)
    for task in pending:
        print(f"Cancelling pending task: {task.get_name()}...")
        task.cancel()
        try:
            await task  # Allow cancellation to complete
        except asyncio.CancelledError:
            print(f"Task {task.get_name()} cancelled successfully.")

    print("Main application loop finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n--- Ctrl+C detected. Shutting down ---")
    finally:
        print("Application exited.")
