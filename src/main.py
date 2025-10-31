import threading
import asyncio
import chromadb
import os # Keep os import for touch_file helper if used elsewhere, or general path ops

# --- Project Imports ---
from src import config
from src.database.metadata_db import MetadataDB
# IMPORT the specific watcher functions and Handler
from src.watcher.file_watcher import perform_initial_scan, run_observer_loop, FileChangeHandler
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
from llama_index.core import PromptTemplate
from src.retrieval_service import CustomHybridRetriever, BehavioralReRanker

QA_TEMPLATE_STR = (
    "Context information is provided below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are a helpful assistant. Based *only* on the context information provided above, "
    "answer the query.\n"
    "If the context does not contain the answer, simply state 'The context does not provide "
    "information to answer this query.' Do not add any other information.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_template = PromptTemplate(QA_TEMPLATE_STR)


async def user_chat_loop(query_engine: RetrieverQueryEngine):
    """
    Waits for user input and queries the engine (without LLM synthesis).
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

            # This still involves embedding the query and retrieving
            print(f"\nThinking...")

            print("Attempting query embedding...")
            try:
                query_embedding = await asyncio.to_thread(Settings.embed_model.get_query_embedding, user_query)
                print(f"Query embedded successfully (shape: {len(query_embedding)}).")
            except Exception as embed_err:
                print(f"ERROR during query embedding: {embed_err}")
                continue

            # --- Use the Query Engine ---
            # Run the synchronous query method in a thread
            response = await asyncio.to_thread(query_engine.query, user_query)

            # --- Print Retrieved Files ---
            print("\n💡 Top Retrieved Files :")
            print(response.response)
            # if response.source_nodes:
            #     # Display top K nodes (using config.DISPLAY_TOP_K)
            #     for node in response.source_nodes[:config.DISPLAY_TOP_K]:
            #         # --- Make path printing more robust ---
            #         file_path = node.metadata.get('path', 'N/A')
            #         # Ensure node score exists and is float before formatting
            #         score = node.score if node.score is not None else 0.0
            #         print(f"- {file_path} (Score: {score:.4f})")
            #         # --- Uncomment to see more metadata for debugging ---
            #         # print(f"  Accessed: {node.node.extra_info.get('accessed_at', 'N/A')}")
            #         # print(f"  Modified: {node.node.extra_info.get('modified_at', 'N/A')}")
            #         # print(f"  Time Spent: {node.node.extra_info.get('total_time_spent_hrs', 'N/A')}")
            #         # print(f"  Access Count: {node.node.extra_info.get('access_count', 'N/A')}")
            #         # --- ---
            # else:
            #     print("No relevant documents found.")
            # # --- REMOVED the print(response.response) lines ---

        except Exception as e:
            print(f"\nAn error occurred: {e}")

            
# --- Main Application Logic ---
async def main():
    print("--- 🚀 Initializing Fused Document Recommender ---")

    # 1. Initialize SQLite Database
    db_instance = MetadataDB(str(config.SQLITE_DB_PATH)) # Ensure path is string
    print("SQLite DB initialized.")

    # 2. Configure LlamaIndex Settings
    Settings.llm = Ollama(model=config.LLM_MODEL_NAME, request_timeout=config.LLM_REQUEST_TIMEOUT)
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)
    Settings.embed_model = embed_model
    print(f"LLM: {config.LLM_MODEL_NAME} | Embed: {config.EMBED_MODEL_NAME} ({config.EMBED_DEVICE})")

    # 3. Initialize ChromaDB & LlamaIndex VectorStoreIndex
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection("fused_recommender_store")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        [], storage_context=storage_context, embed_model=embed_model
    )
    print("ChromaDB and LlamaIndex VectorStoreIndex initialized.")

    # --- Perform Initial Scan Synchronously ---
    print("Starting initial file scan. This may take a long time...")
    try:
        # Create the FileChangeHandler instance *before* the scan
        watcher_event_handler = FileChangeHandler(db_instance, index)

        # Run the blocking initial scan in a separate thread using asyncio.to_thread
        await asyncio.to_thread(
            perform_initial_scan,
            db_instance,
            index,
            watcher_event_handler # Pass the created handler
        )
    except FileNotFoundError as e:
        print(f"Error during startup: {e}")
        print("Exiting.")
        return # Stop execution if scan fails critically
    except KeyboardInterrupt:
        print("\nInitial scan cancelled by user. Exiting.")
        return # Stop if scan is cancelled
    except Exception as e:
        print(f"Unexpected error during initial scan: {e}")
        print("Continuing, but the index might be incomplete...")

    # --- Start the Watcher Service (Observer Loop Only) in a background thread ---
    print("Starting background file watcher...")
    watcher_thread = threading.Thread(
        target=run_observer_loop, # Use the observer loop function
        args=(db_instance, index, watcher_event_handler), # Pass the handler
        daemon=True # Make it a daemon thread
    )
    watcher_thread.start()
    print("File watcher service started in background thread.")

    # 5. Initialize Retrieval Components
    custom_retriever = CustomHybridRetriever(index=index, metadata_db=db_instance, top_k=3)
    reranker = BehavioralReRanker()
    print("Custom retriever and re-ranker initialized.")

    # 6. Create the LlamaIndex Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=custom_retriever,
        node_postprocessors=[reranker],
        llm=Settings.llm,
        text_qa_template=qa_template
        # response_mode="no_text"
    )
    print("Retriever Query Engine initialized.")
    print("--- ✅ System is ready ---")

    # --- UPDATED: Run Only the Chat Loop ---
    print("Starting chat loop...")
    # Create coroutine object for chat loop
    chat_coro = user_chat_loop(query_engine)

    # Explicitly create Task object for chat loop
    chat_task = asyncio.create_task(chat_coro, name="ChatLoopTask")

    # Wait for the chat_task to complete (when user types 'exit')
    await chat_task # Directly await the single chat task

    print("Main application loop finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n--- Ctrl+C detected. Shutting down ---")
    finally:
        print("Application exited.")