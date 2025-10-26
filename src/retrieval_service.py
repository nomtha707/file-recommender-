import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core import VectorStoreIndex

# Project Imports
from src.database.metadata_db import MetadataDB
from src import config


class BehavioralReRanker(BaseNodePostprocessor):
    """
    Custom post-processor to re-rank nodes based on a hybrid score using data
    fetched from SQLite and attached during retrieval.
    """

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes:
            return []

        # --- Normalization based on fetched behavioral data ---
        max_time_spent = 0.0
        max_access_count = 0
        for node in nodes:
            # Access data added by the retriever (e.g., in node.node.extra_info)
            bh_data = node.node.extra_info or {}
            time_spent = bh_data.get("total_time_spent_hrs", 0.0)
            access_count = bh_data.get("access_count", 0)

            if time_spent > max_time_spent:
                max_time_spent = time_spent
            if access_count > max_access_count:
                max_access_count = access_count

        # Avoid division by zero
        if max_time_spent == 0:
            max_time_spent = 1.0
        if max_access_count == 0:
            max_access_count = 1

        # --- Re-scoring ---
        rescored_nodes = []
        for node in nodes:
            semantic_score = node.score or 0.0
            bh_data = node.node.extra_info or {}
            time_spent = bh_data.get("total_time_spent_hrs", 0.0)
            access_count = bh_data.get("access_count", 0)

            normalized_time = time_spent / max_time_spent
            normalized_access = access_count / max_access_count

            # --- UPDATED Weighted formula (Example: adds access count) ---
            # Adjust weights as needed (e.g., 50% semantic, 30% time, 20% access)
            final_score = (semantic_score * 0.5) + \
                (normalized_time * 0.3) + (normalized_access * 0.2)
            # --- ---

            new_node = NodeWithScore(node=node.node, score=final_score)
            rescored_nodes.append(new_node)

        # Sort by the new final_score in descending order
        rescored_nodes.sort(key=lambda x: x.score, reverse=True)

        print("\n🏆 Re-ranking complete. Top document by hybrid score:")
        if rescored_nodes:
            top_node = rescored_nodes[0]
            bh_data = top_node.node.extra_info or {}
            print(f"   - File: {top_node.metadata.get('path', 'N/A')}")
            print(f"   - Hybrid Score: {top_node.score:.4f}")
            print(
                f"   - Time Spent (hrs): {bh_data.get('total_time_spent_hrs', 'N/A')}")
            print(f"   - Access Count: {bh_data.get('access_count', 'N/A')}")

        return rescored_nodes


class CustomHybridRetriever(BaseRetriever):
    """
    Custom retriever implementing a two-stage process:
    1. Semantic retrieval from LlamaIndex (ChromaDB).
    2. Metadata fetching & filtering using SQLite DB.
    Attaches behavioral data to nodes for re-ranking.
    """

    def __init__(self, index: VectorStoreIndex, metadata_db: MetadataDB, top_k: int = 20):
        self._index = index
        self._metadata_db = metadata_db
        self._top_k = top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieval combining vector search and DB filtering."""
        print(
            f"\n🔍 Stage 1: Performing semantic search for: '{query_bundle.query_str}' (Top {self._top_k})")

        # Use LlamaIndex's async retriever
        base_retriever = self._index.as_retriever(similarity_top_k=self._top_k)
        # Run synchronous retrieve in a separate thread to avoid blocking asyncio
        initial_nodes = await asyncio.to_thread(base_retriever.retrieve, query_bundle)

        print(f"   └── Found {len(initial_nodes)} initial candidates.")

        # --- Metadata Fetching & Filtering (using SQLite) ---
        print("🔍 Stage 2: Fetching metadata from SQLite & filtering inactive/old files.")

        seven_days_ago_iso = (datetime.now() - timedelta(days=7)).isoformat()
        filtered_nodes_with_bh_data = []

        # Fetch metadata asynchronously (can be improved with batching if needed)
        tasks = []
        for node in initial_nodes:
            path = node.metadata.get('path')
            if path:
                # Use to_thread for the synchronous DB call
                tasks.append(asyncio.to_thread(self._metadata_db.get, path))

        metadata_results = await asyncio.gather(*tasks)

        valid_node_count = 0
        for i, node in enumerate(initial_nodes):
            path = node.metadata.get('path')
            if not path or i >= len(metadata_results):  # Safety check
                continue

            meta = metadata_results[i]

            # 1. Filter out inactive or non-existent metadata
            if not meta or meta.get('active', 0) == 0:
                # print(f"  Skipping inactive/missing meta: {path}") # Optional Debug
                continue

            # 2. Filter by recency (e.g., modified in last 7 days)
            modified_at_str = meta.get('modified_at', '')
            if modified_at_str < seven_days_ago_iso:
                # print(f"  Skipping old file (modified {modified_at_str}): {path}") # Optional Debug
                continue

            # 3. Attach behavioral data for the re-ranker
            # Store in extra_info which is designed for arbitrary data
            node.node.extra_info = {
                "total_time_spent_hrs": meta.get("total_time_spent_hrs", 0.0),
                "access_count": meta.get("access_count", 0),
                "accessed_at": meta.get("accessed_at", ""),
                "modified_at": modified_at_str  # Already checked
            }
            filtered_nodes_with_bh_data.append(node)
            valid_node_count += 1

        print(
            f"   └── {valid_node_count} candidates remain after filtering & metadata fetch.")

        return filtered_nodes_with_bh_data

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Synchronous wrapper for async version - needed for non-async query engines
        # This might block if called from an async context without await asyncio.to_thread
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._aretrieve(query_bundle))
