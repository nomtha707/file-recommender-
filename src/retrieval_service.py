import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # Import TextNode
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

        # --- DEBUG PRINT 3 ---
        print("\nDEBUG (ReRanker Input): Metadata & ExtraInfo entering re-ranker:")
        for i, node in enumerate(nodes[:5]):  # Print first 5
            print(f"  Node {i} Metadata: {node.metadata}")
            print(f"  Node {i} NodeID: {node.node_id}")
            # Safer access
            print(
                f"  Node {i} Extra Info: {getattr(node.node, 'extra_info', 'N/A')}")
        # --- END DEBUG ---

        # ... (Normalization logic remains the same) ...
        max_time_spent = 0.0
        max_access_count = 0
        for node in nodes:
            bh_data = getattr(node.node, 'extra_info', {}) or {}  # Safer access
            time_spent = bh_data.get("total_time_spent_hrs", 0.0)
            access_count = bh_data.get("access_count", 0)
            if time_spent > max_time_spent:
                max_time_spent = time_spent
            if access_count > max_access_count:
                max_access_count = access_count
        if max_time_spent == 0:
            max_time_spent = 1.0
        if max_access_count == 0:
            max_access_count = 1

        # --- Re-scoring ---
        rescored_nodes = []
        for node in nodes:
            semantic_score = node.score or 0.0
            bh_data = getattr(node.node, 'extra_info',
                              {}) or {}  # Safer access
            time_spent = bh_data.get("total_time_spent_hrs", 0.0)
            access_count = bh_data.get("access_count", 0)

            normalized_time = time_spent / max_time_spent
            normalized_access = access_count / max_access_count

            final_score = (semantic_score * 0.5) + \
                (normalized_time * 0.3) + (normalized_access * 0.2)

            # --- Ensure metadata is preserved when creating new node ---
            # Create a new TextNode instance to hold potentially modified extra_info
            # This prevents modifying the original node object if it's referenced elsewhere
            new_internal_node = TextNode(
                text=node.node.get_text(),  # Copy text
                id_=node.node.node_id,      # Copy ID
                metadata=node.metadata,    # Copy original metadata
                extra_info=bh_data         # Assign the behavioral data
                # Copy other relevant fields if needed (relationships, hash, etc.)
            )
            new_node_with_score = NodeWithScore(
                node=new_internal_node, score=final_score)
            rescored_nodes.append(new_node_with_score)

        # Sort by the new final_score
        rescored_nodes.sort(key=lambda x: x.score, reverse=True)

        print("\n🏆 Re-ranking complete. Top document by hybrid score:")
        if rescored_nodes:
            top_node = rescored_nodes[0]
            bh_data = getattr(top_node.node, 'extra_info', {}) or {}
            # --- Now try getting path ONLY from metadata ---
            file_path = top_node.metadata.get('path', 'N/A')
            print(f"   - File: {file_path}")
            print(f"   - Hybrid Score: {top_node.score:.4f}")
            print(
                f"   - Time Spent (hrs): {bh_data.get('total_time_spent_hrs', 'N/A')}")
            print(f"   - Access Count: {bh_data.get('access_count', 'N/A')}")

        # --- DEBUG PRINT 4 ---
        print("\nDEBUG (ReRanker Output): Metadata & ExtraInfo of top re-ranked node:")
        if rescored_nodes:
            top_node_debug = rescored_nodes[0]
            print(f"  Top Node Metadata: {top_node_debug.metadata}")
            print(f"  Top Node NodeID: {top_node_debug.node_id}")
            print(
                f"  Top Node Extra Info: {getattr(top_node_debug.node, 'extra_info', 'N/A')}")
        # --- END DEBUG ---

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
        print(f"\n🔍 Stage 1: Performing semantic search for: '{query_bundle.query_str}' (Top {self._top_k})")

        base_retriever = self._index.as_retriever(similarity_top_k=self._top_k)
        initial_nodes = await asyncio.to_thread(base_retriever.retrieve, query_bundle)

        print(f"   └── Found {len(initial_nodes)} initial candidates.")

        # --- DEBUG PRINT 1 ---
        print("\nDEBUG (Initial): Metadata of first 5 candidates:")
        for i, node in enumerate(initial_nodes[:5]):
            print(f"  Node {i} Metadata: {node.metadata}")
            print(f"  Node {i} NodeID: {node.node_id}")
        # --- END DEBUG ---

        print("🔍 Stage 2: Fetching metadata from SQLite & filtering inactive/old files.")
        seven_days_ago_iso = (datetime.now() - timedelta(days=7)).isoformat()

        # Create tasks to fetch metadata ONLY for nodes with paths
        path_to_node_map = {}
        tasks = []
        valid_initial_nodes = [] # Keep track of nodes we're fetching meta for
        for node in initial_nodes:
            path = node.metadata.get('path')
            if path:
                tasks.append(asyncio.to_thread(self._metadata_db.get, path))
                path_to_node_map[path] = node # Map path back to the original node object
                valid_initial_nodes.append(node) # Add node to the list we'll iterate later
            else:
                 print(f"  Warning: Node {node.node_id} missing 'path' in metadata during task creation.")

        # --- Check if there are any tasks to run ---
        if not tasks:
            print("   └── No valid paths found in initial candidates to fetch metadata for. Returning empty.")
            return [] # Return empty list immediately

        # --- Fetch metadata ---
        # metadata_results_list will contain results for nodes in valid_initial_nodes
        metadata_results_list = await asyncio.gather(*tasks)

        # Create a dictionary for quick lookup: path -> sqlite_metadata
        # Ensure meta is not None before accessing 'path'
        metadata_map = {meta['path']: meta for meta in metadata_results_list if meta and 'path' in meta}

        valid_node_count = 0
        filtered_nodes_with_bh_data = []

        # --- Loop through VALID nodes for which we fetched metadata ---
        for node in valid_initial_nodes: # Iterate only the nodes we intended to check
            original_metadata = node.metadata
            original_path = original_metadata.get('path') # Path definitely exists here

            # Get the corresponding metadata fetched from SQLite using the map
            meta_from_db = metadata_map.get(original_path)

            # --- DEBUG PRINT: What did SQLite return for this path? ---
            # print(f"  Processing Node {node.node_id} (Path: {original_path}) - SQLite Meta: {meta_from_db}")
            # --- END DEBUG ---

            # 1. Filter out inactive or non-existent metadata from DB
            if not meta_from_db or meta_from_db.get('active', 0) == 0:
                continue

            # 2. Filter by recency using DB data
            modified_at_str = meta_from_db.get('modified_at', '')
            if modified_at_str < seven_days_ago_iso:
                continue

            # 3. Create the dictionary for behavioral data
            extra_info_dict = {
                "total_time_spent_hrs": meta_from_db.get("total_time_spent_hrs", 0.0),
                "access_count": meta_from_db.get("access_count", 0),
                "accessed_at": meta_from_db.get("accessed_at", ""),
                "modified_at": modified_at_str,
                "path_from_extra_info": original_path # Store path in extra_info too
            }

            # 4. Assign ONLY to extra_info
            node.node.extra_info = extra_info_dict

            # 5. BRUTE FORCE FIX: Restore path if it got lost
            if 'path' not in node.metadata or node.metadata.get('path') != original_path:
                 print(f"  WARNING: Path lost for Node {node.node_id}. Restoring...")
                 node.metadata['path'] = original_path

            # 6. Add the node back
            filtered_nodes_with_bh_data.append(node)
            valid_node_count += 1
        # --- END LOOP ---

        print(f"   └── {valid_node_count} candidates remain after filtering & metadata fetch.")

        # --- DEBUG PRINT 2 ---
        print("\nDEBUG (Filtered): Metadata & ExtraInfo of first 5 remaining candidates:")
        for i, filtered_node in enumerate(filtered_nodes_with_bh_data[:5]):
             print(f"  Node {i} Metadata: {filtered_node.metadata}")
             print(f"  Node {i} NodeID: {filtered_node.node_id}")
             print(f"  Node {i} Extra Info: {getattr(filtered_node.node, 'extra_info', 'N/A')}")
        # --- END DEBUG ---

        return filtered_nodes_with_bh_data

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Synchronous wrapper (remains the same)
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._aretrieve(query_bundle), loop)
            return future.result()
        except RuntimeError:
            return asyncio.run(self._aretrieve(query_bundle))
