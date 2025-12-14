#!/usr/bin/env python3
"""Test knowledge graph generation."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.services.vector_store import VectorStore
from backend.services.knowledge_graph_service import KnowledgeGraphService


async def test():
    """Test knowledge graph generation."""
    print("Testing knowledge graph generation...")
    
    vector_store = VectorStore()
    await vector_store.connect()
    
    # Check if we have data
    count = await vector_store.get_document_count()
    print(f"Total chunks in store: {count}")
    
    if count == 0:
        print("No data in store!")
        return
    
    # Get a sample object to check structure
    collection = vector_store.client.collections.get(vector_store.collection_name)
    response = collection.query.fetch_objects(limit=1)
    
    if response.objects:
        obj = response.objects[0]
        props = obj.properties
        print(f"Sample object properties: {list(props.keys())}")
        print(f"Sample file_path: {props.get('file_path')}")
        print(f"Sample keywords: {props.get('keywords')}")
        print(f"Has vector: {obj.vector is not None}")
    
    # Test graph generation
    graph_service = KnowledgeGraphService(vector_store)
    result = await graph_service.generate_graph()
    
    print(f"\nGraph result:")
    print(f"  Nodes: {len(result['nodes'])}")
    print(f"  Links: {len(result['links'])}")
    print(f"  Categories: {len(result['categories'])}")
    
    if result['nodes']:
        print(f"\nFirst node: {result['nodes'][0]}")
    else:
        print("\nNo nodes generated!")
    
    await vector_store.disconnect()


if __name__ == "__main__":
    asyncio.run(test())
