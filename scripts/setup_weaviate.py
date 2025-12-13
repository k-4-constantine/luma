#!/usr/bin/env python3
"""Setup Weaviate schema for Luma RAG application."""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.vector_store import VectorStore


async def main():
    """Initialize Weaviate schema."""
    print("🔧 Setting up Weaviate schema...")
    
    vector_store = VectorStore()
    
    try:
        # Connect to Weaviate
        await vector_store.connect()
        print("✅ Connected to Weaviate")
        
        # Create schema
        await vector_store.create_schema()
        print("✅ Schema created successfully!")
        
        # Verify by getting document count (should be 0)
        count = await vector_store.get_document_count()
        print(f"✅ Document count: {count} (empty collection)")
        
    except Exception as e:
        print(f"❌ Error setting up Weaviate: {e}")
        sys.exit(1)
    finally:
        await vector_store.disconnect()
        print("✅ Disconnected from Weaviate")


if __name__ == "__main__":
    asyncio.run(main())