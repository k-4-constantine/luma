#!/usr/bin/env python3
"""Clear all documents from Weaviate vector store."""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.vector_store import VectorStore


async def clear_database():
    """Clear all documents from Weaviate."""
    print("🗑️  Clearing Weaviate vector store...")
    print("=" * 60)

    vector_store = VectorStore()

    try:
        # Connect to Weaviate
        await vector_store.connect()
        print("✅ Connected to Weaviate")

        # Get count before clearing
        before_count = await vector_store.get_document_count()
        print(f"📊 Current document count: {before_count}")

        if before_count == 0:
            print("ℹ️  Database is already empty")
            return

        # Ask for confirmation
        response = input("\n⚠️  Are you sure you want to delete all documents? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Operation cancelled")
            return

        # Clear all data
        print("\n🗑️  Clearing all documents...")
        await vector_store.clear_all()

        # Verify count after clearing
        after_count = await vector_store.get_document_count()
        print(f"✅ Deleted {before_count} documents")
        print(f"✅ Verified: {after_count} documents remaining")

        if after_count == 0:
            print("\n🎉 Database cleared successfully!")
        else:
            print(f"\n⚠️  Warning: {after_count} documents still remain")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await vector_store.disconnect()
        print("✅ Disconnected from Weaviate")


if __name__ == "__main__":
    asyncio.run(clear_database())
