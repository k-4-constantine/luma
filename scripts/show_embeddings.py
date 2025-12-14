#!/usr/bin/env python3
"""Display first 5 embeddings for each chunking strategy."""

import asyncio
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.vector_store import VectorStore


async def show_embeddings(show_full_text=False):
    """Display first 5 embeddings for each chunking strategy.

    Args:
        show_full_text: If True, shows full content text. If False, shows preview only.
    """
    print("📊 Displaying embeddings by chunking strategy...")
    print("=" * 80)

    vector_store = VectorStore()

    try:
        # Connect to Weaviate
        await vector_store.connect()
        print("✅ Connected to Weaviate\n")

        # Get total count
        total_count = await vector_store.get_document_count()
        print(f"📈 Total chunks in database: {total_count}\n")

        if total_count == 0:
            print("ℹ️  No documents found in database")
            return

        # Query all documents with their vectors
        collection = vector_store.client.collections.get(vector_store.collection_name)

        # Fetch all documents
        response = collection.query.fetch_objects(
            include_vector=True,
            limit=total_count
        )

        # Group by chunking strategy
        strategy_groups = defaultdict(list)
        for obj in response.objects:
            props = obj.properties
            strategy = props.get("chunking_strategy", "unknown")
            content = props.get("content", "")
            strategy_groups[strategy].append({
                "chunk_id": props.get("chunk_id"),
                "document_id": props.get("document_id"),
                "title": props.get("title"),
                "chunk_index": props.get("chunk_index"),
                "token_count": props.get("token_count"),
                "content": content,
                "content_preview": content[:200] if not show_full_text else content,
                "vector": obj.vector,
            })

        # Display first 5 for each strategy
        strategies = ["whole_file", "pages", "max_tokens"]
        strategy_names = {
            "whole_file": "WHOLE FILE",
            "pages": "BY PAGES",
            "max_tokens": "MAX TOKENS"
        }

        for strategy in strategies:
            chunks = strategy_groups.get(strategy, [])
            display_name = strategy_names.get(strategy, strategy.upper())

            print("=" * 80)
            print(f"📑 STRATEGY: {display_name}")
            print(f"   Total chunks: {len(chunks)}")
            print("-" * 80)

            if not chunks:
                print("   ℹ️  No chunks found for this strategy\n")
                continue

            # Show first 5
            for i, chunk in enumerate(chunks[:5], 1):
                print(f"\n   [{i}] Chunk {chunk['chunk_index']}")
                print(f"       Document: {chunk['title']}")
                print(f"       Tokens: {chunk['token_count']}")

                # Display content
                content_display = chunk['content_preview']
                if show_full_text:
                    print(f"       Content (full):")
                    # Indent each line of content
                    for line in content_display.split('\n'):
                        print(f"           {line}")
                else:
                    # Show preview with ellipsis
                    content_display = content_display.replace('\n', ' ')
                    if len(chunk['content']) > 200:
                        content_display += "..."
                    print(f"       Content: {content_display}")

                # Display vector info
                vector = chunk['vector']
                if isinstance(vector, dict):
                    # Handle named vectors
                    print(f"       Vector: {vector}")
                elif vector:
                    vector_preview = vector[:5] if len(vector) > 5 else vector
                    print(f"       Vector dim: {len(vector)}")
                    print(f"       Vector preview: [{', '.join(f'{v:.4f}' for v in vector_preview)}, ...]")
                else:
                    print(f"       Vector: None")

            if len(chunks) > 5:
                print(f"\n   ... and {len(chunks) - 5} more chunks")
            print()

        # Summary
        print("=" * 80)
        print("📊 SUMMARY")
        print("-" * 80)
        for strategy in strategies:
            display_name = strategy_names.get(strategy, strategy.upper())
            count = len(strategy_groups.get(strategy, []))
            print(f"   {display_name:20s}: {count:5d} chunks")
        print(f"   {'TOTAL':20s}: {total_count:5d} chunks")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await vector_store.disconnect()
        print("\n✅ Disconnected from Weaviate")


if __name__ == "__main__":
    # Check for --full flag
    show_full = "--full" in sys.argv or "-f" in sys.argv

    if show_full:
        print("ℹ️  Full text mode enabled\n")
    else:
        print("ℹ️  Preview mode (use --full or -f to show complete text)\n")

    asyncio.run(show_embeddings(show_full_text=show_full))
