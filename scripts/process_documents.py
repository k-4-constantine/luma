#!/usr/bin/env python3
"""Process all documents and store them in Weaviate vector store."""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.chat_service import ChatService
from backend.config import settings


async def process_all_documents():
    """Process all documents in Example-Files directory and store in Weaviate."""
    print("📚 Processing documents for Weaviate vector store...")
    print("=" * 60)
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    chat_service = ChatService()

    try:
        # Connect to Weaviate
        await vector_store.connect()
        print("✅ Connected to Weaviate")

        # Initialize document processor
        document_processor = DocumentProcessor(embedding_service, chat_service)
        
        # Get documents path
        documents_path = Path(settings.documents_path)
        
        # Find all supported files
        supported_extensions = [".pdf", ".docx", ".pptx"]
        files = [
            f for f in documents_path.iterdir()
            if f.suffix.lower() in supported_extensions
        ]
        
        print(f"📁 Found {len(files)} documents to process")
        print("-" * 60)
        
        total_chunks = 0
        total_documents = 0
        
        # Process each file
        for file_path in files:
            try:
                print(f"📄 Processing: {file_path.name}")
                
                # Process document (extract, summarize, chunk)
                processed_doc = await document_processor.process_document(file_path)
                
                if not processed_doc:
                    print(f"❌ Failed to process {file_path.name}")
                    continue
                
                # Collect ALL chunks (all 3 strategies)
                all_chunks = (
                    processed_doc.whole_file_chunks +
                    processed_doc.page_chunks +
                    processed_doc.token_chunks
                )
                
                print(f"   ✅ Extracted {len(all_chunks)} chunks")
                print(f"   ✅ Summary: {processed_doc.summary[:80]}...")
                print(f"   ✅ Keywords: {', '.join(processed_doc.keywords[:5])}...")
                
                # Generate embeddings
                print(f"   🔢 Generating embeddings...")
                chunk_texts = [chunk.content for chunk in all_chunks]
                embeddings = await embedding_service.embed_texts(chunk_texts)
                print(f"   ✅ Generated {len(embeddings)} embeddings")
                
                # Store in Weaviate
                print(f"   💾 Storing in Weaviate...")
                await vector_store.add_chunks(all_chunks, embeddings)
                print(f"   ✅ Stored successfully")
                
                total_chunks += len(all_chunks)
                total_documents += 1
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
            
            print("-" * 60)
        
        # Final summary
        print(f"\n🎉 Processing complete!")
        print(f"   ✅ Processed {total_documents}/{len(files)} documents")
        print(f"   ✅ Stored {total_chunks} chunks in Weaviate")
        
        # Verify count
        final_count = await vector_store.get_document_count()
        print(f"   ✅ Verified: {final_count} chunks in vector store")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await vector_store.disconnect()
        await chat_service.close()
        await embedding_service.close()
        print("✅ Disconnected from all services")


if __name__ == "__main__":
    asyncio.run(process_all_documents())