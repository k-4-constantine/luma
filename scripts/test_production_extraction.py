"""Test production extraction pipeline with real LLM calls."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.chat_service import ChatService
from backend.config import settings


async def test_single_file(file_path: Path, output_dir: Path):
    """Test extraction on a single file using production pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print('='*60)

    # Use REAL services (production)
    embedding_service = EmbeddingService()
    chat_service = ChatService(embedding_service, None)  # No vector store needed for extraction
    processor = DocumentProcessor(embedding_service, chat_service)

    # Process document
    processed_doc = await processor.process_document(file_path)

    if not processed_doc:
        print(f"❌ Failed to process {file_path.name}")
        return

    # Save summary and keywords
    base_name = file_path.stem
    summary_file = output_dir / f"{base_name}_production_summary.json"

    summary_dict = {
        "summary": processed_doc.summary,
        "keywords": processed_doc.keywords
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)

    print(f"✅ Summary: {processed_doc.summary[:100]}...")
    print(f"✅ Keywords: {', '.join(processed_doc.keywords)}")


async def main():
    """Test extraction on all documents using production pipeline."""
    documents_path = Path(settings.documents_path)
    output_dir = Path("extraction_output")
    output_dir.mkdir(exist_ok=True)

    # Find all supported files
    supported_extensions = [".pdf", ".docx", ".pptx"]
    files = [
        f for f in documents_path.iterdir()
        if f.suffix.lower() in supported_extensions
    ]

    print(f"\nTesting PRODUCTION pipeline on {len(files)} documents")
    print(f"Output directory: {output_dir.absolute()}\n")

    for file_path in files:
        try:
            await test_single_file(file_path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("✅ Production testing complete!")
    print('='*60)


if __name__ == "__main__":
    asyncio.run(main())
