#!/bin/bash

# Check Vector Store Status Script
# Usage: ./scripts/check_vector_store.sh

echo "🔍 Checking Weaviate Vector Store Status"
echo "========================================"

# Check if Weaviate is running
echo "1. Checking Weaviate container status..."
if docker ps | grep -q luma-weaviate; then
    echo "✅ Weaviate container is running"
else
    echo "❌ Weaviate container is NOT running"
    exit 1
fi

# Check Weaviate health
echo ""
echo "2. Checking Weaviate health endpoint..."
if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
    echo "✅ Weaviate is healthy and ready"
else
    echo "❌ Weaviate is NOT ready"
    exit 1
fi

# Check schema exists
echo ""
echo "3. Checking DocumentChunk schema..."
if curl -s http://localhost:8080/v1/schema | grep -q "DocumentChunk"; then
    echo "✅ DocumentChunk schema exists"
else
    echo "❌ DocumentChunk schema NOT found"
    exit 1
fi

# Check document count using Python
echo ""
echo "4. Checking document count..."
cd /home/pjotterb/repos/luma
COUNT=$(uv run python -c "
import asyncio
from backend.services.vector_store import VectorStore

async def check():
    vs = VectorStore()
    await vs.connect()
    count = await vs.get_document_count()
    print(count)
    await vs.disconnect()

asyncio.run(check())
" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "✅ Document count: $COUNT chunks"
    if [ "$COUNT" -eq "0" ]; then
        echo "ℹ️  Vector store is empty - ready for document processing"
    else
        echo "📚 Vector store contains $COUNT document chunks"
    fi
else
    echo "❌ Could not get document count"
    exit 1
fi

# Show sample query
echo ""
echo "5. Sample vector search query:"
echo ""
echo "curl -X POST http://localhost:8080/v1/graphql \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"query\": \"{Get {DocumentChunk(limit: 3) {chunk_id title _additional {vector}}}}\"}'"

echo ""
echo "✅ Vector store check complete!"
echo "========================================"

exit 0