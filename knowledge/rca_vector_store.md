# Root Cause Analysis: Weaviate v4 insert_many Error

## Problem Statement

When running `scripts/process_documents.py`, we encountered the following error:
```
_DataCollectionExecutor.insert_many() got an unexpected keyword argument 'vectors'
```

Previously, we also saw:
```
_DataCollectionExecutor.insert_many() got an unexpected keyword argument 'properties'
```

## Root Cause

The `vector_store.py` implementation was using an incorrect API for Weaviate v4's `insert_many()` method.

### Incorrect Implementation (Current)
```python
# This is WRONG
collection.data.insert_many(
    objects=objects,
    vectors=embeddings
)
```

### Why This Failed

The Weaviate v4 Python client's `insert_many()` method does NOT accept separate `objects` and `vectors` keyword arguments. Instead, it expects a list of `DataObject` instances where each object contains both its properties AND its vector together.

## Correct API Usage

According to the official Weaviate documentation, the correct approach is:

### Option 1: Using DataObject (Recommended)

```python
import weaviate.classes as wvc

# Create list of DataObject instances
data_objects = []
for properties, vector in zip(objects, vectors):
    data_objects.append(
        wvc.data.DataObject(
            properties=properties,
            vector=vector
        )
    )

# Insert the list
collection.data.insert_many(data_objects)
```

### Option 2: Using Batch API (Alternative)

```python
with collection.batch.fixed_size(batch_size=200) as batch:
    for properties, vector in zip(objects, vectors):
        batch.add_object(
            properties=properties,
            vector=vector
        )
```

## Key Learnings

1. **Weaviate v4 Design Philosophy**: Vectors are treated as first-class attributes of objects, not separate parameters
2. **DataObject Class**: The `wvc.data.DataObject` class is the modern way to construct objects with custom vectors
3. **API Breaking Changes**: Weaviate v4 introduced significant API changes from v3, requiring code updates for migration

## Solution Implementation

We need to update `backend/services/vector_store.py` to:
1. Import `weaviate.classes as wvc`
2. Construct `DataObject` instances that combine properties and vectors
3. Pass the list of DataObjects to `insert_many()`

## References

- [Weaviate Custom Vectors Guide](https://docs.weaviate.io/weaviate/starter-guides/custom-vectors)
- [Weaviate Batch Import Documentation](https://docs.weaviate.io/weaviate/manage-data/import)
- [Weaviate Python Client v4 Release](https://weaviate.io/blog/py-client-v4-release)

## Timeline

1. **Initial Issue**: `properties` parameter not recognized
2. **First Fix Attempt**: Changed `properties` to `objects` (still incorrect)
3. **Second Issue**: `vectors` parameter not recognized
4. **Root Cause Discovery**: Need to use `DataObject` class
5. **Solution**: Combine properties and vectors in DataObject instances

## Impact

- **Severity**: HIGH - Complete blocking of document ingestion
- **Affected Component**: `backend/services/vector_store.py::add_chunks()`
- **Workaround**: None until fixed
- **Fix Complexity**: LOW - Simple API usage correction

## Additional Issue: Resource Cleanup

### Problem
After fixing the main issue, ResourceWarnings appeared about unclosed network transports:
```
ResourceWarning: unclosed transport <_SelectorSocketTransport>
ResourceWarning: unclosed <socket.socket>
```

### Root Cause
The `AsyncOpenAI` clients in `ChatService` and `EmbeddingService` were not being properly closed, leaving HTTP connections open.

### Solution
1. Added `async def close()` methods to:
   - `backend/services/chat_service.py`
   - `backend/services/embedding_service.py`
2. Updated `scripts/process_documents.py` finally block to close all services:
   ```python
   finally:
       await vector_store.disconnect()
       await chat_service.close()
       await embedding_service.close()
   ```

### Best Practice
Always close async HTTP clients to prevent resource leaks. The AsyncOpenAI client maintains connection pools that need explicit cleanup.
