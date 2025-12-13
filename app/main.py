import os
import glob
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pptx import Presentation
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv()
API_KEY = os.getenv("GREENPT_API_KEY")
API_BASE = os.getenv("GREENPT_API_URL", "https://api.greenpt.ai/v1").rstrip('/')
# [关键] 使用 raw 模型
CHAT_MODEL = os.getenv("GREENPT_MODEL", "green-l-raw") 

if not API_KEY:
    print("FATAL: GREENPT_API_KEY not set.")

# 2. Embedding 函数 (使用 green-embedding)
class GreenPTEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        url = f"{API_BASE}/embeddings"
        payload = {
            "model": "green-embedding", # 列表里存在的模型
            "input": input,
            "encoding_format": "float"
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"Embedding Error: {resp.text}")
                return []
            data = resp.json()
            sorted_data = sorted(data['data'], key=lambda x: x['index'])
            return [item['embedding'] for item in sorted_data]
        except Exception as e:
            print(f"Embedding Exception: {e}")
            raise e

# 3. 数据库初始化
client = chromadb.PersistentClient(path="/chroma_data")
try: client.delete_collection("simple_rag")
except: pass # 每次重启清空重建，保证干净
collection = client.create_collection(name="simple_rag", embedding_function=GreenPTEmbeddingFunction())

app = FastAPI()

class AskReq(BaseModel):
    question: str

# 4. 读取 PPT
@app.post("/ingest")
def ingest():
    files = glob.glob("/app_data/*.pptx")
    count = 0
    for fpath in files:
        try:
            prs = Presentation(fpath)
            fname = os.path.basename(fpath)
            print(f"Processing: {fname}")
            for i, slide in enumerate(prs.slides):
                texts = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text]
                content = "\n".join(texts).strip()
                if len(content) > 10:
                    collection.add(
                        documents=[content],
                        metadatas=[{"source": fname, "page": i+1}],
                        ids=[f"{fname}_{i}"]
                    )
                    count += 1
        except Exception as e:
            print(f"Error: {e}")
    return {"status": "ok", "slides": count}

# 5. 提问 (使用 green-l-raw)
@app.post("/ask")
def ask(req: AskReq):
    # 检索
    results = collection.query(query_texts=[req.question], n_results=3)
    if not results['documents'][0]:
        return {"answer": "No info found."}
    
    context = "\n---\n".join(results['documents'][0])
    
    # [关键] 因为是 raw 模型，我们可以放心使用 system role
    payload = {
        "model": CHAT_MODEL, 
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Answer the question strictly based on the provided Context."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {req.question}"
            }
        ],
        "stream": False
    }
    
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    print(f"DEBUG: Asking {CHAT_MODEL}...")
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    
    if resp.status_code != 200:
        return {"error": f"API {resp.status_code}: {resp.text}"}
    
    return {
        "answer": resp.json()['choices'][0]['message']['content'],
        "sources": results['metadatas'][0]
    }