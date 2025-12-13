import os
import glob
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pptx import Presentation
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# --- 配置加载 ---
load_dotenv()
API_KEY = os.getenv("GREENPT_API_KEY")
API_BASE = os.getenv("GREENPT_API_URL", "https://api.greenpt.ai/v1").rstrip('/')
CHAT_MODEL = os.getenv("GREENPT_MODEL", "green-l-raw") 

if not API_KEY:
    print("FATAL: GREENPT_API_KEY not set.")

# --- GreenPT Embedding 函数 ---
class GreenPTEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        url = f"{API_BASE}/embeddings"
        payload = {
            "model": "green-embedding",
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

# --- 数据库初始化 ---
client = chromadb.PersistentClient(path="/chroma_data")
# (生产环境建议去掉这行 delete，以免重启丢失数据)
try: client.delete_collection("simple_rag") 
except: pass 
collection = client.create_collection(name="simple_rag", embedding_function=GreenPTEmbeddingFunction())

app = FastAPI()

class AskReq(BaseModel):
    question: str

# --- 核心工具：重排序 (Rerank) ---
def rerank_docs(query: str, documents: List[str], metadatas: List[Dict], top_n=3):
    """
    使用 GreenPT Rerank 模型对检索结果进行精排
    """
    if not documents:
        return [], []

    url = f"{API_BASE}/rerank"
    payload = {
        "model": "green-rerank",
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False # 我们只拿索引，省流量
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        print(f"DEBUG: Reranking {len(documents)} docs...")
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        # 如果 Rerank API 挂了或者报错，降级返回原始的前 N 个
        if resp.status_code != 200:
            print(f"Rerank API Error: {resp.text}, falling back.")
            return documents[:top_n], metadatas[:top_n]

        results = resp.json().get('results', [])
        # 按分数从高到低排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        final_docs = []
        final_metas = []

        for item in results:
            idx = item['index']
            # 确保索引有效
            if 0 <= idx < len(documents):
                final_docs.append(documents[idx])
                final_metas.append(metadatas[idx])
        
        return final_docs, final_metas

    except Exception as e:
        print(f"Rerank Exception: {e}, falling back.")
        return documents[:top_n], metadatas[:top_n]

# --- 接口：Ingest ---
@app.post("/ingest")
def ingest():
    files = glob.glob("/app_data/*.pptx")
    if not files: return {"error": "No files found"}
    
    count = 0
    for fpath in files:
        try:
            prs = Presentation(fpath)
            fname = os.path.basename(fpath)
            print(f"Processing: {fname}")
            for i, slide in enumerate(prs.slides):
                texts = [s.text for s in slide.shapes if hasattr(s, "text") and s.text]
                content = "\n".join(texts).strip()
                if len(content) > 15: # 稍微提高一点阈值过滤空页
                    collection.add(
                        documents=[content],
                        metadatas=[{"source": fname, "page": i+1}],
                        ids=[f"{fname}_{i}"]
                    )
                    count += 1
        except Exception as e:
            print(f"Error: {e}")
    return {"status": "ok", "slides": count}

# --- 接口：Ask (带引用 + 重排序) ---
@app.post("/ask")
def ask(req: AskReq):
    # 1. 粗排：从数据库多捞一点 (Top 10)
    results = collection.query(query_texts=[req.question], n_results=10)
    
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "No info found."}

    raw_docs = results['documents'][0]
    raw_metas = results['metadatas'][0]

    # 2. 精排：使用 GreenPT Rerank 挑出最好的 Top 3
    top_docs, top_metas = rerank_docs(req.question, raw_docs, raw_metas, top_n=3)

    # 3. 组装带引用的 Context
    # 格式：
    # [Source ID: 1] (File: xxx.pptx, Page: 5)
    # Content: ...
    context_parts = []
    for i, (doc, meta) in enumerate(zip(top_docs, top_metas)):
        source_id = i + 1
        meta_info = f"(File: {meta['source']}, Page: {meta['page']})"
        part = f"[Source ID: {source_id}] {meta_info}\nContent:\n{doc}"
        context_parts.append(part)
    
    context_str = "\n\n---\n\n".join(context_parts)

    # 4. Prompt Engineering (强制引用)
    system_prompt = (
        "You are an expert research assistant for LUMC. "
        "Answer the user's question strictly based on the provided Context below.\n"
        "RULES:\n"
        "1. If the answer is not in the context, say 'I don't know'.\n"
        "2. **CITATION IS MANDATORY**: Every time you make a statement, you MUST cite the Source ID at the end of the sentence.\n"
        "   Example: 'Pancreatic cancer survival is low [Source ID: 1]. However, new screenings help [Source ID: 2].'\n"
        "3. Do not make up source IDs."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {req.question}"

    payload = {
        "model": CHAT_MODEL, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    print(f"DEBUG: Asking {CHAT_MODEL}...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return {"error": f"Chat API {resp.status_code}: {resp.text}"}
        
        answer = resp.json()['choices'][0]['message']['content']
        
        # 返回给前端的数据包含结构化的 sources，方便前端做悬浮展示等
        return {
            "answer": answer,
            "citations": [
                {"id": i+1, "file": m['source'], "page": m['page']} 
                for i, m in enumerate(top_metas)
            ]
        }
    except Exception as e:
        return {"error": str(e)}