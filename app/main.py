import os
import glob
import json
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
# 每次重启为了演示数据增强效果，建议清空重建。生产环境请注释掉下面两行。
try: client.delete_collection("simple_rag") 
except: pass 
collection = client.create_collection(name="simple_rag", embedding_function=GreenPTEmbeddingFunction())

app = FastAPI()

class AskReq(BaseModel):
    question: str

# --- 辅助功能：生成文件级元数据 ---
def generate_file_metadata(full_text: str) -> Dict[str, str]:
    """
    调用 GreenPT 为整个 PPT 生成总结和关键词
    """
    # 截断文本防止超长
    truncated_text = full_text[:4000]
    
    system_prompt = (
        "You are a helpful research assistant. "
        "Analyze the provided document text. "
        "Output a JSON object with two keys: "
        "'summary' (a concise summary of the whole file, max 50 words) and "
        "'keywords' (a comma-separated string of top 5 keywords). "
        "Do not include markdown formatting, just raw JSON."
    )
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_text}
        ],
        "stream": False,
        "response_format": {"type": "json_object"} # 尝试请求 JSON 格式
    }
    
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        print("DEBUG: Generating summary and keywords...")
        resp = requests.post(url, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        
        # 尝试解析 JSON
        try:
            data = json.loads(content)
            return {
                "summary": data.get("summary", "N/A"),
                "keywords": data.get("keywords", "N/A")
            }
        except json.JSONDecodeError:
            # 容错：如果模型没返回 JSON，直接存文本
            return {"summary": content[:200], "keywords": "Parsing Failed"}
            
    except Exception as e:
        print(f"Metadata Generation Error: {e}")
        return {"summary": "Error generating summary", "keywords": "Error"}

# --- 核心工具：重排序 (Rerank) ---
def rerank_docs(query: str, documents: List[str], metadatas: List[Dict], top_n=3):
    if not documents:
        return [], []

    url = f"{API_BASE}/rerank"
    payload = {
        "model": "green-rerank",
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False 
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        print(f"DEBUG: Reranking {len(documents)} docs...")
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if resp.status_code != 200:
            print(f"Rerank API Error: {resp.text}, falling back.")
            return documents[:top_n], metadatas[:top_n]

        results = resp.json().get('results', [])
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        final_docs = []
        final_metas = []

        for item in results:
            idx = item['index']
            if 0 <= idx < len(documents):
                final_docs.append(documents[idx])
                final_metas.append(metadatas[idx])
        
        return final_docs, final_metas

    except Exception as e:
        print(f"Rerank Exception: {e}, falling back.")
        return documents[:top_n], metadatas[:top_n]

# --- 接口：Ingest (增强版) ---
@app.post("/ingest")
def ingest():
    files = glob.glob("/app_data/*.pptx")
    if not files: return {"error": "No files found"}
    
    total_slides = 0
    processed_files = 0
    
    for fpath in files:
        try:
            prs = Presentation(fpath)
            fname = os.path.basename(fpath)
            print(f"Processing File: {fname}")
            
            # 1. 提取文件级元数据 (File-level Metadata)
            core_props = prs.core_properties
            author = core_props.author if core_props.author else "Unknown Author"
            # 处理时间格式，防止出错
            created_at = "Unknown Date"
            if core_props.created:
                try: created_at = core_props.created.strftime("%Y-%m-%d")
                except: pass
            
            # 2. 提取全文用于生成总结
            all_text_list = []
            slides_content = [] # 暂存幻灯片内容，避免二次遍历
            
            for i, slide in enumerate(prs.slides):
                texts = [s.text for s in slide.shapes if hasattr(s, "text") and s.text]
                page_text = "\n".join(texts).strip()
                if len(page_text) > 10:
                    all_text_list.append(page_text)
                    slides_content.append({"text": page_text, "page": i+1})
            
            full_text_concat = "\n\n".join(all_text_list)
            
            # 3. AI 生成总结和关键词
            ai_meta = generate_file_metadata(full_text_concat)
            print(f"  > Generated Summary: {ai_meta['summary'][:50]}...")
            
            # 4. 存入数据库 (Chunking + Metadata Injection)
            for item in slides_content:
                # 组合所有元数据
                # 注意：ChromaDB 的 metadata 值通常只能是 String, Int, Float, Bool
                metadata = {
                    "source": fname,
                    "page": item['page'],
                    "author": author,
                    "created_at": created_at,
                    "summary": ai_meta['summary'],   # 每个切片都带上整份 PPT 的总结
                    "keywords": ai_meta['keywords']  # 每个切片都带上整份 PPT 的关键词
                }
                
                collection.add(
                    documents=[item['text']],
                    metadatas=[metadata],
                    ids=[f"{fname}_{item['page']}"]
                )
                total_slides += 1
            
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    return {
        "status": "ok", 
        "files": processed_files, 
        "slides": total_slides,
        "message": "Metadata enrichment complete."
    }

# --- 接口：Ask (展示增强的 Sources) ---
@app.post("/ask")
def ask(req: AskReq):
    # 1. 粗排
    results = collection.query(query_texts=[req.question], n_results=10)
    
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "No info found."}

    # 2. 精排
    top_docs, top_metas = rerank_docs(req.question, results['documents'][0], results['metadatas'][0], top_n=3)

    # 3. 组装 Context (包含更丰富的元数据供 LLM 参考)
    context_parts = []
    for i, (doc, meta) in enumerate(zip(top_docs, top_metas)):
        source_id = i + 1
        # 在 Context 中加入 Summary 和 Keywords，帮助 LLM 更好理解这个片段的背景
        meta_info = (
            f"(ID: {source_id})\n"
            f"File: {meta['source']} (Page {meta['page']})\n"
            f"Author: {meta['author']}, Date: {meta['created_at']}\n"
            f"Context Summary: {meta['summary']}\n"
        )
        part = f"{meta_info}\nContent:\n{doc}"
        context_parts.append(part)
    
    context_str = "\n\n---\n\n".join(context_parts)

    # 4. Prompt (要求引用)
    system_prompt = (
        "You are an expert research assistant. "
        "Answer the user's question strictly based on the provided Context.\n"
        "1. Cite sources using [ID: x] at the end of sentences.\n"
        "2. You can use the 'Context Summary' metadata to understand the broader context of a slide, "
        "but prioritize the 'Content' for specific facts."
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
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return {"error": f"Chat API {resp.status_code}: {resp.text}"}
        
        answer = resp.json()['choices'][0]['message']['content']
        
        return {
            "answer": answer,
            "citations": top_metas # 前端现在可以看到完整的 Author, Created, Keywords, Summary 了！
        }
    except Exception as e:
        return {"error": str(e)}