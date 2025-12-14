# LUMC PPT RAG 使用指南

## 服务地址

- **API 文档**: http://localhost:8000/docs
- **API 基础地址**: http://localhost:8000

## 快速开始

### 1. 确保服务运行

```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 配置环境变量

确保 `.env` 文件包含以下配置：

```env
GREENPT_API_BASE=https://api.greenpt.ai/v1
GREENPT_API_KEY=sk-your_api_key_here
GREENPT_MODEL=green-l
```

### 3. 导入数据（Ingest）

将 PPTX 文件放入 `data/` 目录，然后调用导入接口：

**使用 curl:**
```bash
curl -X POST http://localhost:8000/ingest
```

**使用 PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method POST
```

**使用 Python:**
```python
import requests

response = requests.post("http://localhost:8000/ingest")
print(response.json())
```

**响应示例:**
```json
{
  "status": "success",
  "processed_files": 2
}
```

### 4. 提问（Ask）

调用问答接口进行查询：

**使用 curl:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "你的问题"}'
```

**使用 PowerShell:**
```powershell
$body = @{
    question = "Why do we need surveillance for pancreatic cancer and how?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method POST -Body $body -ContentType "application/json"
```

**使用 Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Why do we need surveillance for pancreatic cancer and how?"}
)
print(response.json())
```

**响应示例:**
```json
{
  "answer": "根据文档内容...",
  "retrieved_count": 5,
  "reranked_sources": ["文档片段1", "文档片段2", "文档片段3"]
}
```

## API 端点说明

### POST /ingest

导入 `data/` 目录下的所有 PPTX 文件到向量数据库。

**功能:**
1. 扫描 `data/` 目录下的所有 `.pptx` 文件
2. 提取每张幻灯片的文本和元数据
3. 使用 GreenPT 生成摘要和关键词
4. 存储到 ChromaDB 向量数据库

**响应:**
- `status`: "success" 或 "warning"
- `processed_files`: 处理的文件数量

### POST /ask

基于已导入的知识库回答问题。

**请求体:**
```json
{
  "question": "你的问题"
}
```

**处理流程:**
1. **混合检索**: 使用向量搜索 + BM25 关键词搜索
2. **重排序**: 使用 GreenPT 模型对候选文档进行智能排序
3. **生成答案**: 基于最相关的文档生成答案

**响应:**
```json
{
  "answer": "生成的答案",
  "retrieved_count": 5,
  "reranked_sources": ["相关文档1", "相关文档2", "相关文档3"]
}
```

## 使用 API 文档界面

最简单的方式是使用 FastAPI 自动生成的交互式文档：

1. 打开浏览器访问: http://localhost:8000/docs
2. 点击 `/ingest` 端点，点击 "Try it out"，然后点击 "Execute"
3. 点击 `/ask` 端点，输入你的问题，然后执行

## 常见问题

### Q: 导入后没有找到文件？

**A:** 确保：
- PPTX 文件在 `data/` 目录下
- 文件扩展名是 `.pptx`（不是 `.ppt`）
- 文件有读取权限

### Q: 提问返回 "No relevant information found"？

**A:** 可能原因：
- 还没有导入数据（先调用 `/ingest`）
- 问题与文档内容不相关
- 尝试用英文或更具体的问题

### Q: 服务无法启动？

**A:** 检查：
- `.env` 文件中的 `GREENPT_API_KEY` 是否正确
- Docker 是否正常运行
- 查看日志: `docker-compose logs`

### Q: 如何重新导入数据？

**A:** 
- 删除 `chroma_db/` 目录（会清空数据库）
- 或者直接再次调用 `/ingest`（会添加新数据，不会删除旧数据）

## 数据目录结构

```
LUMA/
├── data/              # 放置 PPTX 文件的地方
│   └── *.pptx
├── chroma_db/         # 向量数据库存储（自动创建）
└── app/               # 应用代码
```

## 高级用法

### 查看服务日志

```bash
# 实时查看日志
docker-compose logs -f

# 查看最后 50 行
docker-compose logs --tail=50
```

### 重启服务

```bash
docker-compose restart
```

### 停止服务

```bash
docker-compose down
```

### 重新构建（代码修改后）

```bash
docker-compose up --build -d
```

## 技术架构

- **FastAPI**: Web 框架
- **ChromaDB**: 向量数据库（使用 GreenPT embedding）
- **BM25**: 关键词搜索
- **GreenPT API**: 
  - 嵌入模型（`green-embedding`）
  - 生成模型（`green-l`/`green-r`/`green-s`）
  - 重排序功能

## 下一步

- 添加更多文件类型支持（PDF, DOCX）
- 实现批量导入接口
- 添加用户认证
- 优化搜索算法


