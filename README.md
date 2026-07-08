# PDF-Learning-Assistant
基于 LangChain + LangGraph 的 RAG 智能文档问答 Agent
# -
基于 LangChain + LangGraph 的 RAG 智能文档问答 Agent
## ✨ 核心特性

### 1. 高级 RAG 检索管道

| 阶段 | 技术 | 说明 |
|------|------|------|
| **文档分块** | `SemanticChunker` | 基于 Embedding 语义断点检测，替代固定长度切割，保持语义完整性 |
| **粗召回 (Dense)** | `BAAI/bge-small-zh-v1.5` | 中文优化的轻量 Embedding 模型（512维） |
| **粗召回 (Sparse)** | BM25 + jieba 分词 | 关键词级精准匹配，弥补向量检索的词汇盲区 |
| **融合** | RRF (Reciprocal Rank Fusion) | 排名级融合，天然解决 Dense/BM25 分数量纲不一致问题 |
| **精排** | Cross-Encoder | 全交互注意力打分，比双塔模型更理解语义匹配 |
| **生成** | DeepSeek-Chat | 基于检索上下文生成带页码引用的回答 |

### 2. 三种检索策略
basic │ 直接用问题向量检索，零额外成本
MQE │ LLM 改写为 3 个不同角度查询，并行搜索 
适合：问题表述模糊、单一角度覆盖不全 
 HyDE|先生成"假设答案"，用答案向量去搜
  适合：问题+文档语义差异大时架桥
  
### 3. 智能问题路由器

LLM 驱动的三分类路由，避免无效检索：

- **RAG 路径** —问题涉及文档内容，触发检索管道
- **LLM Direct 路径** —通用知识 / 闲聊，直接用 LLM 知识回答
- **Out-of-Scope 路径** —超出文档范围，礼貌拒绝

### 4. 双记忆架构 (Dual Memory)

| 记忆类型 | 存储内容 | 类比 |
|----------|----------|------|
| **情景记忆** (Episodic) | 对话事件、文档加载、笔记记录 | "发生过什么" |
| **语义记忆** (Semantic) | 用户偏好、概念知识、自定义规则 | "我知道了什么" |

- 均基于 **ChromaDB** 向量数据库持久化存储
- 支持语义搜索和历史回溯
- 程序关闭后数据不丢失

### 5. Agent 工具集成

将检索、记忆、历史封装为 3 个 LangChain Tool，Agent 自主决策调用时机：

```python
@tool
def search_document(query: str) -> str: ...   # 文档检索

@tool
def remember_fact(content: str) -> str: ...    # 记忆存储

@tool
def recall_context(query: str) -> str: ...     # 历史回溯
