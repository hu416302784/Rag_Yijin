# 易经 GraphRAG 问答助手（GraphDB 版）

一个基于 **RDF + GraphDB + Chroma + 大语言模型** 的《易经》问答助手，支持两种交互模式：

- **🔍 匹配卦象**：描述你的处境，系统从 64 卦中匹配最相关的 1-3 卦并给出解读。
- **🎲 起卦并引导**：模拟铜钱起卦，得到本卦、动爻、之卦，并以启发式提问引导你自行思考。

## ✨ 功能特点

- **图谱检索（GraphRAG）**：通过 SPARQL 从 GraphDB 查询卦、爻、彖辞、象辞、错卦、综卦等结构化知识。
- **向量检索**：用 Chroma + `BAAI/bge-small-zh-v1.5` 对卦辞、爻辞、现代解读、情境标签做语义检索。
- **双路召回**：图谱上下文 + 文本切片共同作为 LLM 的参考资料，回答更贴合原典。
- **起卦模块**：三枚铜钱法模拟六爻，自动推算本卦、动爻与之卦。
- **情境标签**：通过外部 `yijing_tags.json` 配置每个卦的情境标签与关键词映射，方便扩展。
- **流式输出**：LLM 回答逐字返回，并附带参考片段与相似度。

## 📁 项目结构
```
├── RAGDemo-Gradio-GraphDB.py # 主程序（单文件）
├── yijing_rdf.py # RDF 解析模块，提供 load_yijing_from_rdf()
├── yijing.rdf # 易经 RDF 数据（Protege 导出）
├── yijing_tags.json # 情境标签与关键词映射（可外部编辑）
├── requirements.txt # 依赖列表
├── README.md
└── chroma_yijing_demo/ # Chroma 持久化目录（首次运行自动生成）
```


## 🔧 环境要求

- Python 3.9+
- [GraphDB](https://graphdb.ontotext.com/) 已启动，且满足：
  - 地址为 `http://localhost:7200`
  - 已创建 repository，ID 为 `yijing`
  - `yijing.rdf` 已导入该 repository
- 一个兼容 OpenAI 接口的 LLM 服务（默认使用 DeepSeek）

## 📦 安装依赖

```bash
pip install gradio chromadb sentence-transformers SPARQLWrapper openai rdflib
```
或使用 requirements.txt：
```bash
pip install -r requirements.txt
```

## 🚀 运行
1.确保 GraphDB 已启动，且 yijing repository 中已导入 yijing.rdf。

2.在项目目录下运行
```bash
python RAGDemo-Gradio-GraphDB.py
```

3.终端会输出加载信息，例如：
```
✅ 本地 RDF 元数据：64 卦，384 爻
✅ 文本切片：xxx 条（含爻辞）
✅ 可起卦：64 个二进制 → 卦名映射
✅ GraphDB endpoint：http://localhost:7200/repositories/yijing
✅ GraphDB 连通性检查通过（返回 1 条）
```

4.浏览器打开 Gradio 提供的本地地址（通常是 http://127.0.0.1:7860）

## 🔑 配置 API Key
推荐用环境变量，避免把密钥写进代码：

```bash
# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="你的key"

# Windows (CMD)
set DEEPSEEK_API_KEY=你的key

# macOS / Linux
export DEEPSEEK_API_KEY="你的key"
```

也可以在界面左侧的输入框里直接填写。

## 🖥️ 界面说明
左侧参数区：

```
| 参数              | 说明                               |
| ----------------- | --------------------------------- |
| DeepSeek API Key  | LLM 密钥，建议用环境变量           |
| Base URL          | 默认为 `https://api.deepseek.com` |
| 对话模型           | 默认 `deepseek-chat`              |
| Temperature       | 生成随机性，默认 0.3               |
| top_k 检索条数     | 向量检索返回条数，默认 3           |
| 🗑️ 清空对话历史    | 清空当前会话                      |
```

右侧对话区：

```
输入问题：描述你的处境或困惑。

🔍 匹配卦象：走 GraphRAG 检索 + LLM 解读。

🎲 起卦并引导：先起卦，再让 LLM 用启发式问题引导你思考。

起卦结果：显示本卦、动爻、之卦、六爻数字。

参考片段：显示召回的知识片段及相似度。
```

## 🗂️ 数据文件说明
yijing.rdf
从 Protege 导出的易经本体数据，包含：
```rdf
Hexagram（六十四卦）：卦辞、象辞、彖辞、序号、卦象编码、上下卦等

Yao（爻）：爻题、爻位、爻辞、小象辞、阴阳

Trigram（八卦）：卦德、自然、五行、方位

关系：hasCuoGua（错卦）、hasZongGua（综卦）、hasUpperTrigram、hasLowerTrigram、hasYao
```

yijing_tags.json
可选的外部配置文件，结构如下：
```json
{
  "GUA_TAGS": {
    "乾": ["刚健", "进取", "领导"],
    "坤": ["包容", "承载", "顺势"]
  },
  "SITUATION_KEYWORDS": {
    "转型": ["革", "鼎", "渐"],
    "创业": ["屯", "乾", "大畜"]
  }
}
```

GUA_TAGS：每个卦的情境标签，会作为独立切片进入向量库。

SITUATION_KEYWORDS：用户问题中的关键词 → 候选卦，用于图谱召回。

若文件不存在，程序会以空配置启动，不影响运行。

## 🧠 工作原理
```text
用户提问
   │
   ├─ 关键词/卦名识别 ──► 命中卦
   │                        │
   │                        ├─ GraphDB SPARQL 查询（卦、爻、错卦、综卦…）
   │                        └─ Chroma 向量检索（该卦相关文本）
   │
   └─ 未命中卦 ──► 全局向量检索 + 64 卦总览
                        │
                        ▼
                 拼接为 context
                        │
                        ▼
              LLM（DeepSeek）生成回答
                        │
                        ▼
              流式输出 + 参考片段
```

起卦模式额外经过：铜钱模拟 → 六爻数字 → 本卦/动爻/之卦 → 分别取图谱上下文 → 引导式提示词

## ⚠️ 常见问题
1. 提示 SPARQL 查询失败

检查 GraphDB 是否启动、repository ID 是否为 yijing、yijing.rdf 是否已导入。可用浏览器访问 http://localhost:7200/repositories/yijing 确认。

2. 首次运行很慢

第一次会下载 BAAI/bge-small-zh-v1.5 模型并构建 Chroma 索引，属于正常现象。之后会走本地缓存。

3. 国内下载模型慢

代码已设置 HF_ENDPOINT=https://hf-mirror.com，如仍慢可检查网络或手动配置镜像。

4. 回答提示“未配置 API Key”

填写界面上的 API Key，或设置环境变量 DEEPSEEK_API_KEY。

5. 想换其他 LLM

修改界面中的 Base URL 和 对话模型 即可，只要兼容 OpenAI 接口。

## 📄 许可
本项目采用 [MIT License](LICENSE) 开源协议

## 🙏 致谢
GraphDB：RDF 三元组存储与 SPARQL 查询

Chroma：向量数据库

Sentence-Transformers：中文嵌入模型

Gradio：交互界面

DeepSeek：默认 LLM 服务

