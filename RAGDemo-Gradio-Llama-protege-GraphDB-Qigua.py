"""
易经 GraphRAG 问答助手 - GraphDB 版（单文件，数据来自 RDF + GraphDB）
依赖：
    pip install gradio chromadb sentence-transformers SPARQLWrapper openai rdflib
运行：
    python RAGDemo-Gradio-GraphDB.py
前置：
    1. GraphDB 已启动（localhost:7200）
    2. 已创建 repository，ID 为 yijing
    3. yijing.rdf 已导入该 repository
数据源：
    yijing.rdf        （Protege 导出，已导入 GraphDB；本地仍读一次用于 Chroma 切片和起卦映射）
    yijing_tags.json  （情境标签与关键词映射，可外部编辑）
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions
from SPARQLWrapper import SPARQLWrapper, JSON

import gradio as gr
from openai import OpenAI

from yijing_rdf import load_yijing_from_rdf

# ============================================================
# 一、加载本地 RDF（仅用于 GUA_LIST、起卦映射、Chroma 切片）
# ============================================================

_BASE = os.path.dirname(os.path.abspath(__file__))
RDF_FILE = os.path.join(_BASE, "yijing.rdf")
TAGS_FILE = os.path.join(_BASE, "yijing_tags.json")

YIJING_GRAPH = load_yijing_from_rdf(RDF_FILE)

_GUA_PREFIX = "八卦_"
GUA_LIST = [g["id"] for g in YIJING_GRAPH["nodes"]["卦"]]

# 6 位二进制 -> 卦名（用于起卦）
LINES_TO_GUA = {}
for _g in YIJING_GRAPH["nodes"]["卦"]:
    _code = _g.get("卦象编码", "").strip()
    if _code:
        LINES_TO_GUA[_code] = _g["id"]

# ============================================================
# 二、GraphDB 连接与 SPARQL 查询封装
# ============================================================

GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/yijing"
)

YIJING_NS = "http://www.semanticweb.org/86189/ontologies/yijing#"

LABEL_PREFIXES = ["六十四卦_", "八卦_"]


def _strip_label_prefix(label: str) -> str:
    for p in LABEL_PREFIXES:
        if label.startswith(p):
            return label[len(p):]
    return label


def sparql_query(query: str):
    """向 GraphDB 发 SPARQL 查询，返回 bindings 列表。"""
    sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(10)
    try:
        return sparql.query().convert()["results"]["bindings"]
    except Exception as e:
        print(f"⚠️ SPARQL 查询失败：{e}\n查询语句：\n{query}")
        return []


def _val(row, key, default=""):
    v = row.get(key)
    return v["value"] if v else default


# ============================================================
# 三、图谱上下文格式化（GraphDB 版）
# ============================================================

def _query_gua_full(gua_name):
    q = f"""
    PREFIX yijing: <{YIJING_NS}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?guaci ?xiangci ?tuanci ?sequence ?symbol ?lines
           ?upperLabel ?lowerLabel WHERE {{
        ?h a yijing:Hexagram ;
           rdfs:label ?label .
        FILTER(CONTAINS(STR(?label), "{gua_name}"))
        OPTIONAL {{ ?h yijing:guaci ?guaci }}
        OPTIONAL {{ ?h yijing:xiangci ?xiangci }}
        OPTIONAL {{ ?h yijing:tuanci ?tuanci }}
        OPTIONAL {{ ?h yijing:sequence ?sequence }}
        OPTIONAL {{ ?h yijing:symbol ?symbol }}
        OPTIONAL {{ ?h yijing:lines ?lines }}
        OPTIONAL {{
            ?h yijing:hasUpperTrigram ?ut .
            ?ut rdfs:label ?upperLabel .
        }}
        OPTIONAL {{
            ?h yijing:hasLowerTrigram ?lt .
            ?lt rdfs:label ?lowerLabel .
        }}
    }}
    LIMIT 1
    """
    rows = sparql_query(q)
    return rows[0] if rows else None


def _query_yaos_of_gua(gua_name):
    q = f"""
    PREFIX yijing: <{YIJING_NS}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?yaoTitle ?yaoPosition ?yaoText ?yaoXiang ?yaoYinYang WHERE {{
        ?h a yijing:Hexagram ;
           rdfs:label ?label .
        FILTER(CONTAINS(STR(?label), "{gua_name}"))
        ?h yijing:hasYao ?yao .
        ?yao yijing:yaoTitle ?yaoTitle ;
             yijing:yaoPosition ?yaoPosition ;
             yijing:yaoText ?yaoText .
        OPTIONAL {{ ?yao yijing:yaoXiang ?yaoXiang }}
        OPTIONAL {{ ?yao yijing:yaoYinYang ?yaoYinYang }}
    }}
    ORDER BY ?yaoPosition
    """
    return sparql_query(q)


def _query_relation_targets(gua_name, relation_pred):
    q = f"""
    PREFIX yijing: <{YIJING_NS}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?targetLabel WHERE {{
        ?h a yijing:Hexagram ;
           rdfs:label ?label .
        FILTER(CONTAINS(STR(?label), "{gua_name}"))
        ?h yijing:{relation_pred} ?target .
        ?target rdfs:label ?targetLabel .
    }}
    """
    rows = sparql_query(q)
    return [_strip_label_prefix(_val(r, "targetLabel")) for r in rows
            if _val(r, "targetLabel")]


def _query_trigram(gua_name):
    q = f"""
    PREFIX yijing: <{YIJING_NS}>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?virtue ?nature ?element ?direction WHERE {{
        ?t a yijing:Trigram ;
           rdfs:label ?label .
        FILTER(CONTAINS(STR(?label), "{gua_name}"))
        OPTIONAL {{ ?t yijing:virtue ?virtue }}
        OPTIONAL {{ ?t yijing:nature ?nature }}
        OPTIONAL {{ ?t yijing:element ?element }}
        OPTIONAL {{ ?t yijing:direction ?direction }}
    }}
    LIMIT 1
    """
    rows = sparql_query(q)
    return rows[0] if rows else None


def format_graph_context(gua_name):
    parts = []

    row = _query_gua_full(gua_name)
    if row:
        label = _val(row, "label")
        short = _strip_label_prefix(label)
        guaci = _val(row, "guaci")
        xiangci = _val(row, "xiangci")
        tuanci = _val(row, "tuanci")
        sequence = _val(row, "sequence")
        symbol = _val(row, "symbol")
        lines = _val(row, "lines").strip()
        upper = _strip_label_prefix(_val(row, "upperLabel"))
        lower = _strip_label_prefix(_val(row, "lowerLabel"))

        parts.append(f"【图谱】{short}卦（第{sequence}卦）：{guaci}")
        parts.append(f"  卦象：{symbol} {lines} ｜ 上{upper} 下{lower}")
        parts.append(f"  彖辞：{tuanci}")
        parts.append(f"  象辞：{xiangci}")

        yaos = _query_yaos_of_gua(gua_name)
        if yaos:
            parts.append("  爻辞：")
            for y in yaos:
                parts.append(
                    f"    第{_val(y, 'yaoPosition')}爻 {_val(y, 'yaoTitle')}"
                    f"（{_val(y, 'yaoYinYang')}）：{_val(y, 'yaoText')}"
                    f" ｜ 小象：{_val(y, 'yaoXiang')}"
                )
    else:
        t = _query_trigram(gua_name)
        if t:
            parts.append(
                f"【图谱】{gua_name}：卦德={_val(t, 'virtue')}，"
                f"自然={_val(t, 'nature')}，五行={_val(t, 'element')}，"
                f"方位={_val(t, 'direction')}"
            )

    for rel_label, rel_pred in [("错卦", "hasCuoGua"), ("综卦", "hasZongGua")]:
        targets = [t for t in _query_relation_targets(gua_name, rel_pred)
                   if t and t != gua_name]
        if targets:
            parts.append(f"  {rel_label}：{', '.join(targets)}")

    return "\n".join(parts)


# ============================================================
# 四、从外部 JSON 加载情境标签与关键词映射
# ============================================================

def load_tags_config(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {path}，GUA_TAGS / SITUATION_KEYWORDS 将为空。")
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        gua_tags = data.get("GUA_TAGS", {})
        situation_keywords = data.get("SITUATION_KEYWORDS", {})
        print(f"✅ 已加载标签配置：GUA_TAGS {len(gua_tags)} 条，"
              f"SITUATION_KEYWORDS {len(situation_keywords)} 条")
        return gua_tags, situation_keywords
    except Exception as e:
        print(f"⚠️ 读取 {path} 失败：{e}，将使用空配置。")
        return {}, {}


GUA_TAGS, SITUATION_KEYWORDS = load_tags_config(TAGS_FILE)


# ============================================================
# 五、文本切片生成（仍从本地 RDF 构建，供 Chroma 用）
# ============================================================

def build_yijing_chunks():
    chunks = []
    ba_gua_dict = {x["id"]: x for x in YIJING_GRAPH["nodes"]["八卦"]}
    yao_by_gua = {}
    for y in YIJING_GRAPH["nodes"].get("爻", []):
        gua_sid = y.get("_gua_sid", "")
        for g in YIJING_GRAPH["nodes"]["卦"]:
            if g.get("_sid") == gua_sid:
                yao_by_gua.setdefault(g["id"], []).append(y)
                break

    for g in YIJING_GRAPH["nodes"]["卦"]:
        gid, seq = g["id"], g["序号"]
        chunks.append({"id": f"{gid}_yuan_guaci", "卦": gid, "序号": seq, "爻位": "卦辞",
                       "注家": "原文", "学派": "原典", "类型": "卦辞",
                       "text": f"{gid}卦（第{seq}卦）卦辞：{g['卦辞']}"})
        chunks.append({"id": f"{gid}_yuan_xiangci", "卦": gid, "序号": seq, "爻位": "象辞",
                       "注家": "原文", "学派": "原典", "类型": "象辞",
                       "text": f"{gid}卦象辞：{g['象辞']}"})
        chunks.append({"id": f"{gid}_yuan_tuanci", "卦": gid, "序号": seq, "爻位": "彖辞",
                       "注家": "原文", "学派": "原典", "类型": "彖辞",
                       "text": f"{gid}卦彖辞：{g['彖辞']}"})
        shang, xia = g["上下卦"]
        shang_info = ba_gua_dict.get(shang, {})
        xia_info = ba_gua_dict.get(xia, {})
        chunks.append({"id": f"{gid}_modern", "卦": gid, "序号": seq, "爻位": "卦辞",
                       "注家": "现代解读", "学派": "现代", "类型": "译文",
                       "text": (f"{gid}卦，上{shang}（{shang_info.get('自然', '')}）"
                                f"下{xia}（{xia_info.get('自然', '')}）。"
                                f"卦辞「{g['卦辞']}」意为：{g['象辞']}")})
        tags = GUA_TAGS.get(gid, [])
        if tags:
            chunks.append({
                "id": f"{gid}_tags", "卦": gid, "序号": seq, "爻位": "情境标签",
                "注家": "现代", "学派": "现代", "类型": "标签",
                "text": f"{gid}卦情境标签：{'、'.join(tags)}"
            })
        for y in yao_by_gua.get(gid, []):
            chunks.append({
                "id": f"{gid}_{y['id']}",
                "卦": gid, "序号": seq,
                "爻位": y.get("爻题", f"第{y.get('爻位')}爻"),
                "注家": "原文", "学派": "原典", "类型": "爻辞",
                "text": (f"{gid}卦 {y.get('爻题', '')}（{y.get('阴阳', '')}）爻辞："
                         f"{y.get('爻辞', '')} ｜ 小象辞：{y.get('小象辞', '')}")
            })
    return chunks


YIJING_CHUNKS = build_yijing_chunks()


# ============================================================
# 六、Chroma 文本库
# ============================================================

@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_yijing_demo")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    collection = client.get_or_create_collection(
        name="yijing_texts_64_v3",
        embedding_function=ef,
    )
    if collection.count() == 0:
        collection.add(
            ids=[c["id"] for c in YIJING_CHUNKS],
            documents=[c["text"] for c in YIJING_CHUNKS],
            metadatas=[{"卦": c["卦"], "序号": c["序号"], "爻位": c["爻位"],
                        "注家": c["注家"], "学派": c["学派"], "类型": c["类型"]}
                       for c in YIJING_CHUNKS],
        )
    return collection


def search_yijing(collection, query_text, gua_name=None, n_results=3):
    where = {"卦": gua_name} if gua_name else None
    res = collection.query(query_texts=[query_text], n_results=n_results, where=where)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "score": 1 - dist})
    return hits


# ============================================================
# 七、起卦模块
# ============================================================

def divine_by_coin():
    lines = []
    for _ in range(6):
        total = sum(random.choice([2, 3]) for _ in range(3))
        lines.append(total)
    return lines


def lines_to_hexagrams(lines):
    ben, zhi, moving = [], [], []
    for i, v in enumerate(lines):
        if v == 7:
            ben.append("1"); zhi.append("1")
        elif v == 8:
            ben.append("0"); zhi.append("0")
        elif v == 9:
            ben.append("1"); zhi.append("0"); moving.append(i + 1)
        elif v == 6:
            ben.append("0"); zhi.append("1"); moving.append(i + 1)
    return "".join(ben), "".join(zhi), moving


def divine():
    lines = divine_by_coin()
    ben_bin, zhi_bin, moving = lines_to_hexagrams(lines)
    ben_gua = LINES_TO_GUA.get(ben_bin, "未知")
    zhi_gua = LINES_TO_GUA.get(zhi_bin, ben_gua)
    return {
        "本卦": ben_gua,
        "之卦": zhi_gua,
        "动爻": moving,
        "本卦二进制": ben_bin,
        "之卦二进制": zhi_bin,
        "六爻数字": lines,
    }


# ============================================================
# 八、GraphRAG 检索
# ============================================================

def _extract_gua_from_query(query):
    hit_gua = []
    for g in sorted(GUA_LIST, key=len, reverse=True):
        if g in query and g not in hit_gua:
            hit_gua.append(g)
    if not hit_gua:
        for kw, guas in SITUATION_KEYWORDS.items():
            if kw in query:
                for g in guas:
                    if g not in hit_gua:
                        hit_gua.append(g)
    return hit_gua


def graphrag_retrieve(query, top_k=3, gua_list=None):
    collection = get_chroma_collection()

    hit_gua = gua_list if gua_list else _extract_gua_from_query(query)
    context_parts, hits_for_display = [], []

    if hit_gua:
        for g in hit_gua:
            ctx = format_graph_context(g)  # ← 走 GraphDB
            if ctx:
                context_parts.append(ctx)
        for g in hit_gua:
            for h in search_yijing(collection, query, gua_name=g, n_results=top_k):
                context_parts.append(f"【{h['meta']['注家']}·{h['meta']['类型']}】{h['text']}")
                hits_for_display.append({
                    "text": h["text"],
                    "source": f"{h['meta']['卦']}卦·{h['meta']['爻位']}·{h['meta']['注家']}",
                    "score": h["score"],
                })
    else:
        for h in search_yijing(collection, query, n_results=max(top_k * 4, 12)):
            context_parts.append(f"【{h['meta']['注家']}·{h['meta']['类型']}】{h['text']}")
            hits_for_display.append({
                "text": h["text"],
                "source": f"{h['meta']['卦']}卦·{h['meta']['爻位']}·{h['meta']['注家']}",
                "score": h["score"],
            })
        gua_summary = []
        for g in YIJING_GRAPH["nodes"]["卦"]:
            tags = "、".join(GUA_TAGS.get(g["id"], []))
            gua_summary.append(
                f"{g['id']}卦（第{g['序号']}卦）：{g['卦辞']}"
                + (f" ｜ 情境标签：{tags}" if tags else "")
            )
        context_parts.append("【64卦总览】\n" + "\n".join(gua_summary))

    return "\n\n".join(context_parts), hits_for_display


# ============================================================
# 九、提示词
# ============================================================

YIJING_PROMPT_TEMPLATE = """你是一位精通《易经》的顾问。用户会描述自己的人生处境、职业背景或困惑。
请你根据下面提供的 64 卦资料，从中选出 1-3 个与用户处境最匹配的卦，并说明理由。

匹配时请综合考虑：
- 用户当前所处的阶段（如转型、创业、合伙、稳定期）
- 用户的核心诉求（如求变、求稳、求突破）
- 卦象的象征意义（如乾为刚健进取、坤为厚德承载、屯为初生艰难、革为变革除旧等）

参考资料：
{context}

用户情况：{question}

请按以下结构回答：
1. **最匹配的卦**：卦名 + 序号
2. **匹配理由**：结合用户情况与卦辞、象辞、卦象解释
3. **行动建议**：该卦对用户的启示
4. **备选卦**：如有，列出 1-2 个并简述

请基于《易经》常识与上述资料给出合理解读，不要简单说"无法回答"。
回答："""


DIVINE_PROMPT_TEMPLATE = """你是一位精通《易经》的导师。用户提出了一个问题，
现在已经为他起卦。请你根据卦象资料，引导用户自己思考，而不是直接给答案。

【本卦】{ben_gua}
【动爻】{moving_line}
【之卦】{zhi_gua}

【本卦资料】
{ben_context}

【之卦资料】
{zhi_context}

【用户的问题】
{question}

请按以下结构回答：
1. **本卦的启示**：这个卦象在讲什么道理？和用户的处境有什么相似？
2. **动爻的关键**：动爻的爻辞和小象辞提示了什么？这是变化的契机。
3. **之卦的方向**：事情可能往哪个方向发展？
4. **三个引导性问题**：让用户自己思考，例如：
   - 这个卦象让你想到了什么？
   - 动爻提示你需要注意什么？
   - 之卦的方向是你想要的吗？

请用温和、启发式的语气，不要武断下结论，不要直接说"你应该怎么做"。
回答："""


# ============================================================
# 十、流式生成与辅助
# ============================================================

def stream_answer(client: OpenAI, model, prompt, temperature):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def to_chatbot_messages(history):
    return [{"role": m["role"], "content": m["content"]} for m in history]


def format_references(hits):
    if not hits:
        return ""
    md = "### 📎 参考片段\n\n"
    for h in hits:
        md += f"**{h['source']}** ｜ 相似度 `{h['score']:.3f}`\n\n"
        md += f"> {h['text'][:600]}{'…' if len(h['text']) > 600 else ''}\n\n---\n\n"
    return md


# ============================================================
# 十一、Gradio 回调
# ============================================================

def respond(question, chat_history,
            api_key, base_url, llm_model, temperature, top_k):
    print(f"[DEBUG] respond called, question={question!r}")
    if not question:
        yield "", chat_history, to_chatbot_messages(chat_history), ""
        return

    context, hits = graphrag_retrieve(question, top_k=top_k)

    new_history = chat_history + [{"role": "user", "content": question}]
    yield "", new_history, to_chatbot_messages(new_history), ""

    prompt = YIJING_PROMPT_TEMPLATE.format(context=context, question=question)
    assistant_msg = {"role": "assistant", "content": "", "hits": hits}
    new_history = new_history + [assistant_msg]
    refs_md = format_references(hits)
    yield "", new_history, to_chatbot_messages(new_history), refs_md

    if not api_key:
        assistant_msg["content"] = "（未配置 API Key，无法回答）"
        new_history[-1] = assistant_msg
        yield "", new_history, to_chatbot_messages(new_history), refs_md
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        for chunk in stream_answer(client, llm_model, prompt, temperature):
            assistant_msg["content"] += chunk
            new_history[-1] = assistant_msg
            yield "", new_history, to_chatbot_messages(new_history), refs_md
    except Exception as e:
        assistant_msg["content"] = f"调用模型失败：{e}"
        new_history[-1] = assistant_msg
        yield "", new_history, to_chatbot_messages(new_history), refs_md


def respond_divine(question, chat_history,
                   api_key, base_url, llm_model, temperature, top_k):
    print(f"[DEBUG] respond_divine called, question={question!r}")
    if not question:
        yield "", chat_history, to_chatbot_messages(chat_history), "", ""
        return

    result = divine()
    ben_gua = result["本卦"]
    zhi_gua = result["之卦"]
    moving = result["动爻"]
    moving_str = "、".join(f"第{i}爻" for i in moving) if moving else "无动爻"

    ben_context = format_graph_context(ben_gua)
    zhi_context = format_graph_context(zhi_gua) if zhi_gua != ben_gua else "（之卦同本卦）"

    rag_context, hits = graphrag_retrieve(question, top_k=top_k,
                                          gua_list=[ben_gua] + ([zhi_gua] if zhi_gua != ben_gua else []))
    full_ben_context = ben_context + "\n\n" + rag_context

    divine_md = (
        f"### 🎲 起卦结果\n\n"
        f"- **本卦**：{ben_gua}卦\n"
        f"- **动爻**：{moving_str}\n"
        f"- **之卦**：{zhi_gua}卦\n"
        f"- 六爻数字（初→上）：{result['六爻数字']}\n"
    )

    new_history = chat_history + [{"role": "user", "content": f"【起卦】{question}"}]
    yield "", new_history, to_chatbot_messages(new_history), divine_md, ""

    prompt = DIVINE_PROMPT_TEMPLATE.format(
        ben_gua=ben_gua, zhi_gua=zhi_gua, moving_line=moving_str,
        ben_context=full_ben_context, zhi_context=zhi_context,
        question=question,
    )
    assistant_msg = {"role": "assistant", "content": "", "hits": hits}
    new_history = new_history + [assistant_msg]
    refs_md = divine_md + "\n" + format_references(hits)
    yield "", new_history, to_chatbot_messages(new_history), divine_md, refs_md

    if not api_key:
        assistant_msg["content"] = "（未配置 API Key，无法回答）"
        new_history[-1] = assistant_msg
        yield "", new_history, to_chatbot_messages(new_history), divine_md, refs_md
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        for chunk in stream_answer(client, llm_model, prompt, temperature):
            assistant_msg["content"] += chunk
            new_history[-1] = assistant_msg
            yield "", new_history, to_chatbot_messages(new_history), divine_md, refs_md
    except Exception as e:
        assistant_msg["content"] = f"调用模型失败：{e}"
        new_history[-1] = assistant_msg
        yield "", new_history, to_chatbot_messages(new_history), divine_md, refs_md


# ============================================================
# 十二、界面
# ============================================================

with gr.Blocks(title="易经 GraphRAG 问答助手") as demo:
    gr.Markdown("# 📚 易经 GraphRAG 问答助手（GraphDB 版）")

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="DeepSeek API Key", type="password",
                value=os.getenv("DEEPSEEK_API_KEY", ""),
                info="建议用环境变量 DEEPSEEK_API_KEY",
            )
            base_url = gr.Textbox(label="Base URL", value="https://api.deepseek.com")
            llm_model = gr.Textbox(label="对话模型", value="deepseek-chat")
            temperature = gr.Slider(0.0, 1.0, 0.3, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 10, 3, step=1, label="top_k 检索条数")
            clear_btn = gr.Button("🗑️ 清空对话历史")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=480)
            with gr.Row():
                msg = gr.Textbox(
                    label="输入问题",
                    placeholder="试试问：我是1984年出生，重大研究生，三一研发5年，现在做PLM实施合伙人，想转型AI，64卦哪一卦和我匹配？",
                    scale=4,
                )
            with gr.Row():
                send_btn = gr.Button("🔍 匹配卦象", scale=1, variant="secondary")
                divine_btn = gr.Button("🎲 起卦并引导", scale=1, variant="primary")
            divine_info = gr.Markdown(label="起卦结果")   # ← 新增
            refs = gr.Markdown(label="参考片段")

    chat_history = gr.State([])

    respond_inputs = [
        msg, chat_history,
        api_key, base_url, llm_model, temperature, top_k,
    ]
    respond_outputs = [msg, chat_history, chatbot, refs]

    send_btn.click(fn=respond, inputs=respond_inputs, outputs=respond_outputs)
    msg.submit(fn=respond, inputs=respond_inputs, outputs=respond_outputs)

    divine_inputs = [
        msg, chat_history,
        api_key, base_url, llm_model, temperature, top_k,
    ]
    divine_outputs = [msg, chat_history, chatbot, divine_info, refs]   # ← 改这里
    divine_btn.click(fn=respond_divine, inputs=divine_inputs, outputs=divine_outputs)

    clear_btn.click(
        fn=lambda: ([], [], "", ""),
        inputs=None,
        outputs=[chat_history, chatbot, divine_info, refs],   # ← 加上 divine_info
    )


if __name__ == "__main__":
    n_yao = len(YIJING_GRAPH["nodes"].get("爻", []))
    print(f"✅ 本地 RDF 元数据：{len(GUA_LIST)} 卦，{n_yao} 爻")
    print(f"✅ 文本切片：{len(YIJING_CHUNKS)} 条（含爻辞）")
    print(f"✅ 可起卦：{len(LINES_TO_GUA)} 个二进制 → 卦名映射")
    print(f"✅ GraphDB endpoint：{GRAPHDB_ENDPOINT}")

    # 预热 Chroma，避免第一次提问慢
    get_chroma_collection()

    # 探测 GraphDB 是否可达
    try:
        test = sparql_query(
            f"PREFIX yijing: <{YIJING_NS}> "
            f"SELECT ?s WHERE {{ ?s a yijing:Hexagram }} LIMIT 1"
        )
        print(f"✅ GraphDB 连通性检查通过（返回 {len(test)} 条）")
    except Exception as e:
        print(f"⚠️ GraphDB 连通性检查失败：{e}")

    demo.launch()