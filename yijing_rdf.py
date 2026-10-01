"""
从 Protege 导出的 RDF/XML 解析易经数据（不依赖 Excel）
命名空间：http://www.semanticweb.org/86189/ontologies/yijing#
类：八卦 / 六十四卦 / 爻
ObjectProperty：hasUpperTrigram / hasLowerTrigram / hasCuoGua / hasZongGua / belongsToGua
DataProperty：sequence / symbol / guaci / tuanci / xiangci /
              yaoPosition / yaoText / yaoTitle / yaoXiang / yaoYinYang
"""

import os
from rdflib import Graph, RDF, RDFS, OWL, Literal

# ---------- 字段候选名 ----------
BAGUA_FIELDS = {
    "卦象": ["guaxiang", "guaXiang", "卦象", "symbol", "hasSymbol"],
    "卦德": ["guade", "guaDe", "卦德", "virtue", "hasVirtue"],
    "自然": ["ziran", "自然", "nature", "hasNature"],
    "五行": ["wuxing", "五行", "element", "hasElement"],
    "方位": ["fangwei", "方位", "direction", "hasDirection"],
}

GUA_FIELDS = {
    "序号": ["xuhao", "序号", "order", "sequence", "hasOrder"],
    "卦象": ["guaxiang", "guaXiang", "卦象", "symbol", "hasSymbol"],
    "卦辞": ["guaci", "guaCi", "卦辞", "judgment"],
    "彖辞": ["tuanci", "tuanCi", "彖辞", "tuan"],
    "象辞": ["xiangci", "xiangCi", "象辞", "image"],
    "卦象编码": ["lines", "卦象编码", "linesCode"],
}

YAO_FIELDS = {
    "爻位": ["yaoPosition", "yaowei", "爻位", "position"],
    "爻题": ["yaoTitle", "yaoti", "爻题", "title"],
    "爻辞": ["yaoText", "yaoci", "爻辞", "text"],
    "小象辞": ["yaoXiang", "yaoxiang", "小象辞", "xiang"],
    "阴阳": ["yaoYinYang", "yinyang", "阴阳", "yinyang"],
}

OBJ_REFS = {
    "上卦": ["hasUpperTrigram", "shanggua", "上卦"],
    "下卦": ["hasLowerTrigram", "xiagua", "下卦"],
    "错卦": ["hasCuoGua", "cuogua", "错卦"],
    "综卦": ["hasZongGua", "zonggua", "综卦"],
}

_GUA_PREFIX = "八卦_"
_HEX_PREFIX = "六十四卦_"


def _local(uri) -> str:
    s = str(uri)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def _strip_self(s: str) -> str:
    s = str(s).strip()
    for m in ("（自身）", "(自身)", "（自）", "(自)"):
        s = s.replace(m, "")
    return s.strip()


def _get_literal(g, subj, candidates, default=""):
    for p in g.predicates(subject=subj):
        if _local(p) in candidates:
            for o in g.objects(subject=subj, predicate=p):
                if isinstance(o, Literal):
                    return str(o).strip()
    return default


def _get_int(g, subj, candidates, default=0):
    v = _get_literal(g, subj, candidates, None)
    if v is None:
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _get_obj_ref(g, subj, candidates):
    for p in g.predicates(subject=subj):
        if _local(p) in candidates:
            for o in g.objects(subject=subj, predicate=p):
                return _local(o)
    return ""


def _get_label_zh(g, subj) -> str:
    """取 rdfs:label，优先 zh 语言标签"""
    labels = []
    for o in g.objects(subject=subj, predicate=RDFS.label):
        if isinstance(o, Literal):
            labels.append((o.language or "", str(o).strip()))
    for lang, txt in labels:
        if lang.startswith("zh"):
            return txt
    return labels[0][1] if labels else ""


def _strip_class_prefix(name: str) -> str:
    """剥掉 '六十四卦_' / '八卦_' 前缀"""
    for pre in (_HEX_PREFIX, _GUA_PREFIX):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _find_class(g, names):
    if isinstance(names, str):
        names = [names]
    for s in g.subjects(RDF.type, OWL.Class):
        if _local(s) in names:
            return s
    for s, p, o in g.triples((None, RDF.type, None)):
        if _local(o) in names:
            return o
    return None


def load_yijing_from_rdf(rdf_path: str) -> dict:
    if not os.path.exists(rdf_path):
        raise FileNotFoundError(f"未找到 RDF：{rdf_path}")

    g = Graph()
    g.parse(rdf_path, format="xml")
    print(f"📄 RDF: {len(g)} triples")

    trigram_class = _find_class(g, ["八卦", "Trigram", "Bagua"])
    hexagram_class = _find_class(g, ["六十四卦", "Hexagram", "Gua"])
    yao_class = _find_class(g, ["爻", "Yao"])
    print(f"  八卦类: {trigram_class}")
    print(f"  六十四卦类: {hexagram_class}")
    print(f"  爻类: {yao_class}")

    if not trigram_class or not hexagram_class:
        raise ValueError("找不到『八卦』或『六十四卦』类")

    # ---------- 八卦 ----------
    bagua_id2name = {}
    bagua_nodes = []
    for s in g.subjects(RDF.type, trigram_class):
        sid = _local(s)
        name = _strip_class_prefix(_get_label_zh(g, s)) or sid
        bagua_id2name[sid] = name

        node = {"id": name, "type": "八卦"}
        for field, cands in BAGUA_FIELDS.items():
            node[field] = _get_literal(g, s, cands)
        bagua_nodes.append(node)

    # ---------- 六十四卦 ----------
    gua_id2name = {}
    gua_nodes = []
    raw_refs = {}

    for s in g.subjects(RDF.type, hexagram_class):
        sid = _local(s)
        name = _strip_class_prefix(_get_label_zh(g, s)) or sid
        gua_id2name[sid] = name

        node = {
            "id": name,
            "序号": _get_int(g, s, GUA_FIELDS["序号"]),
            "卦象": _get_literal(g, s, GUA_FIELDS["卦象"]),
            "卦辞": _get_literal(g, s, GUA_FIELDS["卦辞"]),
            "彖辞": _get_literal(g, s, GUA_FIELDS["彖辞"]),
            "象辞": _get_literal(g, s, GUA_FIELDS["象辞"]),
            "卦象编码": _get_literal(g, s, GUA_FIELDS["卦象编码"]).strip(),
            "_sid": sid,
        }
        raw_refs[sid] = {k: _get_obj_ref(g, s, v) for k, v in OBJ_REFS.items()}
        gua_nodes.append(node)

    # ---------- 映射 ref → 中文名 ----------
    for node in gua_nodes:
        r = raw_refs.get(node["_sid"], {})
        shang = bagua_id2name.get(r.get("上卦"), r.get("上卦", ""))
        xia = bagua_id2name.get(r.get("下卦"), r.get("下卦", ""))
        node["上下卦"] = [shang, xia]
        node["_cuo_sid"] = r.get("错卦", "")
        node["_zong_sid"] = r.get("综卦", "")

    gua_nodes.sort(key=lambda x: x["序号"] if x["序号"] else 999)

    # ---------- 爻 ----------
    yao_nodes = []
    if yao_class:
        for s in g.subjects(RDF.type, yao_class):
            sid = _local(s)
            node = {
                "id": sid,
                "爻位": _get_int(g, s, YAO_FIELDS["爻位"]),
                "爻题": _get_literal(g, s, YAO_FIELDS["爻题"]),
                "爻辞": _get_literal(g, s, YAO_FIELDS["爻辞"]),
                "小象辞": _get_literal(g, s, YAO_FIELDS["小象辞"]),
                "阴阳": _get_literal(g, s, YAO_FIELDS["阴阳"]),
                "_gua_sid": _get_obj_ref(g, s, ["belongsToGua", "所属卦"]),
            }
            yao_nodes.append(node)

    # ---------- 构建边 ----------
    edges = []
    for node in gua_nodes:
        gid = node["id"]
        shang, xia = node["上下卦"]
        if shang:
            edges.append({"source": gid, "target": _GUA_PREFIX + shang, "relation": "上卦"})
        if xia:
            edges.append({"source": gid, "target": _GUA_PREFIX + xia, "relation": "下卦"})

        cuo_sid = node.get("_cuo_sid", "")
        if cuo_sid:
            cuo = _strip_self(gua_id2name.get(cuo_sid, cuo_sid))
            if cuo:
                edges.append({"source": gid, "target": cuo, "relation": "错卦"})

        zong_sid = node.get("_zong_sid", "")
        if zong_sid:
            zong = _strip_self(gua_id2name.get(zong_sid, zong_sid))
            if zong:
                edges.append({"source": gid, "target": zong, "relation": "综卦"})

    # 爻 → 卦 边
    for yao in yao_nodes:
        gua_name = gua_id2name.get(yao.get("_gua_sid", ""), "")
        if gua_name:
            edges.append({
                "source": yao["id"],
                "target": gua_name,
                "relation": "belongsToGua",
            })

    print(f"  → 八卦 {len(bagua_nodes)} 个，卦 {len(gua_nodes)} 个，"
          f"爻 {len(yao_nodes)} 个，边 {len(edges)} 条")

    sample = gua_nodes[0] if gua_nodes else {}
    print(f"  示例卦：{sample.get('id')} | 序号={sample.get('序号')} | "
          f"上下卦={sample.get('上下卦')} | 卦辞={sample.get('卦辞', '')[:20]}...")
    if yao_nodes:
        ys = yao_nodes[0]
        print(f"  示例爻：{ys.get('id')} | 爻题={ys.get('爻题')} | "
              f"爻位={ys.get('爻位')} | 爻辞={ys.get('爻辞', '')[:20]}...")

    return {
        "nodes": {
            "八卦": bagua_nodes,
            "卦": gua_nodes,
            "爻": yao_nodes,
        },
        "edges": edges,
    }


if __name__ == "__main__":
    data = load_yijing_from_rdf("yijing.rdf")
    print("\n=== 示例八卦 ===")
    print(data["nodes"]["八卦"][0])
    print("\n=== 示例卦 ===")
    print({k: v for k, v in data["nodes"]["卦"][0].items() if not k.startswith("_")})
    print("\n=== 示例爻 ===")
    if data["nodes"]["爻"]:
        print({k: v for k, v in data["nodes"]["爻"][0].items() if not k.startswith("_")})
    print("\n=== 前 6 条边 ===")
    for e in data["edges"][:6]:
        print(" ", e)