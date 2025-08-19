# -*- coding: utf-8 -*-
import os
import json
import time
import re
import requests
import pandas as pd
import streamlit as st
from io import StringIO
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 可选更强向量检索（自动降级） =====
_USE_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _USE_ST = True
except Exception:
    _USE_ST = False

# ===== UI 样式 =====
st.set_page_config(page_title="LLM 学术写作流水线（DeepSeek + RAG）", page_icon="🧭", layout="wide")
st.markdown("""
<style>
h1, h2, h3 { font-family: "Segoe UI", "PingFang SC", "Helvetica Neue", Arial; }
.block-container { padding-top: 1.2rem; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
div[data-testid="stSidebar"] * { color: #f3f4f6; }
.kpi { background: #0ea5e910; border: 1px solid #0ea5e933; padding: 12px 14px; border-radius: 16px; }
.codebox { background: #111827; color: #e5e7eb; padding: 12px 14px; border-radius: 12px; font-size: 13px; border: 1px solid #374151; }
hr{ border: none; border-top: 1px dashed #d1d5db44; margin: 0.8rem 0;}
.tag { display:inline-block; padding:2px 8px; border-radius:999px; background:#111827; color:#a7f3d0; border:1px solid #065f46; margin-right:6px; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ===== 工具函数 =====
def truncate_text(s: str, max_chars: int = 12000) -> str:
    if s is None: return ""
    return s if len(s) <= max_chars else s[:max_chars] + "\n...[TRUNCATED]..."

def extract_json_block(text: str) -> Optional[dict]:
    """尽力从模型输出中提取 JSON"""
    if not text: return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}$', text.strip(), re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def looks_like_meta(text: str) -> bool:
    """判断是否像元描述而非正文"""
    if not text: return True
    bad_keywords = ["将会", "将要", "会通过", "will ", "revised draft will", "we will", "we plan to"]
    return any(k.lower() in text.lower() for k in bad_keywords) or len(text) < 300

# ===== LLM 客户端 =====
class LLMClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat", timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: Optional[int] = None, response_format: Optional[dict] = None) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if response_format is not None: payload["response_format"] = response_format
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API错误[{resp.status_code}]: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"无法解析LLM返回：{data}")

    def chat_json(self, messages: List[Dict[str,str]], temperature: float = 0.2,
                  max_tokens: Optional[int]=None, required_keys: Optional[List[str]]=None, retries:int=2) -> dict:
        """优先 JSON 模式；失败则二次提示只返回 JSON；最后回退正则抽取"""
        # 尝试 1：JSON 模式
        out = self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                        response_format={"type":"json_object"})
        j = extract_json_block(out)
        if j and (not required_keys or all(k in j for k in required_keys)):
            return j
        # 尝试 2：加一条系统提示，强制最小 JSON
        forced = [{"role":"system","content":"请只返回严格合法的最小化 JSON，不得包含任何额外文字或解释。"},
                  *messages]
        out2 = self.chat(forced, temperature=temperature, max_tokens=max_tokens,
                         response_format={"type":"json_object"})
        j2 = extract_json_block(out2)
        if j2 and (not required_keys or all(k in j2 for k in required_keys)):
            return j2
        # 尝试 3：普通模式 + 正则抽取
        out3 = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        j3 = extract_json_block(out3) or {}
        return j3

# ===== RAG 检索 =====
class RAGRetriever:
    def __init__(self, df: pd.DataFrame, use_sentence_transformer: bool = _USE_ST, st_model: str = "all-MiniLM-L6-v2"):
        assert "title" in df.columns and "text" in df.columns, "xlsx 需要包含 'title' 和 'text' 列"
        self.df = df.reset_index(drop=True).copy()
        self.use_st = use_sentence_transformer
        self._vectorizer = None
        self._tfidf_matrix = None
        self._embedder = None
        self._embeddings = None
        self.st_model_name = st_model

    @st.cache_data(show_spinner=False)
    def _build_tfidf(_self, corpus: List[str]):
        vec = TfidfVectorizer(stop_words="english", max_features=200000)
        mat = vec.fit_transform(corpus)
        return vec, mat

    def _ensure_index(self):
        corpus = (self.df["title"].astype(str) + " " + self.df["text"].astype(str)).tolist()
        if self.use_st:
            if self._embedder is None:
                self._embedder = SentenceTransformer(self.st_model_name)
                self._embeddings = self._embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
        else:
            if self._vectorizer is None or self._tfidf_matrix is None:
                self._vectorizer, self._tfidf_matrix = self._build_tfidf(corpus)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        self._ensure_index()
        if self.use_st:
            qv = self._embedder.encode([query], convert_to_numpy=True)
            sim = (self._embeddings @ qv.T).reshape(-1)
        else:
            qv = self._vectorizer.transform([query])
            sim = cosine_similarity(qv, self._tfidf_matrix).flatten()
        idx = sim.argsort()[-top_k:][::-1]
        out = self.df.iloc[idx].copy()
        out["score"] = sim[idx]
        out["doc_id"] = out.index.astype(str)
        return out[["doc_id","title","text","score"]]

# ===== 产物存储 =====
@dataclass
class PipelineArtifacts:
    plan_json: Dict[str, Any] = field(default_factory=dict)
    retrieved: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["doc_id","title","text","score"]))
    outline_json: Dict[str, Any] = field(default_factory=dict)
    title: str = ""
    drafts: Dict[str, str] = field(default_factory=dict)
    verified: Dict[str, Any] = field(default_factory=dict)
    refined: str = ""
    evaluation: Dict[str, Any] = field(default_factory=dict)
    abstract: str = ""
    full_paper_md: str = ""

# ===== Prompt 模板（内置“信源优先级”） =====
def source_of_truth_block(user_need: str) -> str:
    return f"""
【信源优先级】：
1) 用户需求文本（最高优先级）：{user_need}
2) RAG/HyDE 文档（仅作辅助；如与需求冲突，必须以需求为准）
3) 证据不足时：输出 NEED MORE EVIDENCE: <建议检索查询>，而不是臆测
"""

def prompt_plan(keywords: str, requirements: str, hyde_summary: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
你是一位学术写作规划师。已知：
关键词：{keywords}
需求约束：{requirements}
HyDE摘要：{hyde_summary}

请输出 3–7 步写作计划（严格 JSON）：
{{"plan":[{{"step":1,"objective":"","artifact":""}}], "open_questions":[]}}
"""
    return [{"role":"user","content":user_prompt}]

def prompt_outline(venue_style: str, research_need: str, evidence_pack: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
你是一位学术写作专家（{venue_style} 风格）。任务：
1) 生成一条论文标题
2) 生成 IMRaD 大纲（严格 JSON）：
{{
  "title": "",
  "outline": {{
    "Introduction": [],
    "Methods": [],
    "Results": [],
    "Discussion": []
  }}
}}
要求：
- 优先满足用户需求；RAG/HyDE 仅作辅助
- 每项要点 10 字以内关键词；不得添加额外字段
参考证据（可选）： 
{evidence_pack}
"""
    return [{"role":"user","content":user_prompt}]

def prompt_title_only(research_need: str, user_need: str) -> List[Dict[str,str]]:
    return [{"role":"user","content":f"""{source_of_truth_block(user_need)}
请基于用户需求生成 1 个顶刊风格标题。只输出标题文本，不要前后缀或解释。
需求：{research_need}
"""}]

def prompt_expand(section_name: str, points: List[str], evidence_pack: str, venue_style: str, user_need: str) -> List[Dict[str,str]]:
    pts = "\n- " + "\n- ".join(points) if points else ""
    user_prompt = f"""{source_of_truth_block(user_need)}
请扩写《{section_name}》部分，参考要点：{pts}

证据（仅可引用其中内容；不得杜撰）：
{evidence_pack}

写作要求：
- 风格：{venue_style} 顶刊风格（严谨、客观、可检验）
- 结构：3—5段，逻辑清楚，不跑题
- 如证据不足：输出 NEED MORE EVIDENCE: <查询>
"""
    return [{"role":"user","content":user_prompt}]

def prompt_verify(draft: str, evidence_pack: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
对以下草稿执行 Chain-of-Verification（严格 JSON）：
草稿：
{draft}

证据（仅此为准）：
{evidence_pack}

输出：
{{
  "verification_questions": [],
  "qa_log": [{{"q":"","a":""}}],
  "revised_draft": ""
}}
规则：
- 仅对证据不足之处做最小修改
- 引用使用 [doc_id]
"""
    return [{"role":"user","content":user_prompt}]

def prompt_refine(draft: str, venue_style: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
角色A（审稿人）：给出 strengths / weaknesses / must_fixes（清晰度、连贯性、论证严谨性、引用合规、{venue_style} 风格）
角色B（作者）：先落实 must_fixes，再处理 weaknesses，保持引用不变。
最终只返回严格 JSON，且 "revised" 必须是**修订后的完整正文**，不得写计划或意图。

{{
  "review": {{"strengths":[],"weaknesses":[],"must_fixes":[]}},
  "revised": ""
}}
草稿：
{draft}
"""
    return [{"role":"user","content":user_prompt}]

def prompt_force_rewrite(draft: str, user_need: str) -> List[Dict[str,str]]:
    return [{"role":"user","content":f"""{source_of_truth_block(user_need)}
请直接输出修订后的完整正文（覆盖原文所有部分），不要写计划或意图，不要使用将来时描述。
保持引用标注不变。
原稿：
{draft}
"""}]

def prompt_evaluate(draft: str, venue_style: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
请按 0-5 评分并简述理由（严格 JSON）：
{{"clarity":0,"rigor":0,"style":0,"structure_citation":0,"comments":""}}
维度：清晰度 clarity；严谨性 rigor；{venue_style} 风格 style；结构与引用完整性 structure_citation
文本：
{draft}
"""
    return [{"role":"user","content":user_prompt}]

def prompt_abstract(full_text: str, user_need: str) -> List[Dict[str,str]]:
    user_prompt = f"""{source_of_truth_block(user_need)}
基于论文正文（可择要），使用 Chain-of-Density 打磨摘要。
要求：**以用户需求为唯一优先信源**，RAG/HyDE 仅作辅助；不得引入不存在于需求/正文的主张。
输出三部分：
1) 简洁摘要（~130字，实体稀疏）
2) 缺失的重要实体列表
3) 在不增加长度情况下重写的摘要（补齐实体）
正文：
{full_text}
"""
    return [{"role":"user","content":user_prompt}]

# ===== 侧边栏配置 =====
with st.sidebar:
    st.title("⚙️ 配置")
    api_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY",""), type="password")
    base_url = st.text_input("Base URL", value=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com"))
    model_name = st.text_input("Model 名称", value="deepseek-chat")
    venue_style = st.selectbox("目标期刊风格", ["MIS Quarterly (MISQ)", "Information Systems Research (ISR)", "Management Science", "POMS"], index=0)
    top_k = st.slider("RAG Top-K", min_value=5, max_value=20, value=10, step=1)
    use_st = st.toggle("使用句向量（更准但更慢）", value=_USE_ST)
    st.caption("※ 未勾选时使用 TF-IDF + 余弦相似度")

# ===== 主页面输入 =====
st.title("🧭 最小化 LLM 学术写作流水线（关键词 + HyDE + RAG）")
st.markdown('<div class="kpi">输入需求 → 规划 → RAG → 标题+大纲 → 扩写 → 校验 → 精修 → 评估 → 摘要</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    user_keywords = st.text_area("🔎 关键词 / 一句话需求（最高优先级信源）", height=90, placeholder="例如：团队购买对非折扣商品的溢出效应；或一段研究需求描述")
with col2:
    hyde_summary = st.text_area("📄 HyDE 摘要（可选）", height=90, placeholder="可为空；用于辅助检索/写作")
requirements = st.text_input("📌 其他硬性约束（字数、期刊、领域、读者对象等）", "")

def USER_NEED():
    return (user_keywords or "").strip() + ("；" + requirements if requirements else "")

# 上传知识库
st.subheader("📚 上传知识库（.xlsx，含 title / text）")
kb_file = st.file_uploader("选择 Excel 文件", type=["xlsx"], accept_multiple_files=False)

def read_kb(file) -> Optional[pd.DataFrame]:
    if file is None: return None
    try:
        df = pd.read_excel(file)
        assert "title" in df.columns and "text" in df.columns
        return df
    except Exception as e:
        st.error(f"读取 Excel 失败：{e}")
        return None

df_kb = read_kb(kb_file)

# 初始化状态
if "artifacts" not in st.session_state:
    st.session_state.artifacts = PipelineArtifacts()

_client = LLMClient(api_key, base_url, model_name) if (api_key and base_url and model_name) else None
_retriever = RAGRetriever(df_kb, use_sentence_transformer=use_st) if df_kb is not None else None

def build_evidence_pack(df_sel: pd.DataFrame, max_chars: int = 12000) -> str:
    chunks = []
    for _, r in df_sel.iterrows():
        block = f"[{r['doc_id']}] {r['title']}\n{truncate_text(str(r['text']), 1500)}"
        chunks.append(block)
    combined = "\n\n".join(chunks)
    return truncate_text(combined, max_chars=max_chars)

# ===== 步骤函数 =====
def step_plan():
    msgs = prompt_plan(user_keywords, requirements, hyde_summary, USER_NEED())
    plan = _client.chat_json(msgs, temperature=0.2, required_keys=["plan","open_questions"])
    st.session_state.artifacts.plan_json = plan
    return plan

def step_rag():
    q = USER_NEED() or hyde_summary
    docs = _retriever.search(q, top_k=top_k)
    st.session_state.artifacts.retrieved = docs
    return docs

def step_outline():
    ev = build_evidence_pack(st.session_state.artifacts.retrieved)
    msgs = prompt_outline(venue_style, USER_NEED() or requirements, ev, USER_NEED())
    j = _client.chat_json(msgs, temperature=0.1, required_keys=["title","outline"])
    st.session_state.artifacts.outline_json = j or {}
    # 标题兜底
    title = (j or {}).get("title","").strip()
    if not title:
        title = step_title()  # 独立生成
    st.session_state.artifacts.title = title
    return j, ev

def step_title() -> str:
    msgs = prompt_title_only(USER_NEED() or requirements, USER_NEED())
    title = _client.chat([{"role":"system","content":"只输出标题文本。"},
                          *msgs], temperature=0.2, max_tokens=80).strip().strip('"').strip()
    return title

def step_expand():
    outline = st.session_state.artifacts.outline_json or {}
    ev = build_evidence_pack(st.session_state.artifacts.retrieved)
    drafts = {}
    if "outline" not in outline:
        raise RuntimeError("尚未生成大纲或大纲格式异常。")
    for sec in ["Introduction","Methods","Results","Discussion"]:
        pts = outline["outline"].get(sec, [])
        msgs = prompt_expand(sec, pts, ev, venue_style, USER_NEED())
        text = _client.chat(msgs, temperature=0.15, max_tokens=1600)
        drafts[sec] = text
    st.session_state.artifacts.drafts = drafts
    return drafts, ev

def step_verify():
    drafts = st.session_state.artifacts.drafts
    merged = ""
    for sec in ["Introduction","Methods","Results","Discussion"]:
        if sec in drafts: merged += f"# {sec}\n{drafts[sec]}\n\n"
    ev = build_evidence_pack(st.session_state.artifacts.retrieved)
    msgs = prompt_verify(merged, ev, USER_NEED())
    ver = _client.chat_json(msgs, temperature=0.1, required_keys=["verification_questions","qa_log","revised_draft"])
    st.session_state.artifacts.verified = ver
    return ver

def step_refine():
    verified = st.session_state.artifacts.verified or {}
    revised = verified.get("revised_draft","").strip()
    base_text = revised if revised else "\n".join([f"# {k}\n{v}" for k,v in st.session_state.artifacts.drafts.items()])
    # 先走“评审+修订”的 JSON 流
    msgs = prompt_refine(base_text, venue_style, USER_NEED())
    j = _client.chat_json(msgs, temperature=0.1, required_keys=["review","revised"])
    refined = j.get("revised","").strip()
    # 如果仍是元描述或太短，则强制改写一次
    if looks_like_meta(refined):
        forced = _client.chat(prompt_force_rewrite(base_text, USER_NEED()), temperature=0.1, max_tokens=3500)
        refined = forced.strip()
    st.session_state.artifacts.refined = refined
    return refined, j

def step_evaluate():
    draft = st.session_state.artifacts.refined or "\n".join([f"# {k}\n{v}" for k,v in st.session_state.artifacts.drafts.items()])
    msgs = prompt_evaluate(draft, venue_style, USER_NEED())
    eval_json = _client.chat_json(msgs, temperature=0.0, required_keys=["clarity","rigor","style","structure_citation","comments"])
    st.session_state.artifacts.evaluation = eval_json
    return eval_json

def step_abstract():
    title = st.session_state.artifacts.title or "（待定标题）"
    draft = st.session_state.artifacts.refined or "\n".join([f"# {k}\n{v}" for k,v in st.session_state.artifacts.drafts.items()])
    msgs = prompt_abstract(truncate_text(draft, 9000), USER_NEED())
    abs_out = _client.chat(msgs, temperature=0.2, max_tokens=800)
    st.session_state.artifacts.abstract = abs_out
    md = f"# {title}\n\n{draft}\n\n---\n## Abstract\n\n{abs_out}\n"
    st.session_state.artifacts.full_paper_md = md
    return abs_out, md

# ===== 按钮区 =====
run_cols = st.columns([1,1,1,1,1,1,1,1,2])
btn_plan    = run_cols[0].button("🧩 规划")
btn_rag     = run_cols[1].button("📖 检索")
btn_outline = run_cols[2].button("🧱 大纲/标题")
btn_expand  = run_cols[3].button("✍️ 扩写")
btn_verify  = run_cols[4].button("🔬 校验")
btn_refine  = run_cols[5].button("🛠️ 精修")
btn_eval    = run_cols[6].button("📊 评估")
btn_abs     = run_cols[7].button("✨ 摘要")
btn_all     = run_cols[8].button("🚀 一键全流程")

if not _client:
    st.warning("请在左侧输入 API Key / Base URL / 模型名。")
if _retriever is None:
    st.info("请上传知识库 Excel。")

def run_all():
    if not _client or _retriever is None:
        st.error("请先配置 API 和上传知识库。")
        return
    with st.spinner("全流程执行中..."):
        step_plan()
        step_rag()
        step_outline()
        step_expand()
        step_verify()
        step_refine()
        step_evaluate()
        step_abstract()
    st.success("全流程完成！")

if btn_all: run_all()
if btn_plan and _client: st.session_state.artifacts.plan_json = step_plan()
if btn_rag and _retriever is not None: st.session_state.artifacts.retrieved = step_rag()
if btn_outline and _client: st.session_state.artifacts.outline_json, _ = step_outline()
if btn_expand and _client: st.session_state.artifacts.drafts, _ = step_expand()
if btn_verify and _client: st.session_state.artifacts.verified = step_verify()
if btn_refine and _client:
    refined, _ = step_refine()
if btn_eval and _client: st.session_state.artifacts.evaluation = step_evaluate()
if btn_abs and _client: st.session_state.artifacts.abstract, st.session_state.artifacts.full_paper_md = step_abstract()

# ===== 可视化输出（可折叠） =====
st.subheader("🔭 过程产物（点击展开/收起）")

with st.expander("🧩 步骤 2 — 规划（Plan）", expanded=False):
    st.json(st.session_state.artifacts.plan_json or {"tips":"尚未生成"})

with st.expander("📖 步骤 3 — RAG 检索结果", expanded=False):
    if not st.session_state.artifacts.retrieved.empty:
        st.dataframe(st.session_state.artifacts.retrieved, use_container_width=True)
        st.caption("↑ 已按相似度排序；doc_id 可用于引用。")
    else:
        st.info("尚未检索或无结果")

with st.expander("🧱 步骤 4 — 标题与大纲（IMRaD）", expanded=False):
    j = st.session_state.artifacts.outline_json
    title = st.session_state.artifacts.title
    if title: st.write("**标题**：", title)
    if j:
        st.json(j.get("outline", {}))
        st.download_button("下载大纲 JSON", data=json.dumps(j, ensure_ascii=False, indent=2),
                           file_name="outline.json", mime="application/json")
    else:
        st.info("尚未生成")

with st.expander("✍️ 步骤 5 — 各部分扩写草稿", expanded=False):
    drafts = st.session_state.artifacts.drafts
    if drafts:
        for sec in ["Introduction","Methods","Results","Discussion"]:
            if sec in drafts:
                st.markdown(f"#### {sec}")
                st.markdown(f"<div class='codebox'>{drafts[sec]}</div>", unsafe_allow_html=True)
        st.download_button("下载草稿（MD）",
                           data="\n\n".join([f"# {k}\n{v}" for k,v in drafts.items()]),
                           file_name="draft_sections.md", mime="text/markdown")
    else:
        st.info("尚未扩写")

with st.expander("🔬 步骤 6 — 校验（CoVe）", expanded=False):
    ver = st.session_state.artifacts.verified
    if ver: st.json(ver)
    else: st.info("尚未校验")

with st.expander("🛠️ 步骤 7 — 精修（Self-Refine）", expanded=False):
    if st.session_state.artifacts.refined:
        st.markdown(f"<div class='codebox'>{st.session_state.artifacts.refined}</div>", unsafe_allow_html=True)
        st.download_button("下载修订稿（MD）",
                           data=st.session_state.artifacts.refined,
                           file_name="revised.md", mime="text/markdown")
    else:
        st.info("尚未精修")

with st.expander("📊 步骤 8 — 评估（LLM-as-Judge）", expanded=False):
    if st.session_state.artifacts.evaluation:
        st.json(st.session_state.artifacts.evaluation)
    else:
        st.info("尚未评估")

with st.expander("✨ 步骤 9 — 摘要打磨（CoD） & 全文合成", expanded=True):
    if st.session_state.artifacts.abstract:
        st.markdown("**摘要结果（含三步）**")
        st.markdown(f"<div class='codebox'>{st.session_state.artifacts.abstract}</div>", unsafe_allow_html=True)
        if st.session_state.artifacts.full_paper_md:
            st.download_button("下载全文（Markdown）",
                               data=st.session_state.artifacts.full_paper_md,
                               file_name="paper_full.md", mime="text/markdown")
    else:
        st.info("尚未生成摘要")

st.markdown("---")
st.caption("© 2025 LLM Pipeline · DeepSeek API · 信源优先级：需求 > RAG/HyDE · 顶刊风格（MISQ/ISR/MS/POMS）")
