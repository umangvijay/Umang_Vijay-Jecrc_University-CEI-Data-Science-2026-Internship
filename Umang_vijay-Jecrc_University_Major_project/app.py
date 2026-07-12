"""
Autonomous Data Science Co-Pilot — Streamlit Application (Fluid Analyst UI)
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import os, time, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Fluid Analyst — Data Science Co-Pilot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ── Load CSS ──
css_path = Path(__file__).parent / "static" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

import streamlit.components.v1 as components

# ── WebGL Shader Background ──
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;
    if (parentDoc.getElementById('fluid-shader-bg')) return;
    
    const canvas = parentDoc.createElement('canvas');
    canvas.id = 'fluid-shader-bg';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.zIndex = '-1';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '0.6';
    parentDoc.body.insertBefore(canvas, parentDoc.body.firstChild);
    
    if (!parentDoc.getElementById('fluid-deco-1')) {
        const deco1 = parentDoc.createElement('div');
        deco1.id = 'fluid-deco-1';
        deco1.style.position = 'fixed';
        deco1.style.right = '-50px';
        deco1.style.top = '10%';
        deco1.style.opacity = '0.4';
        deco1.style.pointerEvents = 'none';
        deco1.style.zIndex = '-5';
        deco1.style.mixBlendMode = 'screen';
        deco1.style.animation = 'float 6s ease-in-out infinite';
        
        const img1 = parentDoc.createElement('img');
        img1.src = 'https://lh3.googleusercontent.com/aida/AP1WRLvRWdUcBCQMv_yL7Emv-IEjvAbaQh_xU1LVMt3EwPOXoJ-PWYCN6YlPb4L7ICrEgWFLdqq6ugFLgicdmPupQcEDUNKxL6x6ze8GFTm2zvdmBgwLRuXdbYnwuP4-PjvT7ta4rz7w6PCB7eawdyG5VADvTfkB886uVzMU6N5iPEgC3YMgElrdlGPzNezJUBaUu06wxKBDEVkK-RxXmF5VawoiDvV4otDKSv3Ua8jF2VhiTz9Of6QDx8YEG2y4';
        img1.style.width = '500px';
        img1.style.height = 'auto';
        img1.style.borderRadius = '50%';
        img1.style.filter = 'blur(2px)';
        deco1.appendChild(img1);
        parentDoc.body.appendChild(deco1);
    }

    if (!parentDoc.getElementById('fluid-deco-2')) {
        const deco2 = parentDoc.createElement('div');
        deco2.id = 'fluid-deco-2';
        deco2.style.position = 'fixed';
        deco2.style.left = '-50px';
        deco2.style.bottom = '10%';
        deco2.style.opacity = '0.3';
        deco2.style.pointerEvents = 'none';
        deco2.style.zIndex = '-5';
        deco2.style.mixBlendMode = 'screen';
        deco2.style.animation = 'float 6s ease-in-out infinite';
        deco2.style.animationDelay = '2s';
        
        const img2 = parentDoc.createElement('img');
        img2.src = 'https://lh3.googleusercontent.com/aida/AP1WRLsmWch2msYedCsDvxc0q05lZDphpzztwfS-tHqdNJs6xZMaSUN61E5qtbc4QRzGGZF4rPRKCSo37Q3M3_6cUpmEuWUuDVZ-gS7LlE9lrZUSXsllfg7ajSpQg9nV5iTaNcw37nYzjhrooihA5IZNEYDkahl51kCbtHZGlT5G0OdI2JUpO6zzAIrOfyualhru0xhBnuCNqXZer24-BCOAD8WHA6nRmCcjuXOK8WTAREroEkVLZUQmaTebj36i';
        img2.style.width = '400px';
        img2.style.height = 'auto';
        img2.style.filter = 'blur(1px)';
        deco2.appendChild(img2);
        parentDoc.body.appendChild(deco2);
    }
    
    function syncSize() {
        const w = parentDoc.documentElement.clientWidth;
        const h = parentDoc.documentElement.clientHeight;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }
    window.parent.addEventListener('resize', syncSize);
    syncSize();
    
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) return;
    const vs = `attribute vec2 a_position; varying vec2 v_texCoord; void main() { v_texCoord = a_position * 0.5 + 0.5; gl_Position = vec4(a_position, 0.0, 1.0); }`;
    const fs = `precision highp float; uniform float u_time; uniform vec2 u_resolution; void main() { vec2 uv = gl_FragCoord.xy / u_resolution.xy; float color = 0.0; vec2 p = uv * 2.0 - 1.0; p.x *= u_resolution.x / u_resolution.y; float t = u_time * 0.2; for(float i = 1.0; i < 4.0; i++){ p.x += 0.3 / i * sin(i * 3.0 * p.y + t + i * 0.5); p.y += 0.3 / i * cos(i * 3.0 * p.x + t + i * 0.5); } vec3 baseColor = vec3(0.02, 0.01, 0.02); vec3 accentColor = vec3(0.1, 0.02, 0.08) * abs(sin(u_time * 0.1)); vec3 finalColor = mix(baseColor, accentColor, 0.5 / length(p)); gl_FragColor = vec4(finalColor * 0.3, 1.0); }`;
    function cs(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; }
    const prog = gl.createProgram();
    gl.attachShader(prog, cs(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, cs(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog); gl.useProgram(prog);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(prog, 'a_position'); gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
    const uTime = gl.getUniformLocation(prog, 'u_time');
    const uRes = gl.getUniformLocation(prog, 'u_resolution');
    
    function render(t) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        if (uTime) gl.uniform1f(uTime, t * 0.001);
        if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        window.parent.requestAnimationFrame(render);
    }
    window.parent.requestAnimationFrame(render);
})();
</script>
""", height=0, width=0)

# ── Session State ──
def init_state():
    defaults = {"df": None, "schema_report": None, "agent": None,
                "api_key": os.getenv("GOOGLE_API_KEY", ""), "file_name": "",
                "analysis_result": None, "rag_initialized": False,
                "query_history": [], "session_start": time.strftime("%H:%M:%S"),
                "agent_status": "Idle", "llm_backend": "auto"}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ── Sidebar ──
with st.sidebar:
    # Brand
    st.markdown("""<div class="fluid-brand">
        <div class="ping"><span style="color:#fff;font-size:1rem;">◉</span></div>
        <span>Fluid Analyst</span>
    </div>""", unsafe_allow_html=True)

    # Home / New Session button (visible when data is loaded)
    if st.session_state.df is not None:
        if st.button("🏠 New Session", key="home_btn", use_container_width=True):
            st.session_state.df = None
            st.session_state.schema_report = None
            st.session_state.analysis_result = None
            st.session_state.file_name = ""
            st.session_state.agent = None
            st.session_state.rag_initialized = False
            st.session_state._agent_init_failed = False
            st.session_state.query_history = []
            st.rerun()

    # LLM Backend + API Key
    st.markdown('<div class="sidebar-label">Configuration</div>', unsafe_allow_html=True)

    backend_options = {"🔄 Auto-Detect": "auto", "✨ Google Gemini": "gemini",
                       "🦙 Ollama (Local)": "ollama", "🧠 GPT4All (Local)": "gpt4all",
                       "🎯 Demo Mode": "demo"}
    sel_backend = st.selectbox("LLM Backend", list(backend_options.keys()), index=0, key="backend_sel")
    st.session_state.llm_backend = backend_options[sel_backend]

    # Show API key input only when relevant
    if st.session_state.llm_backend in ("auto", "gemini"):
        api_key_input = st.text_input("🔑 API Key", value=st.session_state.api_key,
                                       type="password", help="Google Gemini API Key", key="api_key_w")
        if api_key_input:
            if api_key_input != st.session_state.api_key:
                st.session_state.agent = None
                st.session_state._agent_init_failed = False
                st.session_state.rag_initialized = False
            st.session_state.api_key = api_key_input

    # Backend status indicator
    from core.llm_backend import get_backend_status
    @st.cache_data(ttl=30, show_spinner=False)
    def _cached_backend_status(key):
        return get_backend_status(key)
    _bstatus = _cached_backend_status(st.session_state.api_key)
    if st.session_state.llm_backend == "demo":
        st.markdown('<span class="badge badge-ok">🎯 Demo Mode Active</span>', unsafe_allow_html=True)
        st.caption("Pre-computed templates • No setup needed")
    elif st.session_state.llm_backend == "gpt4all":
        if _bstatus["gpt4all"]["available"]:
            models = _bstatus["gpt4all"].get("models", [])
            if not models:
                st.markdown('<span class="badge badge-work">🧠 GPT4All — Needs Download</span>', unsafe_allow_html=True)
                st.caption("Phi-4-mini (~2.5GB) downloads on first use. Check terminal for progress.")
            else:
                st.markdown(f'<span class="badge badge-ok">🧠 GPT4All — {models[0][:20]}</span>', unsafe_allow_html=True)
                st.caption("Open-source LLM running locally on CPU")
        else:
            st.markdown('<span class="badge badge-ok">🎯 Demo Mode (fallback)</span>', unsafe_allow_html=True)
            st.caption("Install GPT4All: `pip install gpt4all`")
    elif st.session_state.llm_backend == "ollama":
        if _bstatus["ollama"]["available"]:
            st.markdown('<span class="badge badge-ok">🦙 Ollama Connected</span>', unsafe_allow_html=True)
        elif _bstatus["gpt4all"]["available"]:
            st.markdown('<span class="badge badge-ok">🧠 GPT4All (fallback)</span>', unsafe_allow_html=True)
            st.caption("Ollama not found → using GPT4All local LLM")
        else:
            st.markdown('<span class="badge badge-ok">🎯 Demo Mode (fallback)</span>', unsafe_allow_html=True)
            st.caption("No local LLM → using pre-computed analysis")
    elif st.session_state.llm_backend == "gemini":
        if _bstatus["gemini"]["available"]:
            st.markdown('<span class="badge badge-ok">✨ Gemini Ready</span>', unsafe_allow_html=True)
        elif _bstatus["gpt4all"]["available"]:
            st.markdown('<span class="badge badge-ok">🧠 GPT4All (fallback)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-ok">🎯 Demo Mode (fallback)</span>', unsafe_allow_html=True)
    elif st.session_state.llm_backend == "auto":
        if _bstatus["gemini"]["available"]:
            st.markdown('<span class="badge badge-ok">✨ Gemini (auto)</span>', unsafe_allow_html=True)
        elif _bstatus["ollama"]["available"]:
            st.markdown('<span class="badge badge-ok">🦙 Ollama (auto)</span>', unsafe_allow_html=True)
        elif _bstatus["gpt4all"]["available"]:
            st.markdown('<span class="badge badge-ok">🧠 GPT4All (auto)</span>', unsafe_allow_html=True)
            models = _bstatus["gpt4all"].get("models", [])
            if not models:
                st.caption("Phi-4-mini downloads on first use. Check terminal.")
            else:
                st.caption(f"{models[0][:24]} running locally on CPU")
        else:
            st.markdown('<span class="badge badge-ok">🎯 Demo Mode (auto)</span>', unsafe_allow_html=True)

    # Data Upload
    st.markdown('<div class="sidebar-label">Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json"],
                                      help="CSV, Excel, JSON up to 200MB", key="file_uploader_w")

    # Sample Data
    st.markdown('<div class="sidebar-label">Sample Data</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Sales", key="btn_s", width="stretch"):
            st.session_state._load_sample = "sales_data.csv"
    with c2:
        if st.button("👥 Cust.", key="btn_c", width="stretch"):
            st.session_state._load_sample = "customer_data.csv"
    with c3:
        if st.button("📈 Time", key="btn_t", width="stretch"):
            st.session_state._load_sample = "timeseries_data.csv"

    # Detected Columns (dynamic from loaded data)
    if st.session_state.df is not None:
        df_side = st.session_state.df
        st.markdown('<div class="sidebar-label">Detected Columns</div>', unsafe_allow_html=True)
        dtype_map = {"int64": "int", "float64": "float", "object": "str",
                     "datetime64[ns]": "date", "bool": "bool", "category": "cat"}
        for col in df_side.columns[:15]:
            dt = dtype_map.get(str(df_side[col].dtype), str(df_side[col].dtype))
            st.markdown(f'<div class="col-item"><span class="name">{col}</span><span class="dtype">{dt}</span></div>', unsafe_allow_html=True)
        if len(df_side.columns) > 15:
            st.caption(f"...and {len(df_side.columns) - 15} more columns")

    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        max_retries = st.slider("Max retries", 1, 5, 3, key="max_retries_s")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temp_s")

    # RAG Status
    st.markdown('<div class="sidebar-label">RAG Pipeline</div>', unsafe_allow_html=True)
    if st.session_state.rag_initialized:
        _rag_backend = ""
        if st.session_state.agent:
            _rag_backend = getattr(st.session_state.agent.rag, '_backend', '')
        if _rag_backend == "faiss":
            st.markdown('<span class="badge badge-ok">● FAISS Index Ready</span>', unsafe_allow_html=True)
        elif _rag_backend == "tfidf":
            st.markdown('<span class="badge badge-ok">● TF-IDF Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-ok">● Index Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-work">○ Not Initialized</span>', unsafe_allow_html=True)

    # Session Stats
    if st.session_state.agent:
        stats = st.session_state.agent.get_session_stats()
        if stats["total_queries"] > 0:
            st.markdown('<div class="sidebar-label">Session Stats</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="token-panel">
                <div class="token-row"><span class="tk-label">Queries</span><span class="tk-value">{stats['total_queries']}</span></div>
                <div class="token-row"><span class="tk-label">Total Tokens</span><span class="tk-value">{stats['total_tokens']:,}</span></div>
                <div class="token-row"><span class="tk-label">Est. Cost</span><span class="tk-value">${stats['total_cost_usd']:.4f}</span></div>
            </div>""", unsafe_allow_html=True)

    # Agent Status
    status_label = st.session_state.get("agent_status", "Idle")
    st.markdown(f"""<div class="agent-status">
        <div class="dot"></div><span class="label">Agent {status_label}</span>
    </div>""", unsafe_allow_html=True)

    st.caption("Built by **Umang Vijay** | JECRC University")

# ── Load Data ──
def load_data():
    from utils.data_loader import load_file
    if hasattr(st.session_state, '_load_sample') and st.session_state._load_sample:
        p = Path(__file__).parent / "sample_data" / st.session_state._load_sample
        if p.exists():
            st.session_state.df = pd.read_csv(p)
            st.session_state.file_name = st.session_state._load_sample
            st.session_state.schema_report = None
            st.session_state.analysis_result = None
        st.session_state._load_sample = None
    if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
        df, msg = load_file(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.schema_report = None
            st.session_state.analysis_result = None
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
load_data()

# ── Init Agent ──
def get_agent():
    from core.agent import DataScienceCoPilot
    if st.session_state.agent is None:
        if st.session_state.get("_agent_init_failed"):
            return None  # Don't retry if init already failed this session
        try:
            st.session_state.agent = DataScienceCoPilot(
                api_key=st.session_state.api_key,
                max_retries=st.session_state.get("max_retries_s", 3),
                temperature=st.session_state.get("temp_s", 0.2),
                backend=st.session_state.get("llm_backend", "auto"))
        except Exception as e:
            err_msg = str(e).lower()
            if "api_key_invalid" in err_msg or "api key not valid" in err_msg:
                st.error("🔑 Invalid API key. Please check your Google Gemini API key in the sidebar.")
            elif "quota" in err_msg:
                st.error("⚠️ API quota exceeded. Please wait or use a different API key.")
            else:
                st.error(f"⚠️ Agent initialization failed: {e}")
            st.session_state._agent_init_failed = True
            return None
    if st.session_state.agent and not st.session_state.rag_initialized:
        try:
            st.session_state.rag_initialized = st.session_state.agent.initialize_rag()
        except Exception:
            st.session_state.rag_initialized = False
    return st.session_state.agent

# Eagerly initialize agent so RAG is ready before the user clicks "Run Analysis"
get_agent()

# ── LLM Availability Check ──
# With the 4-tier fallback (Gemini → Ollama → GPT4All → Demo), an LLM is always available.
from core.llm_backend import check_ollama_available, check_gpt4all_available

@st.cache_data(ttl=30, show_spinner=False)
def _cached_llm_availability(api_key_val):
    """Cache expensive LLM availability checks to avoid blocking every rerender."""
    _key = api_key_val.strip()
    _placeholders = {"", "your_gemini_api_key_here", "your_api_key_here", "your_key_here", "PASTE_YOUR_KEY"}
    _has_valid = bool(_key) and _key.lower() not in {p.lower() for p in _placeholders}
    _ollama = check_ollama_available()
    _gpt4all = check_gpt4all_available()
    _active = ("gemini" if _has_valid else
               ("ollama" if _ollama else
                ("gpt4all" if _gpt4all else "demo")))
    return _has_valid, _ollama, _gpt4all, _active

_has_valid_key, _ollama_ok, _gpt4all_ok, active_backend = _cached_llm_availability(st.session_state.api_key)
has_api_key = True  # Always True — Demo Mode is the final fallback

# ── MAIN CONTENT ──
if st.session_state.df is not None:
    df = st.session_state.df

    # Auto Schema Analysis — works with or without API key
    if st.session_state.schema_report is None:
        with st.spinner("🔍 Analyzing dataset schema..."):
            if has_api_key:
                agent = get_agent()
                if agent:
                    try:
                        st.session_state.schema_report = agent.analyze_schema(df)
                    except Exception:
                        st.session_state.schema_report = None
            # Pandas-only fallback when no API key or agent failed
            if st.session_state.schema_report is None:
                from core.schema_analyzer import SchemaAnalyzer
                analyzer = SchemaAnalyzer(api_key=None)
                st.session_state.schema_report = analyzer.analyze(df)

    # Editorial AI Summary (dynamic from schema analysis)
    if st.session_state.schema_report:
        report = st.session_state.schema_report
        ai_summary = report.get("ai_summary", "")
        if ai_summary:
            # Split to highlight first number/percentage
            import re
            highlighted = re.sub(r'(\d+[\.,]?\d*\s*%?)', r'<span class="highlight">\1</span>', ai_summary, count=2)
            st.markdown(f'<div class="editorial">{highlighted}</div>', unsafe_allow_html=True)

        # Metric cards
        st.markdown(f"""<div class="metric-row">
            <div class="metric-box"><div class="val">{report['rows']:,}</div><div class="lbl">Rows</div></div>
            <div class="metric-box"><div class="val">{report['columns']}</div><div class="lbl">Columns</div></div>
            <div class="metric-box"><div class="val">{report['memory_mb']} MB</div><div class="lbl">Memory</div></div>
            <div class="metric-box"><div class="val">{report.get('missing_pct', 0)}%</div><div class="lbl">Missing</div></div>
            <div class="metric-box"><div class="val">{report.get('quality_score', 0)}%</div><div class="lbl">Quality</div></div>
        </div>""", unsafe_allow_html=True)

    # Tabs
    tab_schema, tab_preview, tab_analysis, tab_history = st.tabs(
        ["📋 Schema", "📊 Data Preview", "🤖 AI Analysis", "📜 History"])

    # ── Schema Tab ──
    with tab_schema:
        if st.session_state.schema_report:
            report = st.session_state.schema_report
            if report.get("ai_summary"):
                st.markdown(f'<div class="surface-card">{report["ai_summary"]}</div>', unsafe_allow_html=True)
            cl, cr = st.columns(2)
            with cl:
                st.subheader("📊 Column Details")
                col_data = []
                for cn in report.get("column_names", []):
                    dt = report["dtypes"].get(cn, "?")
                    mi = report["missing_values"].get(cn, {"count": 0, "percentage": 0})
                    col_data.append({"Column": cn, "Type": dt, "Missing": f"{mi['count']} ({mi['percentage']}%)"})
                if col_data:
                    st.dataframe(pd.DataFrame(col_data), width="stretch", hide_index=True)
            with cr:
                st.subheader("💡 Suggested Analyses")
                for s in report.get("suggested_use_cases", []):
                    st.markdown(f"• {s}")
                if report.get("numeric_stats"):
                    st.subheader("📈 Numeric Summary")
                    st.dataframe(pd.DataFrame(report["numeric_stats"]).round(2), width="stretch")

    # ── Preview Tab ──
    with tab_preview:
        st.subheader(f"📊 {st.session_state.file_name}")
        st.dataframe(df.head(100), width="stretch", height=400)
        with st.expander("📊 Column Statistics"):
            st.write(df.describe().round(2))

    # ── Analysis Tab ──
    with tab_analysis:
        st.subheader("🤖 Ask Your Data — Real-Time Analysis")

        use_case_map = {
            "📊 Sales Dashboard": "sales_dashboard",
            "🔍 Data Quality Audit": "data_quality",
            "📈 Trend Analysis": "trend_analysis",
            "👥 Cohort Analysis": "cohort_analysis",
            "🔮 Predictive Forecasting": "predictive_forecast",
            "💬 Ad-hoc Query": "ad_hoc",
        }
        suggestions = {
            "📊 Sales Dashboard": "Show total revenue by region as a bar chart",
            "🔍 Data Quality Audit": "Analyze data quality — find missing values and outliers",
            "📈 Trend Analysis": "Plot the time-series trend with a rolling average",
            "👥 Cohort Analysis": "Segment customers by spending level",
            "🔮 Predictive Forecasting": "Forecast the next 30 days using regression and ARIMA",
            "💬 Ad-hoc Query": "What are the top 5 products by total revenue?",
        }

        c_uc, c_q = st.columns([1, 2])
        with c_uc:
            sel_uc = st.selectbox("Analysis Type", list(use_case_map.keys()), index=5, key="uc_sel")
        with c_q:
            query = st.text_area("Your Question", placeholder="Ask anything about your data...", height=80, key="q_in")

        if not query:
            st.caption(f"💡 Suggestion: *{suggestions.get(sel_uc, '')}*")

        # Show which backend will be used
        if active_backend == "demo":
            st.info("🎯 **Demo Mode** — Using pre-computed analysis templates. For AI-powered analysis, add a Gemini API key or start Ollama.")
        elif active_backend == "ollama":
            st.info("🦙 **Ollama** — Using local LLM for analysis.")

        analyze_btn = st.button("🚀 Run Analysis", type="primary", width="stretch",
                                disabled=not query, key="run_btn")

        if analyze_btn and query:
            agent = get_agent()
            if agent is None:
                st.error("Agent not initialized. Check your API key.")
            else:
                uc_key = use_case_map[sel_uc]
                progress_bar = st.progress(0)
                status_container = st.empty()
                reasoning_container = st.container()
                reasoning_steps = []

                def progress_cb(msg):
                    reasoning_steps.append(msg)
                    status_container.markdown(f'<span class="badge badge-work">{msg}</span>', unsafe_allow_html=True)
                    # Build timeline
                    tl_html = '<div class="timeline">'
                    for i, step in enumerate(reasoning_steps):
                        is_last = (i == len(reasoning_steps) - 1)
                        dot_cls = "active" if is_last else "done"
                        inner = '<div class="inner"></div>' if is_last else "✓"
                        dim = "" if not is_last else ""
                        tl_html += f"""<div class="tl-step {dim}">
                            <div class="tl-dot {dot_cls}">{inner}</div>
                            <div class="tl-body"><h4>{step}</h4></div>
                        </div>"""
                    tl_html += '</div>'
                    reasoning_container.markdown(tl_html, unsafe_allow_html=True)

                st.session_state.agent_status = "Working"
                with st.spinner("🤖 Agent is working autonomously..."):
                    progress_bar.progress(10)
                    progress_cb("📋 Preparing analysis pipeline...")
                    result = agent.analyze(df=df, query=query, use_case=uc_key, progress_callback=progress_cb)
                    progress_bar.progress(100)

                st.session_state.agent_status = "Idle"
                st.session_state.analysis_result = result
                st.session_state.query_history.append({
                    "query": query, "use_case": sel_uc, "success": result.success,
                    "timestamp": result.timestamp, "tokens": result.token_usage.total_tokens,
                    "cost": result.token_usage.estimated_cost_usd,
                    "wall_time": result.token_usage.total_wall_time, "attempts": result.total_attempts,
                })
                status_container.empty()
                progress_bar.empty()

        # ── Display Results ──
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            tu = result.token_usage

            # Token metrics
            st.markdown("---")
            t1, t2, t3, t4, t5, t6 = st.columns(6)
            with t1: st.metric("🔤 Prompt", f"{tu.prompt_tokens:,}")
            with t2: st.metric("📝 Output", f"{tu.completion_tokens:,}")
            with t3: st.metric("📊 Total", f"{tu.total_tokens:,}")
            with t4: st.metric("⏱️ Time", f"{tu.total_wall_time}s")
            with t5: st.metric("🤖 LLM Calls", tu.llm_calls)
            with t6: st.metric("💰 Cost", f"${tu.estimated_cost_usd:.4f}")

            with st.expander("⏱️ Performance Breakdown"):
                perf_data = {
                    "Component": ["LLM Generation", "Sandbox Execution", "RAG Retrieval", "Total"],
                    "Time (s)": [tu.total_llm_time, tu.total_sandbox_time, tu.total_rag_time, tu.total_wall_time],
                    "% of Total": [
                        round(tu.total_llm_time / max(tu.total_wall_time, 0.01) * 100, 1),
                        round(tu.total_sandbox_time / max(tu.total_wall_time, 0.01) * 100, 1),
                        round(tu.total_rag_time / max(tu.total_wall_time, 0.01) * 100, 1), 100.0],
                }
                st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

            st.markdown("---")

            if result.success:
                st.markdown('<span class="badge badge-ok">✅ Analysis Complete</span>', unsafe_allow_html=True)

                # Charts in surface cards
                if result.chart_paths:
                    st.subheader("📊 Generated Charts")
                    for cp in result.chart_paths:
                        if cp.endswith(".png"):
                            st.image(cp, width="stretch")
                        elif cp.endswith(".html"):
                            try:
                                with open(cp, "r", encoding="utf-8") as f:
                                    st.components.v1.html(f.read(), height=500, scrolling=True)
                            except:
                                pass

                # Insights
                if result.insights:
                    st.subheader("💡 Insights")
                    raw_out = result.insights.pop("_raw_output", "")
                    if result.insights:
                        for k, v in result.insights.items():
                            st.markdown(f'<div class="surface-card"><strong>{k}:</strong> {v}</div>', unsafe_allow_html=True)
                    if raw_out:
                        with st.expander("📝 Full Output"):
                            st.text(raw_out)

                # Downloads
                st.subheader("📥 Download Results")

                # Code download with format selector
                st.markdown("**📝 Code Download**")
                code_fmt_col, code_dl_col = st.columns([1, 2])
                with code_fmt_col:
                    code_fmt = st.selectbox("Code Format", ["Python (.py)", "Jupyter Notebook (.ipynb)"], key="code_fmt_sel")
                with code_dl_col:
                    if code_fmt == "Python (.py)":
                        st.download_button("⬇️ Download Code", result.code,
                                           file_name="analysis_code.py", mime="text/x-python", width="stretch")
                    elif code_fmt == "Jupyter Notebook (.ipynb)":
                        # Build proper .ipynb JSON structure
                        nb_cells = [
                            {"cell_type": "markdown", "metadata": {}, "source": [
                                f"# Analysis Report\n", f"**Query:** {result.query}\n",
                                f"**Use Case:** {result.use_case}\n",
                                f"**Generated:** {result.timestamp}\n"]},
                            {"cell_type": "code", "metadata": {}, "source": result.code.split("\n"),
                             "execution_count": None, "outputs": []},
                        ]
                        notebook = {"nbformat": 4, "nbformat_minor": 5,
                                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                 "language_info": {"name": "python", "version": "3.10.0"}},
                                    "cells": nb_cells}
                        st.download_button("⬇️ Download Notebook", json.dumps(notebook, indent=2),
                                           file_name="analysis.ipynb", mime="application/json", width="stretch")

                # Report download with format selector
                st.markdown("**📋 Report Download**")
                rpt_fmt_col, rpt_dl_col = st.columns([1, 2])
                with rpt_fmt_col:
                    rpt_fmt = st.selectbox("Report Format", ["Markdown (.md)", "JSON (.json)", "CSV (.csv)"], key="rpt_fmt_sel")
                with rpt_dl_col:
                    if rpt_fmt == "Markdown (.md)":
                        report_md = f"# Analysis Report\n\n**Query:** {result.query}\n**Use Case:** {result.use_case}\n**Time:** {result.timestamp}\n\n## Insights\n"
                        for k, v in result.insights.items():
                            report_md += f"- **{k}:** {v}\n"
                        report_md += f"\n## Token Usage\n{json.dumps(tu.to_dict(), indent=2)}\n\n## Generated Code\n```python\n{result.code}\n```"
                        st.download_button("⬇️ Download Report", report_md,
                                           file_name="analysis_report.md", mime="text/markdown", width="stretch")
                    elif rpt_fmt == "JSON (.json)":
                        report_json = {
                            "query": result.query, "use_case": result.use_case,
                            "timestamp": result.timestamp, "success": result.success,
                            "insights": result.insights, "code": result.code,
                            "token_usage": tu.to_dict(),
                            "iterations": result.iterations,
                        }
                        st.download_button("⬇️ Download Report", json.dumps(report_json, indent=2, default=str),
                                           file_name="analysis_report.json", mime="application/json", width="stretch")
                    elif rpt_fmt == "CSV (.csv)":
                        import csv, io as _io
                        buf = _io.StringIO()
                        writer = csv.writer(buf)
                        writer.writerow(["Field", "Value"])
                        writer.writerow(["Query", result.query])
                        writer.writerow(["Use Case", result.use_case])
                        writer.writerow(["Timestamp", result.timestamp])
                        writer.writerow(["Success", result.success])
                        writer.writerow(["Total Tokens", tu.total_tokens])
                        writer.writerow(["Cost (USD)", f"${tu.estimated_cost_usd:.4f}"])
                        for k, v in result.insights.items():
                            writer.writerow([f"Insight: {k}", str(v)])
                        st.download_button("⬇️ Download Report", buf.getvalue(),
                                           file_name="analysis_report.csv", mime="text/csv", width="stretch")

                # Chart downloads
                if result.chart_paths:
                    st.markdown("**📊 Chart Download**")
                    for cp in result.chart_paths:
                        fname = Path(cp).name
                        if cp.endswith(".png"):
                            with open(cp, "rb") as f:
                                st.download_button(f"⬇️ {fname}", f.read(),
                                                   file_name=fname, mime="image/png", width="stretch")
                        elif cp.endswith(".html"):
                            with open(cp, "r", encoding="utf-8") as f:
                                st.download_button(f"⬇️ {fname}", f.read(),
                                                   file_name=fname, mime="text/html", width="stretch")

                # Code + Execution Log
                with st.expander("🔧 Generated Code"):
                    st.code(result.code, language="python")

                if result.iterations:
                    with st.expander(f"🔄 Execution Log ({result.total_attempts} attempts)"):
                        tl_html = '<div class="timeline">'
                        for it in result.iterations:
                            att = it.get("attempt", "?")
                            if it.get("success"):
                                tl_html += f"""<div class="tl-step"><div class="tl-dot done">✓</div>
                                    <div class="tl-body"><h4>Attempt {att} — Success</h4>
                                    <p>LLM: {it.get('llm_time','?')}s | Sandbox: {it.get('sandbox_time','?')}s</p></div></div>"""
                            else:
                                err = it.get("error", "Unknown")[:150]
                                fix = it.get("fix_applied", "")
                                tl_html += f"""<div class="tl-step"><div class="tl-dot done" style="border-color:var(--error);color:var(--error);">✗</div>
                                    <div class="tl-body"><h4>Attempt {att} — Failed</h4>
                                    <p>{err}</p>{"<p>🔧 " + fix + "</p>" if fix else ""}</div></div>"""
                        tl_html += '</div>'
                        st.markdown(tl_html, unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-err">❌ Analysis Failed</span>', unsafe_allow_html=True)
                st.error(result.error)
                if result.code:
                    with st.expander("🔧 Last Generated Code"):
                        st.code(result.code, language="python")
                if result.iterations:
                    with st.expander("🔄 Execution Log"):
                        for it in result.iterations:
                            st.markdown(f"**Attempt {it.get('attempt')}:** ❌ {it.get('error', 'Unknown')[:200]}")

    # ── History Tab ──
    with tab_history:
        st.subheader("📜 Query History")
        if st.session_state.query_history:
            hist = st.session_state.query_history
            total_tok = sum(h["tokens"] for h in hist)
            total_cost = sum(h["cost"] for h in hist)
            success_rate = sum(1 for h in hist if h["success"]) / len(hist) * 100

            hm1, hm2, hm3, hm4 = st.columns(4)
            with hm1: st.metric("📝 Queries", len(hist))
            with hm2: st.metric("🔤 Tokens", f"{total_tok:,}")
            with hm3: st.metric("💰 Cost", f"${total_cost:.4f}")
            with hm4: st.metric("✅ Success", f"{success_rate:.0f}%")

            st.markdown("---")
            hist_df = pd.DataFrame(hist)
            hist_df = hist_df[["timestamp", "query", "use_case", "success", "attempts", "tokens", "cost", "wall_time"]]
            hist_df.columns = ["Time", "Query", "Use Case", "Success", "Attempts", "Tokens", "Cost ($)", "Time (s)"]
            st.dataframe(hist_df, width="stretch", hide_index=True)

            st.download_button("⬇️ Export History", hist_df.to_csv(index=False),
                               file_name="query_history.csv", mime="text/csv", width="stretch")
        else:
            st.info("No queries yet. Run an analysis from the **🤖 AI Analysis** tab.")

else:
    # ── Welcome Page — Immersive Fluid Analyst Design ──

    # Hero Section
    st.markdown("""<div class="hero-section">
        <div class="hero-title">
            Your <span class="gradient-text">AI-Powered</span> Data Analyst
        </div>
        <div class="hero-subtitle">
            Upload a CSV, Excel, or JSON file — ask questions in plain English,
            get professional charts, insights, and predictive forecasts instantly.
        </div>
        <div class="hero-powered">Powered by Gemini • Ollama • GPT4All • RAG</div>
    </div>""", unsafe_allow_html=True)

    # 3D Organic AI Core Animation
    components.html("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r125/three.min.js"></script>
    <div id="threejs-container" style="width:100%;height:350px; display:flex; justify-content:center; align-items:center;"></div>
    <script>
    (function() {
        const container = document.getElementById('threejs-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);
        
        const geometry = new THREE.IcosahedronGeometry(2, 4);
        const material = new THREE.MeshPhongMaterial({
            color: 0xff2d55, wireframe: true, transparent: true, opacity: 0.4,
            emissive: 0x8e2de2, emissiveIntensity: 0.5
        });
        const core = new THREE.Mesh(geometry, material);
        scene.add(core);
        
        const innerGeo = new THREE.SphereGeometry(1.2, 32, 32);
        const innerMat = new THREE.MeshBasicMaterial({ color: 0xff2d55, transparent: true, opacity: 0.8 });
        const innerCore = new THREE.Mesh(innerGeo, innerMat);
        scene.add(innerCore);
        
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(5, 5, 5);
        scene.add(light);
        
        camera.position.z = 5;
        
        function animate() {
            requestAnimationFrame(animate);
            core.rotation.x += 0.005;
            core.rotation.y += 0.005;
            const scale = 1 + Math.sin(Date.now() * 0.001) * 0.05;
            innerCore.scale.set(scale, scale, scale);
            innerCore.material.opacity = 0.4 + Math.sin(Date.now() * 0.002) * 0.2;
            renderer.render(scene, camera);
        }
        
        window.addEventListener('resize', () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });
        
        animate();
    })();
    </script>
    """, height=350)

    # Centered Upload Card
    st.markdown("""<div class="upload-card">
        <div class="upload-icon">☁️</div>
        <div class="upload-title">Upload Dataset</div>
        <div class="upload-desc">CSV, Excel, JSON — or use a sample dataset from the sidebar</div>
    </div>""", unsafe_allow_html=True)

    # File uploader in main area for easy access
    col_pad1, col_up, col_pad2 = st.columns([1, 2, 1])
    with col_up:
        main_upload = st.file_uploader("Drop your file here", type=["csv", "xlsx", "xls", "json"],
                                        help="CSV, Excel, JSON up to 200MB", key="main_uploader",
                                        label_visibility="collapsed")
        if main_upload is not None and main_upload.name != st.session_state.file_name:
            from utils.data_loader import load_file
            df_up, msg_up = load_file(main_upload)
            if df_up is not None:
                st.session_state.df = df_up
                st.session_state.file_name = main_upload.name
                st.session_state.schema_report = None
                st.session_state.analysis_result = None
                st.rerun()
            else:
                st.error(msg_up)

    # Core Capabilities Grid
    st.markdown("""<div class="cap-grid">
        <div class="cap-card">
            <div class="cap-icon">🔄</div>
            <div class="cap-title">Self-Correcting AI Agent</div>
            <div class="cap-desc">Autonomous code generation with error detection and RAG-powered self-correction. Retries up to 3 times using documentation context.</div>
        </div>
        <div class="cap-card">
            <div class="cap-icon">📊</div>
            <div class="cap-title">Professional Charts</div>
            <div class="cap-desc">Interactive Plotly and Matplotlib visualizations generated from plain English queries. Publication-ready with dark theme styling.</div>
        </div>
        <div class="cap-card">
            <div class="cap-icon">🔮</div>
            <div class="cap-title">Predictive Forecasting</div>
            <div class="cap-desc">Linear regression and ARIMA time-series forecasting with confidence intervals. Automatic trend detection and growth analysis.</div>
        </div>
        <div class="cap-card">
            <div class="cap-icon">🧠</div>
            <div class="cap-title">RAG Pipeline</div>
            <div class="cap-desc">FAISS or TF-IDF vector search over Python/Pandas documentation for intelligent error recovery and code self-correction.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Backend Status Cards
    st.markdown("")
    st.markdown("")
    from core.llm_backend import check_gpt4all_available
    _gpt4all_ok_w = check_gpt4all_available()
    _gemini_status = "✅ Ready" if _has_valid_key else "❌ No key"
    _ollama_status = "✅ Running" if _ollama_ok else "❌ Offline"
    _gpt4all_status = "✅ Installed" if _gpt4all_ok_w else "❌ Missing"
    _gemini_cls = "be-ok" if _has_valid_key else "be-err"
    _ollama_cls = "be-ok" if _ollama_ok else "be-err"
    _gpt4all_cls = "be-ok" if _gpt4all_ok_w else "be-err"

    st.markdown(f"""<div class="backend-grid">
        <div class="backend-card">
            <div class="be-tier">TIER 1</div>
            <div class="be-name">✨ Gemini</div>
            <div class="be-status {_gemini_cls}">{_gemini_status}</div>
        </div>
        <div class="backend-card">
            <div class="be-tier">TIER 2</div>
            <div class="be-name">🦙 Ollama</div>
            <div class="be-status {_ollama_cls}">{_ollama_status}</div>
        </div>
        <div class="backend-card">
            <div class="be-tier">TIER 3</div>
            <div class="be-name">🧠 GPT4All</div>
            <div class="be-status {_gpt4all_cls}">{_gpt4all_status}</div>
        </div>
        <div class="backend-card">
            <div class="be-tier">TIER 4</div>
            <div class="be-name">🎯 Demo</div>
            <div class="be-status be-ok">✅ Always On</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if active_backend == "demo":
        st.info("🎯 **Demo Mode active** — Upload data and click Run Analysis. Works without any API keys!")
    elif active_backend == "gpt4all":
        st.success("🧠 **GPT4All (Phi-4-mini)** running locally on CPU — no API keys needed!")

    # Footer
    st.markdown("""<div class="app-footer">
        <div class="footer-brand">Fluid Analyst</div>
        <div class="footer-sub">Built by Umang Vijay • JECRC University • Celebal CEI Internship</div>
    </div>""", unsafe_allow_html=True)
