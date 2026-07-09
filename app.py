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

# ── Session State ──
def init_state():
    defaults = {"df": None, "schema_report": None, "agent": None,
                "api_key": os.getenv("GOOGLE_API_KEY", ""), "file_name": "",
                "analysis_result": None, "rag_initialized": False,
                "query_history": [], "session_start": time.strftime("%H:%M:%S"),
                "agent_status": "Idle"}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ── Sidebar ──
with st.sidebar:
    # Brand
    st.markdown("""<div class="fluid-brand">
        <div class="ping"><span style="color:#00F0FF;font-size:1.2rem;">◉</span></div>
        <span>Fluid Analyst</span>
    </div>""", unsafe_allow_html=True)

    # API Key
    st.markdown('<div class="sidebar-label">Configuration</div>', unsafe_allow_html=True)
    api_key_input = st.text_input("🔑 API Key", value=st.session_state.api_key,
                                   type="password", help="Google Gemini API Key", key="api_key_w")
    if api_key_input:
        st.session_state.api_key = api_key_input

    # Data Upload
    st.markdown('<div class="sidebar-label">Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json"],
                                      help="CSV, Excel, JSON up to 200MB", key="file_uploader_w")

    # Sample Data
    st.markdown('<div class="sidebar-label">Sample Data</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Sales", key="btn_s", use_container_width=True):
            st.session_state._load_sample = "sales_data.csv"
    with c2:
        if st.button("👥 Cust.", key="btn_c", use_container_width=True):
            st.session_state._load_sample = "customer_data.csv"
    with c3:
        if st.button("📈 Time", key="btn_t", use_container_width=True):
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
    if st.session_state.agent is None and st.session_state.api_key:
        try:
            st.session_state.agent = DataScienceCoPilot(
                api_key=st.session_state.api_key,
                max_retries=st.session_state.get("max_retries_s", 3),
                temperature=st.session_state.get("temp_s", 0.2))
        except Exception as e:
            st.error(f"Agent init failed: {e}")
            return None
    if st.session_state.agent and not st.session_state.rag_initialized:
        try:
            st.session_state.rag_initialized = st.session_state.agent.initialize_rag()
        except:
            st.session_state.rag_initialized = False
    return st.session_state.agent

# ── API Key Gate ──
if not st.session_state.api_key:
    st.warning("⚠️ Enter your Google Gemini API key in the sidebar.")
    st.info("**Get a free key:** [Google AI Studio](https://aistudio.google.com/) → Get API Key")
    st.stop()

# ── MAIN CONTENT ──
if st.session_state.df is not None:
    df = st.session_state.df

    # Auto Schema Analysis (real-time via AI)
    if st.session_state.schema_report is None:
        with st.spinner("🔍 Analyzing dataset schema..."):
            agent = get_agent()
            if agent:
                st.session_state.schema_report = agent.analyze_schema(df)

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
                    st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
            with cr:
                st.subheader("💡 Suggested Analyses")
                for s in report.get("suggested_use_cases", []):
                    st.markdown(f"• {s}")
                if report.get("numeric_stats"):
                    st.subheader("📈 Numeric Summary")
                    st.dataframe(pd.DataFrame(report["numeric_stats"]).round(2), use_container_width=True)

    # ── Preview Tab ──
    with tab_preview:
        st.subheader(f"📊 {st.session_state.file_name}")
        st.dataframe(df.head(100), use_container_width=True, height=400)
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

        analyze_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not query, key="run_btn")

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
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            if result.success:
                st.markdown('<span class="badge badge-ok">✅ Analysis Complete</span>', unsafe_allow_html=True)

                # Charts in surface cards
                if result.chart_paths:
                    st.subheader("📊 Generated Charts")
                    for cp in result.chart_paths:
                        if cp.endswith(".png"):
                            st.image(cp, use_container_width=True)
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
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button("⬇️ Code", result.code, file_name="generated_code.py",
                                       mime="text/x-python", use_container_width=True)
                with dl2:
                    report_md = f"# Analysis Report\n\n**Query:** {result.query}\n**Use Case:** {result.use_case}\n**Time:** {result.timestamp}\n\n## Insights\n"
                    for k, v in result.insights.items():
                        report_md += f"- **{k}:** {v}\n"
                    report_md += f"\n## Token Usage\n{json.dumps(tu.to_dict(), indent=2)}\n\n## Generated Code\n```python\n{result.code}\n```"
                    st.download_button("⬇️ Report", report_md, file_name="analysis_report.md",
                                       mime="text/markdown", use_container_width=True)
                with dl3:
                    for cp in result.chart_paths:
                        if cp.endswith(".png"):
                            with open(cp, "rb") as f:
                                st.download_button("⬇️ Chart", f.read(), file_name=Path(cp).name,
                                                   mime="image/png", use_container_width=True)
                            break

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
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            st.download_button("⬇️ Export History", hist_df.to_csv(index=False),
                               file_name="query_history.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No queries yet. Run an analysis from the **🤖 AI Analysis** tab.")

else:
    # Welcome screen
    st.markdown("""<div class="welcome-box">
        <h2>Your AI-Powered Data Analyst</h2>
        <p>Upload a CSV, Excel, or JSON file to get started.<br/>Ask questions in plain English — get professional charts and insights instantly.</p>
        <div class="feature-grid">
            <div class="feature-item"><div class="icon">🔄</div><div class="title">Self-Correcting AI</div><div class="desc">Writes, executes, and fixes code using RAG over Python/Pandas docs.</div></div>
            <div class="feature-item"><div class="icon">📊</div><div class="title">Professional Charts</div><div class="desc">Interactive Plotly and Matplotlib charts from plain English.</div></div>
            <div class="feature-item"><div class="icon">🔮</div><div class="title">Predictive Forecasting</div><div class="desc">Regression + ARIMA forecasting with confidence intervals.</div></div>
            <div class="feature-item"><div class="icon">🧠</div><div class="title">RAG Pipeline</div><div class="desc">Self-correction via FAISS vector search over official docs.</div></div>
        </div>
    </div>""", unsafe_allow_html=True)
