"""
LLM Backend Factory — Multi-provider LLM support with 4-tier fallback.

Tier 1: Google Gemini   (API key required — best quality)
Tier 2: Ollama (local)  (server-based local LLM — good quality)
Tier 3: GPT4All (local) (pip-installable local LLM, no server — good quality)
Tier 4: Demo Mode       (pre-computed, no setup — for evaluation)

Usage:
    llm, backend = get_llm(backend="auto", api_key="...", temperature=0.2)
"""

import urllib.request
import json
import os
from typing import Optional, List
from pathlib import Path

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration


# ─────────────────────────────────────────────────────────
# Utility: Ollama detection
# ─────────────────────────────────────────────────────────

def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running locally."""
    try:
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        return req.status == 200
    except Exception:
        return False


def get_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """Get list of available Ollama models."""
    try:
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        data = json.loads(req.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────
# Utility: GPT4All detection
# ─────────────────────────────────────────────────────────

def check_gpt4all_available() -> bool:
    """Check if GPT4All is installed and a model exists locally."""
    try:
        import gpt4all
        return True
    except ImportError:
        return False


def get_gpt4all_local_models() -> list:
    """Get list of locally downloaded GPT4All GGUF models."""
    try:
        from gpt4all import GPT4All
        model_dir = Path.home() / ".cache" / "gpt4all"
        if not model_dir.exists():
            return []
        return [f.name for f in model_dir.glob("*.gguf")]
    except Exception:
        return []


class GPT4AllChatLLM(BaseChatModel):
    """
    LangChain-compatible wrapper around GPT4All for local CPU inference.

    Uses Llama-3.2-1B-Instruct (700MB) by default — runs on any laptop.
    Auto-downloads the model on first use.
    """

    _gpt4all_instance: object = None
    _model_name: str = ""

    @property
    def _llm_type(self) -> str:
        return "gpt4all"

    def _ensure_model(self):
        """Lazy-load the GPT4All model."""
        if self._gpt4all_instance is not None:
            return
        from gpt4all import GPT4All

        # Prefer already-downloaded models, else download the best for code
        local_models = get_gpt4all_local_models()
        preferred = [
            "Phi-4-mini-instruct-Q4_0.gguf",      # ~2.5GB, best for code gen
            "Llama-3.2-3B-Instruct-Q4_0.gguf",    # 2GB, good quality
            "Llama-3.2-1B-Instruct-Q4_0.gguf",    # 700MB, fast fallback
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",  # 1GB
        ]
        chosen = None
        for pref in preferred:
            if pref in local_models:
                chosen = pref
                break
        if not chosen:
            # Auto-download Phi-4-mini (best code quality for CPU)
            chosen = "Phi-4-mini-instruct-Q4_0.gguf"

        self._model_name = chosen
        self._gpt4all_instance = GPT4All(chosen, allow_download=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using the local GPT4All model."""
        self._ensure_model()

        # Build prompt from messages
        prompt_parts = []
        for m in messages:
            if hasattr(m, "content"):
                role = getattr(m, "type", "user")
                if role == "system":
                    prompt_parts.append(f"[SYSTEM] {m.content}")
                else:
                    prompt_parts.append(m.content)
        prompt = "\n".join(prompt_parts)

        # Generate with the local model (2048 tokens = fast on CPU)
        response = self._gpt4all_instance.generate(
            prompt, max_tokens=2048, temp=0.2
        )

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )


# ─────────────────────────────────────────────────────────
# Tier 4: Demo Mode LLM (pre-computed responses)
# ─────────────────────────────────────────────────────────

class DemoModeLLM(BaseChatModel):
    """
    A pre-computed LLM that generates deterministic Python/Pandas
    analysis code without any external API. Used for evaluation
    when no Gemini API key or Ollama is available.

    This is Tier 3 in the fallback chain.
    """

    @property
    def _llm_type(self) -> str:
        return "demo-mode"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a pre-computed response based on the query context."""
        prompt = " ".join(
            m.content for m in messages if hasattr(m, "content")
        ).lower()

        code = self._pick_demo_code(prompt)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"```python\n{code}\n```"))]
        )

    def _pick_demo_code(self, prompt: str) -> str:
        """Select the best pre-computed code based on the query keywords."""

        # ── Sales Dashboard ──
        if any(k in prompt for k in ["revenue", "sales", "bar chart", "region", "sales_dashboard"]):
            return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Sales Dashboard — Revenue by Region
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0E1117")

# Try to find revenue and region columns
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
revenue_col = next((c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower() or "amount" in c.lower()), num_cols[0] if num_cols else None)
region_col = next((c for c in cat_cols if "region" in c.lower() or "category" in c.lower() or "product" in c.lower()), cat_cols[0] if cat_cols else None)

if revenue_col and region_col:
    grouped = df.groupby(region_col)[revenue_col].sum().sort_values(ascending=False)

    # Bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#0E1117")
    bars = ax1.bar(grouped.index, grouped.values, color=["#00F0FF", "#D200FF", "#00E676", "#FF6B6B", "#FFD93D"][:len(grouped)])
    ax1.set_title(f"{revenue_col} by {region_col}", color="white", fontsize=14)
    ax1.tick_params(colors="white")
    ax1.set_ylabel(revenue_col, color="white")

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor("#0E1117")
    ax2.pie(grouped.values, labels=grouped.index, autopct="%1.1f%%",
            colors=["#00F0FF", "#D200FF", "#00E676", "#FF6B6B", "#FFD93D"][:len(grouped)],
            textprops={"color": "white"})
    ax2.set_title(f"{region_col} Distribution", color="white", fontsize=14)

plt.tight_layout()
save_chart(fig, "sales_dashboard")

total = df[revenue_col].sum()
avg = df[revenue_col].mean()
top = grouped.index[0] if len(grouped) > 0 else "N/A"
report_insights({
    "Total Revenue": f"${total:,.2f}",
    "Average Revenue": f"${avg:,.2f}",
    "Top Region": top,
    "Number of Records": f"{len(df):,}",
    "Unique Categories": f"{df[region_col].nunique()}"
})
print(f"Sales dashboard generated: {len(grouped)} categories, total ${total:,.2f}")
'''

        # ── Data Quality Audit ──
        if any(k in prompt for k in ["quality", "missing", "outlier", "duplicate", "audit", "data_quality"]):
            return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data Quality Audit
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0E1117")

# Missing values heatmap
ax1 = axes[0]
ax1.set_facecolor("#0E1117")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
colors = ["#00E676" if v == 0 else "#FF6B6B" for v in missing.values]
ax1.barh(missing.index[:15], missing_pct.values[:15], color=colors[:15])
ax1.set_title("Missing Values (%)", color="white", fontsize=14)
ax1.tick_params(colors="white")
ax1.set_xlabel("Percentage", color="white")

# Duplicate analysis
ax2 = axes[1]
ax2.set_facecolor("#0E1117")
num_cols = df.select_dtypes(include="number").columns[:5]
if len(num_cols) > 0:
    bp = ax2.boxplot([df[c].dropna().values for c in num_cols], labels=num_cols,
                     patch_artist=True, boxprops=dict(facecolor="#00F0FF", alpha=0.5),
                     medianprops=dict(color="#D200FF"))
    ax2.set_title("Outlier Detection (Box Plot)", color="white", fontsize=14)
    ax2.tick_params(colors="white")

plt.tight_layout()
save_chart(fig, "data_quality_audit")

duplicates = df.duplicated().sum()
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
quality_score = round((1 - total_missing / total_cells) * 100, 2)

# Outlier detection via IQR
outlier_count = 0
for col in df.select_dtypes(include="number").columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_count += ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()

report_insights({
    "Data Quality Score": f"{quality_score}%",
    "Total Missing Values": f"{total_missing:,}",
    "Duplicate Rows": f"{duplicates:,}",
    "Outliers Detected (IQR)": f"{outlier_count:,}",
    "Total Rows": f"{len(df):,}",
    "Total Columns": f"{len(df.columns)}"
})
print(f"Quality audit complete: score={quality_score}%, missing={total_missing}, duplicates={duplicates}")
'''

        # ── Trend Analysis ──
        if any(k in prompt for k in ["trend", "time", "series", "growth", "traffic", "trend_analysis"]):
            return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Trend Analysis with Rolling Average
fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0E1117")
ax.set_facecolor("#0E1117")

num_cols = df.select_dtypes(include="number").columns.tolist()
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
value_col = num_cols[0] if num_cols else None

if date_cols:
    df["_date_parsed"] = pd.to_datetime(df[date_cols[0]], format="mixed", errors="coerce")
    df_sorted = df.dropna(subset=["_date_parsed"]).sort_values("_date_parsed")
    x = df_sorted["_date_parsed"]
else:
    df_sorted = df.copy()
    x = range(len(df_sorted))

if value_col:
    y = df_sorted[value_col].values
    ax.plot(x, y, color="#00F0FF", alpha=0.4, linewidth=1, label="Raw Data")

    # Rolling average
    window = max(7, len(y) // 20)
    rolling = pd.Series(y).rolling(window=window, center=True).mean()
    ax.plot(x, rolling, color="#D200FF", linewidth=2.5, label=f"Rolling Avg ({window})")

    # Linear trend line
    x_num = np.arange(len(y))
    z = np.polyfit(x_num, y, 1)
    trend_line = np.poly1d(z)(x_num)
    ax.plot(x, trend_line, color="#00E676", linewidth=2, linestyle="--", label="Linear Trend")

    ax.set_title(f"Trend Analysis: {value_col}", color="white", fontsize=14)
    ax.set_ylabel(value_col, color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#16203C", edgecolor="#00F0FF", labelcolor="white")

plt.tight_layout()
save_chart(fig, "trend_analysis")

slope_pct = (z[0] / np.mean(y) * 100) if value_col else 0
direction = "📈 Upward" if slope_pct > 0 else "📉 Downward"
report_insights({
    "Trend Direction": direction,
    "Growth Rate": f"{slope_pct:.2f}% per period",
    "Min Value": f"{np.min(y):,.2f}" if value_col else "N/A",
    "Max Value": f"{np.max(y):,.2f}" if value_col else "N/A",
    "Data Points": f"{len(df):,}"
})
print(f"Trend analysis complete: {direction}, slope={slope_pct:.2f}%")
'''

        # ── Cohort Analysis ──
        if any(k in prompt for k in ["cohort", "segment", "customer", "group", "cohort_analysis"]):
            return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Cohort / Segmentation Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0E1117")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
segment_col = next((c for c in cat_cols if "segment" in c.lower() or "category" in c.lower() or "type" in c.lower()), cat_cols[0] if cat_cols else None)
value_col = next((c for c in num_cols if "spend" in c.lower() or "revenue" in c.lower() or "amount" in c.lower()), num_cols[0] if num_cols else None)

if segment_col and value_col:
    grouped = df.groupby(segment_col)[value_col].agg(["mean", "count", "sum"]).sort_values("sum", ascending=False)

    # Grouped bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#0E1117")
    colors = ["#00F0FF", "#D200FF", "#00E676", "#FF6B6B", "#FFD93D"]
    bars = ax1.bar(grouped.index[:5], grouped["mean"][:5], color=colors[:len(grouped[:5])])
    ax1.set_title(f"Average {value_col} by {segment_col}", color="white", fontsize=14)
    ax1.tick_params(colors="white", axis="both")
    ax1.set_ylabel(f"Avg {value_col}", color="white")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Count distribution
    ax2 = axes[1]
    ax2.set_facecolor("#0E1117")
    ax2.pie(grouped["count"][:5].values, labels=grouped.index[:5], autopct="%1.1f%%",
            colors=colors[:len(grouped[:5])], textprops={"color": "white"})
    ax2.set_title(f"{segment_col} Distribution", color="white", fontsize=14)

plt.tight_layout()
save_chart(fig, "cohort_analysis")

n_segments = df[segment_col].nunique() if segment_col else 0
report_insights({
    "Total Segments": str(n_segments),
    "Largest Segment": grouped.index[0] if len(grouped) > 0 else "N/A",
    "Highest Avg Value": f"${grouped['mean'].max():,.2f}" if len(grouped) > 0 else "N/A",
    "Total Records": f"{len(df):,}"
})
print(f"Cohort analysis complete: {n_segments} segments found")
'''

        # ── Predictive Forecasting ──
        if any(k in prompt for k in ["forecast", "predict", "arima", "regression", "future", "predictive_forecast"]):
            return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Predictive Forecasting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0E1117")

num_cols = df.select_dtypes(include="number").columns.tolist()
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
value_col = num_cols[0] if num_cols else None

if date_cols:
    df["_date_parsed"] = pd.to_datetime(df[date_cols[0]], format="mixed", errors="coerce")
    df_sorted = df.dropna(subset=["_date_parsed"]).sort_values("_date_parsed")
    x_vals = np.arange(len(df_sorted))
else:
    df_sorted = df.copy()
    x_vals = np.arange(len(df_sorted))

if value_col:
    y = df_sorted[value_col].values

    # Linear Regression Forecast
    ax1 = axes[0]
    ax1.set_facecolor("#0E1117")
    z = np.polyfit(x_vals, y, 1)
    trend = np.poly1d(z)

    # Extend forecast 30 periods
    future_x = np.arange(len(y) + 30)
    ax1.scatter(x_vals, y, color="#00F0FF", alpha=0.3, s=5, label="Historical")
    ax1.plot(future_x, trend(future_x), color="#D200FF", linewidth=2, label="Linear Forecast")
    ax1.axvline(x=len(y), color="#FF6B6B", linestyle="--", alpha=0.5, label="Forecast Start")
    ax1.set_title(f"Linear Regression Forecast: {value_col}", color="white", fontsize=14)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#16203C", edgecolor="#00F0FF", labelcolor="white")

    # Moving Average Forecast
    ax2 = axes[1]
    ax2.set_facecolor("#0E1117")
    window = max(7, len(y) // 15)
    ma = pd.Series(y).rolling(window=window).mean()
    ax2.plot(x_vals, y, color="#00F0FF", alpha=0.3, linewidth=1, label="Actual")
    ax2.plot(x_vals, ma, color="#00E676", linewidth=2.5, label=f"MA({window})")
    # Simple forecast: extend last MA value
    last_ma = ma.dropna().iloc[-1] if len(ma.dropna()) > 0 else np.mean(y)
    forecast_y = [last_ma + z[0] * i for i in range(30)]
    forecast_x = np.arange(len(y), len(y) + 30)
    ax2.plot(forecast_x, forecast_y, color="#FFD93D", linewidth=2, linestyle="--", label="Forecast")
    ax2.axvline(x=len(y), color="#FF6B6B", linestyle="--", alpha=0.5)
    ax2.set_title("Moving Average + Forecast", color="white", fontsize=14)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16203C", edgecolor="#00F0FF", labelcolor="white")

plt.tight_layout()
save_chart(fig, "forecast")

slope_pct = (z[0] / np.mean(y) * 100) if value_col else 0
report_insights({
    "Forecast Method": "Linear Regression + Moving Average",
    "Trend Slope": f"{z[0]:.4f} per period" if value_col else "N/A",
    "Growth Rate": f"{slope_pct:.2f}% per period",
    "Current Value": f"{y[-1]:,.2f}" if value_col else "N/A",
    "Predicted (30 periods)": f"{trend(len(y)+30):,.2f}" if value_col else "N/A",
    "Data Points": f"{len(df):,}"
})
print(f"Forecast complete: slope={z[0]:.4f}, growth={slope_pct:.2f}%")
'''

        # ── Default: Ad-hoc / Generic Analysis ──
        return '''import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ad-hoc Data Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0E1117")

num_cols = df.select_dtypes(include="number").columns.tolist()[:5]
cat_cols = df.select_dtypes(include="object").columns.tolist()[:5]

# Numeric column distributions
ax1 = axes[0]
ax1.set_facecolor("#0E1117")
if num_cols:
    colors = ["#00F0FF", "#D200FF", "#00E676", "#FF6B6B", "#FFD93D"]
    for i, col in enumerate(num_cols[:3]):
        ax1.hist(df[col].dropna(), bins=30, alpha=0.6, color=colors[i % len(colors)], label=col)
    ax1.set_title("Numeric Distributions", color="white", fontsize=14)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#16203C", edgecolor="#00F0FF", labelcolor="white")

# Categorical value counts
ax2 = axes[1]
ax2.set_facecolor("#0E1117")
if cat_cols:
    top_col = cat_cols[0]
    counts = df[top_col].value_counts().head(8)
    ax2.barh(counts.index, counts.values, color="#00F0FF")
    ax2.set_title(f"Top Values: {top_col}", color="white", fontsize=14)
    ax2.tick_params(colors="white")

plt.tight_layout()
save_chart(fig, "analysis")

insights = {"Total Rows": f"{len(df):,}", "Total Columns": f"{len(df.columns)}"}
for col in num_cols[:3]:
    insights[f"{col} (mean)"] = f"{df[col].mean():,.2f}"
    insights[f"{col} (std)"] = f"{df[col].std():,.2f}"
if cat_cols:
    insights[f"Unique {cat_cols[0]}"] = str(df[cat_cols[0]].nunique())
report_insights(insights)
print(f"Analysis complete: {len(df)} rows, {len(df.columns)} columns")
'''


# ─────────────────────────────────────────────────────────
# Main Factory Function
# ─────────────────────────────────────────────────────────

def get_llm(backend: str = "auto", api_key: Optional[str] = None,
            temperature: float = 0.2, model: Optional[str] = None,
            max_output_tokens: int = 4096):
    """
    Create a LangChain-compatible LLM instance using the 4-tier fallback:
      Tier 1: Google Gemini  (API key) — best quality
      Tier 2: Ollama (local server)    — good quality
      Tier 3: GPT4All (local pip)      — good quality, no server
      Tier 4: Demo Mode      (none)    — pre-computed, for evaluation

    Returns (llm_instance, backend_name).
    """
    if backend == "auto":
        # Tier 1: Gemini
        if api_key and len(api_key) > 10:
            try:
                llm = _create_gemini(api_key, temperature, model, max_output_tokens)
                return llm, "gemini"
            except Exception:
                pass
        # Tier 2: Ollama
        if check_ollama_available():
            try:
                llm = _create_ollama(temperature, model, max_output_tokens)
                return llm, "ollama"
            except Exception:
                pass
        # Tier 3: GPT4All
        if check_gpt4all_available():
            try:
                return GPT4AllChatLLM(), "gpt4all"
            except Exception:
                pass
        # Tier 4: Demo Mode
        return DemoModeLLM(), "demo"

    elif backend == "gemini":
        if not api_key:
            raise RuntimeError("Gemini backend requires an API key.")
        return _create_gemini(api_key, temperature, model, max_output_tokens), "gemini"

    elif backend == "ollama":
        if check_ollama_available():
            return _create_ollama(temperature, model, max_output_tokens), "ollama"
        # Ollama not running — try GPT4All, then Demo
        if check_gpt4all_available():
            return GPT4AllChatLLM(), "gpt4all"
        return DemoModeLLM(), "demo"

    elif backend == "gpt4all":
        if check_gpt4all_available():
            return GPT4AllChatLLM(), "gpt4all"
        return DemoModeLLM(), "demo"

    elif backend == "demo":
        return DemoModeLLM(), "demo"

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'gemini', 'ollama', 'gpt4all', 'demo', or 'auto'.")


def _create_gemini(api_key, temperature, model, max_output_tokens):
    """Create a Google Gemini LLM instance via LangChain."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model or "gemini-2.0-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _create_ollama(temperature, model, max_output_tokens):
    """Create an Ollama LLM instance via LangChain."""
    from langchain_ollama import ChatOllama
    chosen_model = model
    if not chosen_model:
        available = get_ollama_models()
        preferred = ["llama3.2", "llama3.1", "llama3", "codellama",
                     "mistral", "deepseek-coder", "qwen2.5-coder"]
        for pref in preferred:
            for avail in available:
                if pref in avail:
                    chosen_model = avail
                    break
            if chosen_model:
                break
        if not chosen_model:
            chosen_model = available[0] if available else "llama3.2"

    return ChatOllama(model=chosen_model, temperature=temperature,
                      num_predict=max_output_tokens)


# ─────────────────────────────────────────────────────────
# Embeddings Factory
# ─────────────────────────────────────────────────────────

def get_embeddings(backend: str = "auto", api_key: Optional[str] = None):
    """Get embeddings model for RAG pipeline (Gemini → Ollama → None)."""
    if backend in ("gemini", "auto") and api_key and len(api_key) > 10:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=api_key,
            ), "gemini"
        except Exception:
            pass

    if backend in ("ollama", "auto") and check_ollama_available():
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model="llama3.2"), "ollama"
        except Exception:
            pass

    return None, "none"


# ─────────────────────────────────────────────────────────
# Backend Status
# ─────────────────────────────────────────────────────────

def get_backend_status(api_key: Optional[str] = None) -> dict:
    """Get status of all available backends for UI display."""
    _placeholders = {"", "your_gemini_api_key_here", "your_api_key_here",
                     "your_key_here", "paste_your_key"}
    _key_valid = bool(api_key) and api_key.strip().lower() not in _placeholders

    status = {
        "gemini": {"available": _key_valid, "reason": "API key provided" if _key_valid else "No API key"},
        "ollama": {"available": False, "reason": "Not running"},
        "gpt4all": {"available": False, "reason": "Not installed"},
        "demo": {"available": True, "reason": "Always available"},
    }

    if check_ollama_available():
        models = get_ollama_models()
        status["ollama"]["available"] = True
        status["ollama"]["reason"] = f"{len(models)} model(s) available"
        status["ollama"]["models"] = models

    if check_gpt4all_available():
        local_models = get_gpt4all_local_models()
        status["gpt4all"]["available"] = True
        if local_models:
            status["gpt4all"]["reason"] = f"{len(local_models)} model(s) downloaded"
            status["gpt4all"]["models"] = local_models
        else:
            status["gpt4all"]["reason"] = "Installed (will download model on first use)"

    return status
