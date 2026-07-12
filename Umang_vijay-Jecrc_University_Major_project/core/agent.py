"""
Agent — The agentic orchestrator that ties everything together.

Handles the full loop: prompt building → code generation → sandbox execution
→ RAG self-correction → output parsing.

Tracks token usage, latency, and cost for full observability.
"""

import re
import json
import uuid
import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

from core.llm_backend import get_llm, get_embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from core.sandbox import SandboxExecutor, ExecutionResult
from core.rag_pipeline import RAGPipeline
from core.schema_analyzer import SchemaAnalyzer
from core.use_cases import get_use_case_prompt


@dataclass
class TokenUsage:
    """Tracks LLM token consumption and timing."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    total_llm_time: float = 0.0
    total_sandbox_time: float = 0.0
    total_rag_time: float = 0.0
    total_wall_time: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate for Gemini 2.0 Flash (input $0.10 / 1M, output $0.40 / 1M)."""
        input_cost = (self.prompt_tokens / 1_000_000) * 0.10
        output_cost = (self.completion_tokens / 1_000_000) * 0.40
        return round(input_cost + output_cost, 6)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "total_llm_time_s": round(self.total_llm_time, 2),
            "total_sandbox_time_s": round(self.total_sandbox_time, 2),
            "total_rag_time_s": round(self.total_rag_time, 2),
            "total_wall_time_s": round(self.total_wall_time, 2),
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class AgentResult:
    """Final result from the agent's analysis pipeline."""
    success: bool
    code: str = ""
    insights: Dict = field(default_factory=dict)
    chart_paths: List[str] = field(default_factory=list)
    stdout: str = ""
    iterations: List[Dict] = field(default_factory=list)
    total_attempts: int = 0
    error: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    query: str = ""
    use_case: str = ""
    timestamp: str = ""


class DataScienceCoPilot:
    """
    The autonomous data science agent.

    Orchestrates: Query → Code Generation → Sandbox Execution → Self-Correction → Output.
    Full token/cost tracking for every LLM call.
    """

    SYSTEM_PROMPT = """You are an expert Python data analyst agent. Your job is to write
clean, correct, executable Python code using Pandas, NumPy, Matplotlib, Plotly,
scikit-learn, and statsmodels.

RULES:
1. Write ONLY executable Python code. No markdown fences, no explanations.
2. The DataFrame is pre-loaded as `df`. Do NOT re-read any files.
3. Save matplotlib charts using: save_chart(fig, "chart_name")
4. Save plotly charts using: save_plotly_chart(fig, "chart_name")
5. Report insights using: report_insights({"key": "value", ...})
6. Print important findings to stdout.
7. Handle missing values and edge cases gracefully.
8. Use professional chart styling with clear labels and titles.
9. NEVER use os.system, subprocess, eval, exec, or open().
10. If a column doesn't exist, pick the closest match from available columns."""

    def __init__(self, api_key: str = "", max_retries: int = 3,
                 temperature: float = 0.2, backend: str = "auto"):
        self.api_key = api_key
        self.max_retries = max_retries
        self.backend_name = "unknown"

        # Create LLM via multi-backend factory
        self.llm, self.backend_name = get_llm(
            backend=backend, api_key=api_key,
            temperature=temperature, max_output_tokens=4096,
        )

        self.sandbox = SandboxExecutor(timeout=30)

        # RAG pipeline with backend-aware embeddings
        embeddings, emb_backend = get_embeddings(backend=backend, api_key=api_key)
        self.rag = RAGPipeline(api_key=api_key, embeddings=embeddings)
        self.schema_analyzer = SchemaAnalyzer(
            api_key=api_key, llm=self.llm
        )

        # Build/load RAG index
        self._rag_ready = False

        # Cumulative session metrics
        self.session_total_tokens = 0
        self.session_total_cost = 0.0
        self.session_queries = 0

    def initialize_rag(self) -> bool:
        """Initialize the RAG pipeline (build or load FAISS index)."""
        self._rag_ready = self.rag.build_or_load_index()
        return self._rag_ready

    def analyze_schema(self, df: pd.DataFrame) -> dict:
        """Generate automatic schema summary for a dataset."""
        return self.schema_analyzer.analyze(df)

    def _call_llm(self, messages: list, usage: TokenUsage) -> str:
        """
        Call the LLM and track token usage from response metadata.
        Returns the text content of the response.
        """
        start = time.time()
        try:
            response = self.llm.invoke(messages)
            elapsed = time.time() - start

            usage.llm_calls += 1
            usage.total_llm_time += elapsed

            # Extract token usage from response metadata
            meta = getattr(response, "response_metadata", {}) or {}
            u = meta.get("usage_metadata", {}) or meta.get("token_usage", {})

            prompt_tok = u.get("prompt_token_count", 0) or u.get("input_tokens", 0) or u.get("prompt_tokens", 0)
            completion_tok = u.get("candidates_token_count", 0) or u.get("output_tokens", 0) or u.get("completion_tokens", 0)
            total_tok = u.get("total_token_count", 0) or (prompt_tok + completion_tok)

            usage.prompt_tokens += prompt_tok
            usage.completion_tokens += completion_tok
            usage.total_tokens += total_tok

            # Fallback estimation if API doesn't return token counts
            if total_tok == 0:
                prompt_text = " ".join(m.content for m in messages)
                est_prompt = len(prompt_text) // 4
                est_completion = len(response.content) // 4
                usage.prompt_tokens += est_prompt
                usage.completion_tokens += est_completion
                usage.total_tokens += est_prompt + est_completion

            return response.content
        except Exception as e:
            usage.total_llm_time += time.time() - start
            raise e

    def analyze(self, df: pd.DataFrame, query: str,
                use_case: str = "ad_hoc",
                progress_callback=None) -> AgentResult:
        """
        Run the full agentic analysis pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            The user's uploaded dataset.
        query : str
            The user's plain English question.
        use_case : str
            One of: sales_dashboard, data_quality, trend_analysis,
            cohort_analysis, predictive_forecast, ad_hoc.
        progress_callback : callable, optional
            Function to call with progress updates (for Streamlit UI).

        Returns
        -------
        AgentResult
            Contains code, charts, insights, execution log, and token usage.
        """
        wall_start = time.time()
        token_usage = TokenUsage()
        iterations = []
        exec_id = str(uuid.uuid4())[:8]

        # Step 1: Save DataFrame for sandbox access
        if progress_callback:
            progress_callback("📋 Preparing dataset for sandbox...")
        df_path = self.sandbox.save_dataframe(df, exec_id)

        # Step 2: Build schema context from real data
        if progress_callback:
            progress_callback("🔍 Analyzing dataset schema for prompt context...")
        schema_context = self._build_schema_context(df)

        # Step 3: Get use-case-specific prompt
        full_prompt = get_use_case_prompt(use_case, schema_context, query)

        # Step 4: Generate → Execute → Self-Correct loop
        current_code = None
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            if progress_callback:
                progress_callback(f"🤖 Attempt {attempt}/{self.max_retries}: Generating Python code via Gemini...")

            # Generate code (real-time LLM call with token tracking)
            llm_start = time.time()
            if attempt == 1:
                code = self._generate_code(full_prompt, token_usage)
            else:
                code = self._regenerate_with_fix(full_prompt, current_code, last_error, token_usage)
            llm_elapsed = time.time() - llm_start

            current_code = code

            if not code.strip():
                iterations.append({
                    "attempt": attempt, "success": False,
                    "error": "LLM returned empty code",
                    "llm_time": round(llm_elapsed, 2),
                })
                continue

            # Execute in sandbox (real subprocess execution)
            if progress_callback:
                progress_callback(f"⚡ Attempt {attempt}/{self.max_retries}: Executing generated code in sandbox...")

            sandbox_start = time.time()
            result = self.sandbox.execute(code, df_path)
            sandbox_elapsed = time.time() - sandbox_start
            token_usage.total_sandbox_time += sandbox_elapsed

            if result.success:
                iterations.append({
                    "attempt": attempt, "success": True,
                    "execution_time": result.execution_time,
                    "llm_time": round(llm_elapsed, 2),
                    "sandbox_time": round(sandbox_elapsed, 2),
                })

                # Parse output from real execution
                insights = self._parse_insights(result.stdout)

                token_usage.total_wall_time = round(time.time() - wall_start, 2)

                # Update session totals
                self.session_total_tokens += token_usage.total_tokens
                self.session_total_cost += token_usage.estimated_cost_usd
                self.session_queries += 1

                return AgentResult(
                    success=True,
                    code=code,
                    insights=insights,
                    chart_paths=result.chart_paths,
                    stdout=result.stdout,
                    iterations=iterations,
                    total_attempts=attempt,
                    token_usage=token_usage,
                    query=query,
                    use_case=use_case,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                last_error = result.stderr
                fix_context = ""

                # RAG self-correction (real vector search)
                if self._rag_ready and attempt < self.max_retries:
                    if progress_callback:
                        progress_callback(f"🔄 Attempt {attempt}: Querying RAG pipeline for error fix...")
                    rag_start = time.time()
                    fix_context = self.rag.query_for_fix(result.stderr, code)
                    token_usage.total_rag_time += time.time() - rag_start

                iterations.append({
                    "attempt": attempt, "success": False,
                    "error": result.stderr[:500],
                    "fix_applied": "RAG context retrieved" if fix_context else "No RAG context",
                    "llm_time": round(llm_elapsed, 2),
                    "sandbox_time": round(sandbox_elapsed, 2),
                })

        # All retries exhausted
        token_usage.total_wall_time = round(time.time() - wall_start, 2)
        self.session_total_tokens += token_usage.total_tokens
        self.session_total_cost += token_usage.estimated_cost_usd
        self.session_queries += 1

        return AgentResult(
            success=False,
            code=current_code or "",
            iterations=iterations,
            total_attempts=self.max_retries,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error[:300]}",
            token_usage=token_usage,
            query=query,
            use_case=use_case,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _generate_code(self, prompt: str, usage: TokenUsage) -> str:
        """Generate Python code from the prompt using Gemini with token tracking."""
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            content = self._call_llm(messages, usage)
            return self._extract_code(content)
        except Exception as e:
            return f"# Code generation error: {e}"

    def _regenerate_with_fix(self, original_prompt: str, failed_code: str,
                             error: str, usage: TokenUsage) -> str:
        """Regenerate code using error context and RAG documentation."""
        rag_context = ""
        if self._rag_ready:
            rag_context = self.rag.query_for_fix(error, failed_code)

        fix_prompt = f"""{original_prompt}

PREVIOUS ATTEMPT FAILED. Here is the code that failed:
```python
{failed_code}
```

ERROR MESSAGE:
{error[:500]}

RELEVANT DOCUMENTATION (from RAG):
{rag_context[:2000]}

Please fix the code based on the error and documentation above.
Write the COMPLETE corrected code. CODE ONLY, no explanations."""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=fix_prompt),
            ]
            content = self._call_llm(messages, usage)
            return self._extract_code(content)
        except Exception as e:
            return f"# Code regeneration error: {e}"

    def _extract_code(self, response_text: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences."""
        text = response_text.strip()

        # Detect error comments from failed LLM calls
        error_prefixes = ["# Code generation error:", "# Code regeneration error:"]
        for prefix in error_prefixes:
            if text.startswith(prefix):
                return ""  # Return empty so agent treats this as failure

        # Remove markdown code fences
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no fences, check if it looks like code
        if any(kw in text for kw in ["import ", "df.", "plt.", "fig", "print("]):
            lines = text.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if (line.strip().startswith(("import ", "from ", "#", "df", "fig",
                    "plt", "sns", "pd", "np", "data", "result", "chart",
                    "for ", "if ", "def ", "class ", "try:", "with "))
                    or in_code):
                    code_lines.append(line)
                    in_code = True
                elif in_code and (line.strip() == "" or line.startswith(" ")):
                    code_lines.append(line)
            return "\n".join(code_lines).strip()

        return text

    def _build_schema_context(self, df: pd.DataFrame) -> str:
        """Build a concise schema description for the prompt."""
        lines = [
            f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns and types:",
        ]
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            missing = df[col].isnull().sum()
            sample = df[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({dtype}): {nunique} unique, {missing} missing, sample: {sample}")

        lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
        return "\n".join(lines)

    def _parse_insights(self, stdout: str) -> dict:
        """Parse structured insights from sandbox stdout."""
        insights = {}
        for line in stdout.split("\n"):
            if line.startswith("INSIGHTS_JSON:"):
                try:
                    json_str = line.replace("INSIGHTS_JSON:", "", 1)
                    insights.update(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    pass
        # Also capture plain text output
        plain_lines = [
            l for l in stdout.split("\n")
            if not l.startswith(("CHART_SAVED:", "PLOTLY_CHART_SAVED:", "INSIGHTS_JSON:"))
            and l.strip()
        ]
        if plain_lines:
            insights["_raw_output"] = "\n".join(plain_lines)
        return insights

    def get_session_stats(self) -> dict:
        """Get cumulative session-level statistics."""
        return {
            "total_queries": self.session_queries,
            "total_tokens": self.session_total_tokens,
            "total_cost_usd": round(self.session_total_cost, 6),
        }

    def cleanup(self):
        """Clean up temporary sandbox files."""
        self.sandbox.cleanup()
