# frontend.py
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Synthesizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0e0e0e; }

    h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #e8e8e8;
        letter-spacing: -0.03em;
    }

    .pipeline-bar {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 1.5rem;
    }

    .pipeline-bar .active { color: #7fffb2; font-weight: 600; }
    .pipeline-bar .sep { color: #333; }

    .fact-chip {
        display: inline-block;
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 0.35rem 0.6rem;
        font-size: 0.82rem;
        color: #ccc;
        margin: 0.2rem;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .report-box {
        background: #111;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #7fffb2;
        border-radius: 6px;
        padding: 1.5rem 2rem;
        color: #ddd;
        line-height: 1.75;
    }

    .subq-item {
        padding: 0.4rem 0;
        color: #aaa;
        font-size: 0.9rem;
        border-bottom: 1px solid #1e1e1e;
    }

    .metric-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }

    .metric-card .number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #7fffb2;
        font-weight: 600;
    }

    .metric-card .label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stTextInput"] > div > input {
        background: #141414 !important;
        border: 1px solid #2a2a2a !important;
        color: #eee !important;
        font-family: 'IBM Plex Mono', monospace !important;
        border-radius: 6px !important;
    }

    div[data-testid="stButton"] > button {
        background: #7fffb2 !important;
        color: #0e0e0e !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: #5de89a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 Research Synthesizer")
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:-0.5rem;'>"
    "4-agent pipeline · Planner → Search → Critic → Writer"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="query",
        placeholder="e.g. What are the latest advances in Retrieval-Augmented Generation?",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("Run →", use_container_width=True)

# ── Pipeline progress display ──────────────────────────────────────────────────
def show_pipeline(active_step: str):
    steps = ["PLANNER", "SEARCH", "CRITIC", "WRITER"]
    parts = []
    for s in steps:
        cls = "active" if s == active_step else ""
        parts.append(f'<span class="{cls}">{s}</span>')
    bar = ' <span class="sep">→</span> '.join(parts)
    st.markdown(f'<div class="pipeline-bar">{bar}</div>', unsafe_allow_html=True)


# ── Health check helper ────────────────────────────────────────────────────────
def api_is_up() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Main flow ──────────────────────────────────────────────────────────────────
if run and query.strip():

    if not api_is_up():
        st.error(
            "⚠️ Cannot reach the API server at `localhost:8000`. "
            "Make sure you ran `uvicorn api:app --reload` in another terminal."
        )
        st.stop()

    with st.spinner(""):
        # Show animated pipeline steps while waiting
        progress_placeholder = st.empty()

        # Fake step animation while the real request runs in background
        # (we flash each agent name every ~1s until response arrives)
        steps = ["PLANNER", "SEARCH", "CRITIC", "WRITER"]
        step_idx = 0

        import threading

        result_container = {"data": None, "error": None}

        def call_api():
            try:
                resp = requests.post(
                    f"{API_URL}/research",
                    json={"query": query},
                    timeout=180,
                )
                resp.raise_for_status()
                result_container["data"] = resp.json()
            except requests.exceptions.HTTPError as e:
                result_container["error"] = f"API error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                result_container["error"] = str(e)

        thread = threading.Thread(target=call_api)
        thread.start()

        while thread.is_alive():
            active = steps[step_idx % len(steps)]
            with progress_placeholder.container():
                show_pipeline(active)
            step_idx += 1
            time.sleep(1.2)

        thread.join()
        progress_placeholder.empty()

    if result_container["error"]:
        st.error(f"Research failed: {result_container['error']}")
        st.stop()

    data = result_container["data"]

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="number">{len(data["sub_questions"])}</div>'
            f'<div class="label">Sub-questions</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="number">{len(data["verified_facts"])}</div>'
            f'<div class="label">Verified facts</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with m3:
        word_count = len(data["final_report"].split())
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="number">{word_count}</div>'
            f'<div class="label">Report words</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout for sub-questions + facts ───────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 🗂 Sub-questions")
        for i, q in enumerate(data["sub_questions"], 1):
            st.markdown(
                f'<div class="subq-item"><span style="color:#555; font-family:\'IBM Plex Mono\',monospace">{i:02d}</span>&nbsp;&nbsp;{q}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### ✅ Verified facts")
        for fact in data["verified_facts"]:
            st.markdown(f'<div class="fact-chip">· {fact}</div>', unsafe_allow_html=True)

    # ── Final report ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📄 Research Report")
    st.markdown(
        f'<div class="report-box">{data["final_report"].replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # ── Download button ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"# Research Report\n\n**Query:** {data['query']}\n\n"
    report_text += "## Sub-questions\n" + "\n".join(f"- {q}" for q in data["sub_questions"])
    report_text += "\n\n## Verified Facts\n" + "\n".join(f"- {f}" for f in data["verified_facts"])
    report_text += f"\n\n## Report\n{data['final_report']}"

    st.download_button(
        label="⬇ Download report (.md)",
        data=report_text,
        file_name="research_report.md",
        mime="text/markdown",
    )

elif run and not query.strip():
    st.warning("Please enter a research question first.")