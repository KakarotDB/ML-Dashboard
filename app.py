import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

import registry
from pipeline import PreprocessingPipeline

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #f0f0f0;
    }

    .subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 300;
        color: #888;
        font-size: 0.95rem;
        margin-top: -10px;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4fc3f7;
        margin-bottom: 8px;
    }

    .report-card {
        background: #1a1a2e;
        border: 1px solid #2a2a3e;
        border-left: 3px solid #4fc3f7;
        border-radius: 4px;
        padding: 16px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #cdd6f4;
        line-height: 1.8;
    }

    .step-badge {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 3px;
        padding: 6px 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        margin-bottom: 6px;
    }

    .step-badge span {
        color: #4fc3f7;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }

    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 12px 16px;
    }

    .stButton > button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        border-radius: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "pipeline": None,
        "df_loaded": False,
        "dataset_name": "",
        "analysis_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SAMPLE_DATASETS: dict[str, str] = {
    "Titanic": "datasets/titanic.csv",
}

def load_dataframe(source: str | Path) -> pd.DataFrame:
    return pd.read_csv(source)

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()

def get_all_columns(df: pd.DataFrame) -> list[str]:
    return df.columns.tolist()

def render_report_card(report: dict) -> None:
    lines = "".join(
        f"<div><b>{k.replace('_', ' ').title()}</b>: {v}</div>"
        for k, v in report.items()
    )
    st.markdown(f'<div class="report-card">{lines}</div>', unsafe_allow_html=True)

def render_dataframe_stats(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Numeric Columns", len(get_numeric_columns(df)))

# ------------------------------------------------------------------
# Sidebar — Module Parameters
# ------------------------------------------------------------------

def render_module_params(module_name: str) -> dict:
    """
    Dynamically renders parameter inputs for the selected module
    and returns them as a kwargs dict ready to pass to pipeline.apply().

    Add a new elif block here when a new module requires custom params.
    """
    methods = PreprocessingPipeline.get_methods_for(module_name)
    method_label = st.selectbox("Method", list(methods.keys()))
    method_value = methods[method_label]

    params: dict = {"method": method_value}

    df = st.session_state.pipeline.current_df
    numeric_cols = get_numeric_columns(df)
    all_cols = get_all_columns(df)

    if module_name == "Discretization":
        params["column"] = st.selectbox("Column to Discretize", numeric_cols)

        if method_value == "entropy":
            params["class_column"] = st.selectbox("Class Column", all_cols)
            params["max_intervals"] = st.slider("Max Intervals", 2, 20, 5)
            params["gain_threshold"] = st.number_input(
                "Gain Threshold", min_value=0.0001, max_value=1.0,
                value=0.01, step=0.001, format="%.4f"
            )

    elif module_name == "Data Reduction":
        if method_value == "simple_random":
            params["sample_size"] = st.number_input(
                "Sample Size", min_value=1,
                max_value=len(df), value=min(1000, len(df)), step=50
            )
            params["replacement"] = st.checkbox("Sample with Replacement", value=False)

        elif method_value == "stratified":
            params["column"] = st.selectbox("Grouping Column", all_cols)
            params["sample_fraction"] = st.slider(
                "Sample Fraction", min_value=0.05, max_value=1.0, value=0.1, step=0.05
            )

        elif method_value == "feature_selection":
            params["variance_threshold"] = st.slider(
                "Variance Threshold", min_value=0.0, max_value=5.0,
                value=0.01, step=0.01,
                help="Columns with variance below this value will be dropped."
            )

    # ------------------------------------------------------------------
    # Add parameter blocks for new modules below as they are integrated:
    #
    # elif module_name == "Missing Values":
    #     params["column"] = st.selectbox("Column", all_cols)
    #     params["strategy"] = ...
    #
    # elif module_name == "Smoothing":
    #     ...
    # ------------------------------------------------------------------

    return params

# ------------------------------------------------------------------
# Main Layout
# ------------------------------------------------------------------

st.markdown('<div class="main-title">⚙ ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data Preprocessing & Machine Learning Pipeline</div>', unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------------------
# Zone 1: Data Loading
# ------------------------------------------------------------------

if not st.session_state.df_loaded:
    st.markdown('<div class="section-header">Load Dataset</div>', unsafe_allow_html=True)

    load_col1, load_col2 = st.columns([1, 1], gap="large")

    with load_col1:
        st.markdown("**Upload a CSV**")
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = load_dataframe(uploaded)
            st.session_state.pipeline = PreprocessingPipeline(df)
            st.session_state.df_loaded = True
            st.session_state.dataset_name = uploaded.name
            st.rerun()

    with load_col2:
        st.markdown("**Use a Sample Dataset**")
        sample_choice = st.selectbox("", list(SAMPLE_DATASETS.keys()), label_visibility="collapsed")
        if st.button("Load Sample", use_container_width=True):
            path = SAMPLE_DATASETS[sample_choice]
            if Path(path).exists():
                df = load_dataframe(path)
                st.session_state.pipeline = PreprocessingPipeline(df)
                st.session_state.df_loaded = True
                st.session_state.dataset_name = sample_choice
                st.rerun()
            else:
                st.error(f"Sample file not found at `{path}`.")

    st.stop()

# ------------------------------------------------------------------
# Zone 2: Sidebar — Pipeline Controls
# ------------------------------------------------------------------

pipeline: PreprocessingPipeline = st.session_state.pipeline

with st.sidebar:
    st.markdown(f"**Dataset:** `{st.session_state.dataset_name}`")
    st.caption(f"{pipeline.current_df.shape[0]} rows × {pipeline.current_df.shape[1]} cols")
    st.divider()

    st.markdown('<div class="section-header">Apply Technique</div>', unsafe_allow_html=True)

    registered = PreprocessingPipeline.get_registered_modules()
    selected_module = st.selectbox("Module", registered)

    st.markdown("")
    params = render_module_params(selected_module)

    st.markdown("")
    apply_btn = st.button("▶  Apply", use_container_width=True, type="primary")

    col_undo, col_reset = st.columns(2)
    undo_btn = col_undo.button("↩ Undo", use_container_width=True)
    reset_btn = col_reset.button("⟳ Reset", use_container_width=True)

    if apply_btn:
        try:
            pipeline.apply(selected_module, **params)
            st.success(f"{selected_module} applied.")
            st.rerun()
        except NotImplementedError:
            st.warning("This module is not yet integrated.")
        except Exception as e:
            st.error(str(e))

    if undo_btn and pipeline.history:
        pipeline.undo()
        st.rerun()

    if reset_btn:
        pipeline.reset()
        st.session_state.analysis_result = None
        st.rerun()

    st.divider()
    st.markdown('<div class="section-header">Step History</div>', unsafe_allow_html=True)

    history = pipeline.get_history_summary()
    if not history:
        st.caption("No steps applied yet.")
    else:
        for entry in reversed(history):
            st.markdown(
                f'<div class="step-badge">'
                f'<span>#{entry["step"]}</span> {entry["name"]} — {entry["method"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# Zone 3: Results Area
# ------------------------------------------------------------------

tabs = st.tabs(["CURRENT DATA", "BEFORE / AFTER", "VISUALIZE", "HISTORY"])

# -- Tab 1: Current DataFrame --
with tabs[0]:
    st.markdown('<div class="section-header">Current State</div>', unsafe_allow_html=True)
    render_dataframe_stats(pipeline.current_df)
    st.markdown("")
    st.dataframe(pipeline.current_df, use_container_width=True, height=420)

# -- Tab 2: Before / After --
with tabs[1]:
    if not pipeline.history:
        st.info("Apply a technique to see a before/after comparison.")
    else:
        last = pipeline.history[-1]
        st.markdown(
            f'<div class="section-header">Last Step — {last.step_name} ({last.method})</div>',
            unsafe_allow_html=True,
        )
        render_report_card(last.report)
        st.markdown("")

        b_col, a_col = st.columns(2, gap="medium")
        with b_col:
            st.caption("Before")
            st.dataframe(last.before, use_container_width=True, height=380)
        with a_col:
            st.caption("After")
            st.dataframe(last.after, use_container_width=True, height=380)

# -- Tab 3: Visualize --
with tabs[2]:
    st.markdown('<div class="section-header">Visualize Column</div>', unsafe_allow_html=True)
    numeric_cols = get_numeric_columns(pipeline.current_df)

    if not numeric_cols:
        st.info("No numeric columns available to visualize.")
    else:
        viz_col = st.selectbox("Select column", numeric_cols, key="viz_col")
        v1, v2 = st.columns(2, gap="medium")

        with v1:
            fig = px.histogram(
                pipeline.current_df, x=viz_col,
                title=f"Distribution — {viz_col}",
                template="plotly_dark",
                color_discrete_sequence=["#4fc3f7"],
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_family="IBM Plex Mono",
            )
            st.plotly_chart(fig, use_container_width=True)

        with v2:
            fig2 = px.box(
                pipeline.current_df, y=viz_col,
                title=f"Box Plot — {viz_col}",
                template="plotly_dark",
                color_discrete_sequence=["#4fc3f7"],
            )
            fig2.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_family="IBM Plex Mono",
            )
            st.plotly_chart(fig2, use_container_width=True)

        disc_col = f"{viz_col}_discretized"
        if disc_col in pipeline.current_df.columns:
            st.markdown('<div class="section-header">Discretized Intervals</div>', unsafe_allow_html=True)
            counts = pipeline.current_df[disc_col].value_counts().reset_index()
            counts.columns = ["interval", "count"]
            fig3 = px.bar(
                counts, x="interval", y="count",
                title=f"Interval Distribution — {disc_col}",
                template="plotly_dark",
                color_discrete_sequence=["#4fc3f7"],
            )
            fig3.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_family="IBM Plex Mono",
                xaxis_title="Interval", yaxis_title="Count",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # -- Analysis Section --
    analysis_modules = PreprocessingPipeline.get_registered_analysis_modules()
    if analysis_modules:
        st.divider()
        st.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)

        selected_analysis = st.selectbox("Analysis Module", analysis_modules, key="analysis_select")
        analysis_methods = PreprocessingPipeline.get_methods_for_analysis(selected_analysis)
        analysis_method_label = st.selectbox(
            "Method", list(analysis_methods.keys()), key="analysis_method"
        )

        analysis_params: dict = {"method": analysis_methods[analysis_method_label]}

        if selected_analysis == "Histogram":
            analysis_params["column"] = st.selectbox(
                "Column", get_numeric_columns(pipeline.current_df), key="hist_col"
            )
            analysis_params["num_buckets"] = st.slider(
                "Number of Buckets", 2, 50, 10, key="hist_buckets"
            )
            analysis_params["allow_negatives"] = st.checkbox(
                "Include Negative Values", value=True, key="hist_neg"
            )
        elif selected_analysis == "Similarity":
            selected_cols = st.multiselect(
                "Columns (leave empty for all numeric)", get_numeric_columns(pipeline.current_df), key="sim_cols"
            )
            if selected_cols:
                analysis_params["columns"] = selected_cols
            
            analysis_params["max_pairs"] = st.slider(
                "Max Pairs Limit", 100, 5000, 500, step=100, key="sim_max_pairs"
            )
            if analysis_params["method"] == "minkowski":
                analysis_params["p"] = st.slider(
                    "Minkowski power (p)", 1, 10, 3, key="sim_p"
                )

        if st.button("Run Analysis", key="run_analysis"):
            result = pipeline.analyze(selected_analysis, **analysis_params)
            st.session_state.analysis_result = result
            st.rerun()

        if st.session_state.analysis_result is not None:
            result = st.session_state.analysis_result
            st.dataframe(result, use_container_width=True)

            if selected_analysis == "Histogram":
                fig4 = px.bar(
                    result, x="Bucket_Range", y="Frequency_Count",
                    title=f"Histogram — {analysis_params.get('column', '')}",
                    template="plotly_dark",
                    color_discrete_sequence=["#4fc3f7"],
                )
                fig4.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_family="IBM Plex Mono",
                    xaxis_title="Bucket", yaxis_title="Count",
                )
                st.plotly_chart(fig4, use_container_width=True)
            elif selected_analysis == "Similarity":
                if not result.empty and "similarity_score" in result.columns:
                    fig_sim = px.histogram(
                        result, x="similarity_score",
                        title=f"Distribution of {analysis_params.get('method', '').title()} Scores",
                        template="plotly_dark",
                        color_discrete_sequence=["#4fc3f7"],
                    )
                    fig_sim.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font_family="IBM Plex Mono",
                        xaxis_title="Score", yaxis_title="Frequency",
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

# -- Tab 4: Full History --
with tabs[3]:
    st.markdown('<div class="section-header">Full Pipeline History</div>', unsafe_allow_html=True)

    if not pipeline.history:
        st.info("No steps applied yet.")
    else:
        for entry in pipeline.get_history_summary():
            with st.expander(f"Step {entry['step']} — {entry['name']} ({entry['method']})"):
                render_report_card(entry["report"])
                step = pipeline.get_step(entry["step"] - 1)
                b_col, a_col = st.columns(2)
                b_col.caption("Before")
                b_col.dataframe(step.before, use_container_width=True)
                a_col.caption("After")
                a_col.dataframe(step.after, use_container_width=True)
