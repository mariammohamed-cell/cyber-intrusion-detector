from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


APP_DIR = Path(__file__).resolve().parent
RAW_FEATURES = [
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "encryption_used",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access",
]
ARTIFACT_FILES = {
    "model": "rf_model.pkl",
    "scaler": "scaler.pkl",
    "columns": "model_columns.pkl",
    "encryption": "encryption_mode.pkl",
}
DEFAULT_DATASET = "cybersecurity_intrusion_data.csv"


st.set_page_config(
    page_title="CyberShield IDS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #060912;
            --bg-1: #0a1020;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-strong: rgba(17, 34, 64, 0.94);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5edf7;
            --muted: #94a3b8;
            --cyan: #22d3ee;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
            --violet: #a78bfa;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 8%, rgba(34, 211, 238, 0.16), transparent 28rem),
                radial-gradient(circle at 90% 5%, rgba(167, 139, 250, 0.12), transparent 24rem),
                linear-gradient(145deg, var(--bg-0), var(--bg-1) 48%, #050816);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(3, 7, 18, 0.98), rgba(15, 23, 42, 0.96));
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--text);
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(34, 211, 238, 0.12), rgba(15, 23, 42, 0.7)),
                repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.06) 0 1px, transparent 1px 42px);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3.8rem);
            line-height: 1;
            font-weight: 700;
            margin: 0;
            color: #f8fafc;
        }

        .hero-subtitle {
            margin-top: 0.75rem;
            color: var(--muted);
            max-width: 760px;
            font-size: 1rem;
        }

        .metric-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--panel);
            min-height: 118px;
        }

        .metric-card .label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            margin-bottom: 0.55rem;
        }

        .metric-card .value {
            color: #f8fafc;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .metric-card .note {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.55rem;
        }

        .metric-card.green { border-top: 3px solid var(--green); }
        .metric-card.red { border-top: 3px solid var(--red); }
        .metric-card.amber { border-top: 3px solid var(--amber); }
        .metric-card.cyan { border-top: 3px solid var(--cyan); }
        .metric-card.violet { border-top: 3px solid var(--violet); }
        .metric-card.low { border-top: 3px solid var(--green); }
        .metric-card.medium { border-top: 3px solid var(--amber); }
        .metric-card.high { border-top: 3px solid #f97316; }
        .metric-card.critical { border-top: 3px solid var(--red); }

        .section-panel {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.62);
            padding: 1.05rem;
            margin-bottom: 1rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid var(--border);
            background: rgba(148, 163, 184, 0.08);
            color: var(--text);
        }

        .status-pill.low { color: #bbf7d0; border-color: rgba(34, 197, 94, 0.35); background: rgba(34, 197, 94, 0.12); }
        .status-pill.medium { color: #fde68a; border-color: rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.12); }
        .status-pill.high { color: #fed7aa; border-color: rgba(249, 115, 22, 0.4); background: rgba(249, 115, 22, 0.12); }
        .status-pill.critical { color: #fecaca; border-color: rgba(239, 68, 68, 0.45); background: rgba(239, 68, 68, 0.14); }
        .status-pill.ready { color: #a7f3d0; border-color: rgba(20, 184, 166, 0.4); background: rgba(20, 184, 166, 0.12); }
        .status-pill.missing { color: #fecaca; border-color: rgba(239, 68, 68, 0.45); background: rgba(239, 68, 68, 0.12); }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            border: 1px solid rgba(34, 211, 238, 0.35);
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.25), rgba(167, 139, 250, 0.2));
            color: #f8fafc;
            font-weight: 700;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: rgba(34, 211, 238, 0.75);
            color: #ffffff;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .sidebar-brand {
            padding: 0.85rem 0.2rem 1rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
        }

        .sidebar-brand .name {
            font-size: 1.15rem;
            font-weight: 800;
            color: #f8fafc;
        }

        .sidebar-brand .tagline {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def escape_text(value: Any) -> str:
    return html.escape(str(value))


def find_file(filename: str) -> Path | None:
    candidates = [
        APP_DIR / filename,
        Path.cwd() / filename,
        Path.home() / "Downloads" / filename,
        Path.home() / "Desktop" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[dict[str, Any] | None, dict[str, Path | None], list[str], str | None]:
    paths = {key: find_file(filename) for key, filename in ARTIFACT_FILES.items()}
    missing = [ARTIFACT_FILES[key] for key, path in paths.items() if path is None]

    if joblib is None:
        return None, paths, missing, "The joblib package is not installed."

    if missing:
        return None, paths, missing, None

    try:
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        model_columns = list(joblib.load(paths["columns"]))
        encryption_mode = joblib.load(paths["encryption"])
        return (
            {
                "model": model,
                "scaler": scaler,
                "model_columns": model_columns,
                "encryption_mode": encryption_mode,
            },
            paths,
            [],
            None,
        )
    except Exception as exc:
        return None, paths, [], str(exc)


@st.cache_data(show_spinner=False)
def load_default_dataset() -> tuple[pd.DataFrame | None, str | None]:
    path = find_file(DEFAULT_DATASET)
    if path is None:
        return None, None
    return pd.read_csv(path), str(path)


def metric_card(label: str, value: str, note: str = "", tone: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="metric-card {escape_text(tone)}">
            <div class="label">{escape_text(label)}</div>
            <div class="value">{escape_text(value)}</div>
            <div class="note">{escape_text(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, level: str) -> str:
    return f'<span class="status-pill {escape_text(level)}">{escape_text(label)}</span>'


def risk_level(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Critical", "critical"
    if score >= 65:
        return "High", "high"
    if score >= 35:
        return "Medium", "medium"
    return "Low", "low"


def normalize_encryption_mode(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)) and len(value) > 0:
        return str(value[0])
    return "AES"


def validate_raw_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in RAW_FEATURES if column not in df.columns]


def prepare_features(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    missing = validate_raw_columns(df)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required columns: {joined}")

    prepared = df[RAW_FEATURES].copy()
    numeric_columns = [
        "network_packet_size",
        "login_attempts",
        "session_duration",
        "ip_reputation_score",
        "failed_logins",
        "unusual_time_access",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["network_packet_size"] = prepared["network_packet_size"].fillna(prepared["network_packet_size"].median())
    prepared["login_attempts"] = prepared["login_attempts"].fillna(1).clip(lower=1)
    prepared["session_duration"] = prepared["session_duration"].fillna(1).clip(lower=0.5)
    prepared["ip_reputation_score"] = prepared["ip_reputation_score"].fillna(0.5).clip(lower=0, upper=1)
    prepared["failed_logins"] = prepared["failed_logins"].fillna(0).clip(lower=0)
    prepared["unusual_time_access"] = prepared["unusual_time_access"].fillna(0).clip(lower=0, upper=1).astype(int)

    default_encryption = normalize_encryption_mode(artifacts["encryption_mode"])
    prepared["encryption_used"] = prepared["encryption_used"].replace("", np.nan).fillna(default_encryption)
    prepared["protocol_type"] = prepared["protocol_type"].fillna("TCP")
    prepared["browser_type"] = prepared["browser_type"].fillna("Unknown")

    prepared["packet_per_duration"] = prepared["network_packet_size"] / prepared["session_duration"]
    prepared["failed_login_ratio"] = prepared["failed_logins"] / prepared["login_attempts"]

    encoded = pd.get_dummies(prepared, drop_first=True)
    encoded = encoded.reindex(columns=artifacts["model_columns"], fill_value=0)

    if "session_duration" in encoded.columns:
        encoded["session_duration"] = np.log1p(encoded["session_duration"])

    scaled = artifacts["scaler"].transform(encoded)
    return pd.DataFrame(scaled, columns=artifacts["model_columns"])


def predict_dataframe(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    features = prepare_features(df, artifacts)
    model = artifacts["model"]
    predictions = model.predict(features)

    attack_scores = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            attack_index = classes.index(1)
        elif "1" in classes:
            attack_index = classes.index("1")
        else:
            attack_index = min(1, probabilities.shape[1] - 1)
        attack_scores = probabilities[:, attack_index] * 100

    if attack_scores is None:
        attack_scores = np.where(np.asarray(predictions).astype(int) == 1, 75.0, 15.0)

    result = df.copy()
    prediction_int = np.asarray(predictions).astype(int)
    result["prediction"] = np.where(prediction_int == 1, "Attack", "Normal")
    result["risk_score"] = np.round(attack_scores, 2)
    levels = [risk_level(float(score))[0] for score in result["risk_score"]]
    result["risk_level"] = levels
    return result


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">CyberShield IDS</div>
            <div class="hero-subtitle">
                Intrusion detection dashboard for traffic inspection, attack scoring, and model-driven session analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(paths: dict[str, Path | None], missing: list[str], load_error: str | None) -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="name">CyberShield IDS</div>
            <div class="tagline">Traffic intelligence console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if load_error:
        st.sidebar.markdown(status_pill("Artifact error", "missing"), unsafe_allow_html=True)
        st.sidebar.caption(load_error)
    elif missing:
        st.sidebar.markdown(status_pill("Artifacts missing", "missing"), unsafe_allow_html=True)
        st.sidebar.caption(", ".join(missing))
    else:
        st.sidebar.markdown(status_pill("Model ready", "ready"), unsafe_allow_html=True)

    with st.sidebar.expander("Artifact paths", expanded=False):
        for key, filename in ARTIFACT_FILES.items():
            path = paths.get(key)
            st.caption(f"{filename}: {path if path else 'Not found'}")

    return st.sidebar.radio(
        "Navigation",
        ["Overview", "Prediction Lab", "Batch Analysis", "Model Insights", "Data Explorer"],
        label_visibility="visible",
    )


def altair_bar(data: pd.DataFrame, x: str, y: str, color: str | None = None, height: int = 280) -> None:
    if alt is None:
        st.bar_chart(data.set_index(x)[y])
        return

    encoding = {
        "x": alt.X(f"{x}:N", title=None, sort="-y"),
        "y": alt.Y(f"{y}:Q", title=None),
        "tooltip": [alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.2f")],
    }
    if color:
        encoding["color"] = alt.Color(
            f"{color}:N",
            scale=alt.Scale(range=["#22c55e", "#ef4444", "#22d3ee", "#f59e0b", "#a78bfa"]),
            legend=None,
        )

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(**encoding)
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def altair_donut(data: pd.DataFrame, label_col: str, value_col: str, height: int = 280) -> None:
    if alt is None:
        st.bar_chart(data.set_index(label_col)[value_col])
        return

    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=72, outerRadius=118)
        .encode(
            theta=alt.Theta(f"{value_col}:Q"),
            color=alt.Color(
                f"{label_col}:N",
                scale=alt.Scale(range=["#22c55e", "#ef4444", "#22d3ee", "#f59e0b"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip(f"{label_col}:N"), alt.Tooltip(f"{value_col}:Q", format=",")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def altair_line(data: pd.DataFrame, x: str, y: str, height: int = 280) -> None:
    if alt is None:
        st.line_chart(data.set_index(x)[y])
        return

    chart = (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2.5, color="#22d3ee")
        .encode(
            x=alt.X(f"{x}:O", title="Session batch"),
            y=alt.Y(f"{y}:Q", title="Attack rate", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip(f"{x}:O", title="Batch"), alt.Tooltip(f"{y}:Q", title="Attack rate", format=".1%")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def dataset_with_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    if "attack_detected" in labeled.columns:
        labeled["traffic_label"] = np.where(labeled["attack_detected"].astype(int) == 1, "Attack", "Normal")
    return labeled


def overview_page(df: pd.DataFrame | None) -> None:
    st.subheader("Operations Overview")
    if df is None:
        st.info("Default dataset was not found. Upload a CSV in Batch Analysis to inspect traffic.")
        return

    labeled = dataset_with_labels(df)
    total = len(labeled)
    attacks = int(labeled["attack_detected"].sum()) if "attack_detected" in labeled.columns else 0
    normal = total - attacks
    attack_rate = attacks / total if total else 0
    missing_encryption = int(labeled["encryption_used"].isna().sum()) if "encryption_used" in labeled.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Sessions", f"{total:,}", "Rows available for inspection", "cyan")
    with col2:
        metric_card("Detected Attacks", f"{attacks:,}", f"{attack_rate:.1%} attack rate", "red")
    with col3:
        metric_card("Normal Traffic", f"{normal:,}", "Baseline sessions", "green")
    with col4:
        metric_card("Missing Encryption", f"{missing_encryption:,}", "Filled during preprocessing", "amber")

    chart_col1, chart_col2 = st.columns((1, 1))
    with chart_col1:
        st.markdown("#### Traffic Split")
        split = labeled["traffic_label"].value_counts().rename_axis("label").reset_index(name="sessions")
        altair_donut(split, "label", "sessions")

    with chart_col2:
        st.markdown("#### Attack Rate by Protocol")
        protocol = (
            labeled.groupby("protocol_type", dropna=False)
            .agg(sessions=("attack_detected", "size"), attack_rate=("attack_detected", "mean"))
            .reset_index()
        )
        protocol["attack_rate"] = protocol["attack_rate"] * 100
        altair_bar(protocol, "protocol_type", "attack_rate", "protocol_type")

    st.markdown("#### Activity Trend by Session Batch")
    trend = labeled.reset_index(names="row_id")
    trend["batch"] = (trend["row_id"] // 250) + 1
    trend = trend.groupby("batch", as_index=False)["attack_detected"].mean()
    altair_line(trend, "batch", "attack_detected")


def prediction_page(artifacts: dict[str, Any] | None) -> None:
    st.subheader("Prediction Lab")
    st.markdown('<div class="small-muted">Score one network session using the trained Random Forest artifact.</div>', unsafe_allow_html=True)

    if artifacts is None:
        st.warning("Model artifacts are not available. Add the .pkl files next to app.py or keep them in Downloads.")

    with st.form("single_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            network_packet_size = st.number_input("Network packet size", min_value=1, max_value=5000, value=500, step=10)
            protocol_type = st.selectbox("Protocol type", ["TCP", "UDP", "ICMP"])
            login_attempts = st.number_input("Login attempts", min_value=1, max_value=50, value=4, step=1)
        with middle:
            session_duration = st.number_input("Session duration", min_value=0.5, max_value=10000.0, value=556.0, step=10.0)
            encryption_option = st.selectbox("Encryption used", ["AES", "DES", "Not provided"])
            ip_reputation_score = st.slider("IP reputation score", min_value=0.0, max_value=1.0, value=0.31, step=0.01)
        with right:
            failed_logins = st.number_input("Failed logins", min_value=0, max_value=50, value=1, step=1)
            browser_type = st.selectbox("Browser type", ["Chrome", "Firefox", "Edge", "Safari", "Unknown"])
            unusual_time_access = st.selectbox("Unusual time access", ["No", "Yes"])

        submitted = st.form_submit_button("Analyze Session", use_container_width=True)

    if submitted and artifacts is not None:
        raw = pd.DataFrame(
            [
                {
                    "network_packet_size": network_packet_size,
                    "protocol_type": protocol_type,
                    "login_attempts": login_attempts,
                    "session_duration": session_duration,
                    "encryption_used": "" if encryption_option == "Not provided" else encryption_option,
                    "ip_reputation_score": ip_reputation_score,
                    "failed_logins": failed_logins,
                    "browser_type": browser_type,
                    "unusual_time_access": 1 if unusual_time_access == "Yes" else 0,
                }
            ]
        )
        try:
            result = predict_dataframe(raw, artifacts)
            row = result.iloc[0]
            level, css_level = risk_level(float(row["risk_score"]))
            verdict = "Intrusion Detected" if row["prediction"] == "Attack" else "Normal Traffic"
            tone = "red" if row["prediction"] == "Attack" else "green"

            st.markdown("#### Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Verdict", verdict, "Random Forest prediction", tone)
            with col2:
                metric_card("Risk Score", f"{row['risk_score']:.2f}%", "Attack probability estimate", css_level)
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card {css_level}">
                        <div class="label">Risk Level</div>
                        <div class="value">{escape_text(level)}</div>
                        <div class="note">{status_pill(level, css_level)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def batch_page(df: pd.DataFrame | None, artifacts: dict[str, Any] | None) -> None:
    st.subheader("Batch Analysis")
    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
    source_name = "Uploaded CSV" if uploaded is not None else "Default dataset"

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
    else:
        batch_df = df

    if batch_df is None:
        st.info("No dataset is available. Upload a CSV with the required traffic columns.")
        return

    st.caption(f"Source: {source_name} | Rows: {len(batch_df):,}")
    missing = validate_raw_columns(batch_df)
    if missing:
        st.error("The selected CSV is missing required columns: " + ", ".join(missing))
        return

    preview_cols = [column for column in ["session_id", *RAW_FEATURES, "attack_detected"] if column in batch_df.columns]
    st.dataframe(batch_df[preview_cols].head(30), use_container_width=True, hide_index=True)

    if artifacts is None:
        st.warning("Model artifacts are not available. Batch predictions cannot run yet.")
        return

    if st.button("Run Batch Detection", use_container_width=True):
        try:
            results = predict_dataframe(batch_df, artifacts)
            attack_count = int((results["prediction"] == "Attack").sum())
            avg_risk = float(results["risk_score"].mean())
            critical_count = int((results["risk_level"] == "Critical").sum())

            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Predicted Attacks", f"{attack_count:,}", "Sessions flagged by the model", "red")
            with col2:
                metric_card("Average Risk", f"{avg_risk:.2f}%", "Mean attack probability", "amber")
            with col3:
                metric_card("Critical Sessions", f"{critical_count:,}", "Risk score at or above 85%", "violet")

            st.markdown("#### Prediction Distribution")
            distribution = results["prediction"].value_counts().rename_axis("prediction").reset_index(name="sessions")
            altair_donut(distribution, "prediction", "sessions")

            st.markdown("#### Highest Risk Sessions")
            ordered = results.sort_values("risk_score", ascending=False)
            display_cols = [column for column in ["session_id", "prediction", "risk_score", "risk_level", *RAW_FEATURES] if column in ordered.columns]
            st.dataframe(ordered[display_cols].head(100), use_container_width=True, hide_index=True)

            st.download_button(
                "Download Predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="cybershield_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Batch detection failed: {exc}")


def model_insights_page(df: pd.DataFrame | None, artifacts: dict[str, Any] | None) -> None:
    st.subheader("Model Insights")

    benchmark = pd.DataFrame(
        [
            {"Model": "Random Forest", "Accuracy": 0.885744, "Precision": 1.000000, "Recall": 0.744431, "F1 Score": 0.853495},
            {"Model": "SVM", "Accuracy": 0.867925, "Precision": 0.953243, "Recall": 0.740914, "F1 Score": 0.833773},
            {"Model": "Decision Tree", "Accuracy": 0.884172, "Precision": 1.000000, "Recall": 0.740914, "F1 Score": 0.851178},
            {"Model": "Logistic Regression", "Accuracy": 0.758910, "Precision": 0.756863, "Recall": 0.678781, "F1 Score": 0.715698},
            {"Model": "Naive Bayes", "Accuracy": 0.812893, "Precision": 0.888715, "Recall": 0.664713, "F1 Score": 0.760563},
            {"Model": "KNN", "Accuracy": 0.794025, "Precision": 0.891156, "Recall": 0.614302, "F1 Score": 0.727273},
        ]
    )

    col1, col2, col3, col4 = st.columns(4)
    rf_row = benchmark.iloc[0]
    with col1:
        metric_card("RF Accuracy", f"{rf_row['Accuracy']:.2%}", "Notebook test split", "cyan")
    with col2:
        metric_card("RF Precision", f"{rf_row['Precision']:.2%}", "Attack class", "green")
    with col3:
        metric_card("RF Recall", f"{rf_row['Recall']:.2%}", "Attack class", "amber")
    with col4:
        metric_card("RF F1 Score", f"{rf_row['F1 Score']:.2%}", "Attack class", "violet")

    st.markdown("#### Notebook Benchmark")
    st.dataframe(
        benchmark.style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    if artifacts is not None:
        model = artifacts["model"]
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame(
                {
                    "feature": artifacts["model_columns"],
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            st.markdown("#### Feature Importance")
            top_importance = importance.head(12).copy()
            altair_bar(top_importance, "feature", "importance", "feature", height=320)

        if df is not None and "attack_detected" in df.columns:
            st.markdown("#### Artifact Check on Available Dataset")
            try:
                results = predict_dataframe(df, artifacts)
                actual = df["attack_detected"].astype(int)
                predicted = (results["prediction"] == "Attack").astype(int)
                accuracy = float((actual == predicted).mean())
                attack_recall = float(((actual == 1) & (predicted == 1)).sum() / max((actual == 1).sum(), 1))
                false_alerts = int(((actual == 0) & (predicted == 1)).sum())

                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("Dataset Agreement", f"{accuracy:.2%}", "Current artifact vs available CSV", "cyan")
                with c2:
                    metric_card("Attack Recall", f"{attack_recall:.2%}", "Detected attack rows", "amber")
                with c3:
                    metric_card("False Alerts", f"{false_alerts:,}", "Normal rows flagged as attack", "red")
            except Exception as exc:
                st.warning(f"Artifact check could not run: {exc}")

    if artifacts is not None:
        with st.expander("Model Columns", expanded=False):
            st.dataframe(pd.DataFrame({"column": artifacts["model_columns"]}), use_container_width=True, hide_index=True)


def data_explorer_page(df: pd.DataFrame | None) -> None:
    st.subheader("Data Explorer")
    if df is None:
        st.info("Default dataset was not found.")
        return

    labeled = dataset_with_labels(df)
    filters = st.columns(3)
    with filters[0]:
        protocol_options = ["All"] + sorted(labeled["protocol_type"].dropna().unique().tolist())
        selected_protocol = st.selectbox("Protocol", protocol_options)
    with filters[1]:
        browser_options = ["All"] + sorted(labeled["browser_type"].dropna().unique().tolist())
        selected_browser = st.selectbox("Browser", browser_options)
    with filters[2]:
        label_options = ["All", "Normal", "Attack"]
        selected_label = st.selectbox("Traffic label", label_options)

    filtered = labeled.copy()
    if selected_protocol != "All":
        filtered = filtered[filtered["protocol_type"] == selected_protocol]
    if selected_browser != "All":
        filtered = filtered[filtered["browser_type"] == selected_browser]
    if selected_label != "All" and "traffic_label" in filtered.columns:
        filtered = filtered[filtered["traffic_label"] == selected_label]

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Filtered Rows", f"{len(filtered):,}", "Rows after current filters", "cyan")
    with col2:
        if "attack_detected" in filtered.columns and len(filtered) > 0:
            metric_card("Attack Rate", f"{filtered['attack_detected'].mean():.2%}", "Filtered traffic", "red")
        else:
            metric_card("Attack Rate", "N/A", "No labels available", "amber")
    with col3:
        median_duration = float(filtered["session_duration"].median()) if len(filtered) else 0.0
        metric_card("Median Duration", f"{median_duration:.1f}", "Session duration", "violet")

    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    artifacts, paths, missing, load_error = load_artifacts()
    default_df, dataset_path = load_default_dataset()
    page = render_sidebar(paths, missing, load_error)
    render_hero()

    if dataset_path:
        st.caption(f"Dataset loaded from: {dataset_path}")

    if page == "Overview":
        overview_page(default_df)
    elif page == "Prediction Lab":
        prediction_page(artifacts)
    elif page == "Batch Analysis":
        batch_page(default_df, artifacts)
    elif page == "Model Insights":
        model_insights_page(default_df, artifacts)
    elif page == "Data Explorer":
        data_explorer_page(default_df)


if __name__ == "__main__":
    main()
