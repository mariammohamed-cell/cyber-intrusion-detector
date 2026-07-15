import streamlit as st
import joblib
from preprocess import preprocess_input

st.set_page_config(page_title="Intrusion Detection", page_icon="🛡️")

model = joblib.load("rf_model.pkl")

st.title("🛡️ Cybersecurity Intrusion Detector")
st.write("دخّلي بيانات الـ session وهيتقالك في هجوم متوقع ولا لأ")

with st.form("session_form"):
    col1, col2 = st.columns(2)

    with col1:
        network_packet_size = st.number_input(
            "Network Packet Size", min_value=1, value=500
        )
        login_attempts = st.number_input(
            "Login Attempts", min_value=1, value=4
        )
        session_duration = st.number_input(
            "Session Duration (seconds)", min_value=0.1, value=500.0
        )
        ip_reputation_score = st.slider(
            "IP Reputation Score", 0.0, 1.0, 0.3
        )
        failed_logins = st.number_input(
            "Failed Logins", min_value=0, value=1
        )

    with col2:
        protocol_type = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
        encryption_used = st.selectbox(
            "Encryption Used", ["AES", "DES", "Unknown / Not Sure"]
        )
        browser_type = st.selectbox(
            "Browser Type", ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
        )
        unusual_time_access = st.selectbox(
            "Unusual Time Access?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    raw_input = {
        "network_packet_size": network_packet_size,
        "protocol_type": protocol_type,
        "login_attempts": login_attempts,
        "session_duration": session_duration,
        "encryption_used": "" if encryption_used == "Unknown / Not Sure" else encryption_used,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": failed_logins,
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access,
    }

    processed = preprocess_input(raw_input)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"🚨 Attack Detected — احتمالية الهجوم {proba:.1%}")
    else:
        st.success(f"✅ No Attack — احتمالية الهجوم {proba:.1%}")
