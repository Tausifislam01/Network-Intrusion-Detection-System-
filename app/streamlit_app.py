import streamlit as st
import pandas as pd
import requests
import time
import io

st.set_page_config(page_title="NIDS - Live Monitor", layout="wide")

st.title("🛡️ Live Network Intrusion Detection")
st.markdown("### Real-time Traffic Monitoring Dashboard")

API_URL = "http://localhost:8000/predict"

# Initialize session states for our continuous loop
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "is_monitoring" not in st.session_state:
    st.session_state.is_monitoring = False

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Stream (CSV)", type=["csv"])

# Configuration for the stream
BATCH_SIZE = st.sidebar.slider("Rows per batch", 10, 500, 100)
REFRESH_RATE = st.sidebar.slider("Refresh Rate (Seconds)", 2, 30, 5) 

if uploaded_file is not None:
    # Cache the data so we don't reload the file into memory on every single loop
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file, low_memory=False)
        
    df = load_data(uploaded_file)
    total_rows = len(df)
    
    st.sidebar.write(f"**Total flows in file:** {total_rows}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary"):
            st.session_state.is_monitoring = True
    with col2:
        if st.button("⏹️ Stop"):
            st.session_state.is_monitoring = False

    # --- UI Placeholders ---
    status_text = st.empty()
    progress_bar = st.progress(0)
    alert_banner = st.empty()
    
    chart_col, table_col = st.columns([1, 2])
    metrics_placeholder = chart_col.empty()
    table_placeholder = table_col.empty()

    if st.session_state.is_monitoring:
        while st.session_state.is_monitoring:
            if st.session_state.current_index >= total_rows:
                status_text.warning("End of data stream reached.")
                st.session_state.is_monitoring = False
                st.session_state.current_index = 0  # Reset for next run
                break
                
            # 1. Slice the dataframe to simulate a new batch of real-time data
            start_idx = st.session_state.current_index
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            status_text.info(f"📡 **Live Feed Active:** Analyzing flows {start_idx} to {end_idx}...")
            progress_bar.progress(end_idx / total_rows)
            
            # 2. Convert batch to an in-memory CSV to trick FastAPI into thinking it's a file upload
            csv_buffer = io.BytesIO()
            batch_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            files = {"file": ("stream_batch.csv", csv_buffer.getvalue(), "text/csv")}
            
            # 3. Send to API
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    preds_df = pd.DataFrame(result["preview"])
                    
                    # 4. Check for attacks ("BENIGN" is normal traffic)
                    attacks = preds_df[preds_df["prediction"] != "BENIGN"]
                    
                    if not attacks.empty:
                        alert_banner.error(f"🚨 **CRITICAL ALERT:** {len(attacks)} Malicious flows detected in current batch!")
                    else:
                        alert_banner.success("✅ Traffic is normal. No attacks detected.")
                        
                    # 5. Update Visuals
                    counts = pd.Series(result["prediction_counts"])
                    with metrics_placeholder.container():
                        st.subheader("Batch Distribution")
                        st.bar_chart(counts, color="#ff4b4b" if not attacks.empty else "#2e7b32")
                        
                    with table_placeholder.container():
                        st.subheader("Live Traffic Log")
                        
                        # Function to highlight malicious rows in red
                        def highlight_attacks(row):
                            if row['prediction'] != 'BENIGN':
                                return ['background-color: rgba(255, 75, 75, 0.2); color: white'] * len(row)
                            return [''] * len(row)
                            
                        st.dataframe(
                            preds_df.style.apply(highlight_attacks, axis=1), 
                            use_container_width=True,
                            hide_index=True
                        )
                        
                else:
                    status_text.error(f"Backend Error: {response.text}")
                    st.session_state.is_monitoring = False
                    break
                    
            except requests.exceptions.ConnectionError:
                status_text.error("❌ Backend connection failed! Is FastAPI running?")
                st.session_state.is_monitoring = False
                break
            
            # 6. Advance index and pause before next pull
            st.session_state.current_index += BATCH_SIZE
            time.sleep(REFRESH_RATE)
            
            # Force Streamlit to rerun the UI loop
            st.rerun()

else:
    st.info("👈 Please upload a network traffic CSV in the sidebar to begin monitoring.")