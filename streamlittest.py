import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import nltk
import torch
import utils  # Your custom utility file
import sys

# --- Page Config ---
st.set_page_config(page_title="Semantic Analysis Scratchpad", layout="wide")

# --- Verification ---
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Warning: Slow)"

# --- Sidebar Settings ---
with st.sidebar:
    st.title("Settings")
    st.info(f"Running on: {device_name}")
    dimensions = st.selectbox("Dimensions", [3, 4], index=0)
    neighbors = st.slider("PaCMAP Neighbors", 2, 50, 15)
    multiplier = st.number_input("Scale Multiplier", value=15)
    
    st.divider()
    st.write("### Extrema Labels")
    show_extrema = st.checkbox("Show Extrema in UI", value=True)

st.title("🚀 Semantic Comparison Scratchpad")
st.write("Paste your texts below to compare their logic in 3D space.")

# --- Dynamic Text Inputs ---
# We use session state to allow you to add as many boxes as you want
if 'num_texts' not in st.session_state:
    st.session_state.num_texts = 2

col_inputs = st.columns(st.session_state.num_texts)
text_data = []

for i in range(st.session_state.num_texts):
    with col_inputs[i]:
        label = st.text_input(f"Label {i+1}", value=f"Doc {i+1}", key=f"label_{i}")
        content = st.text_area(f"Paste Text {i+1}", height=300, key=f"text_{i}")
        if content:
            text_data.append((label, content))

if st.button("Add Another Text Box"):
    st.session_state.num_texts += 1
    st.rerun()

st.divider()

# --- Main Analysis Logic ---
if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    if len(text_data) < 1:
        st.error("Please paste at least one text to analyze.")
    else:
        with st.spinner("Processing embeddings on GPU..."):
            all_sentences = []
            all_sources = []
            embeddings_list = []

            for label, raw_text in text_data:
                # 1. Clean and Tokenize
                text = utils.clean_text(raw_text)
                sentences = nltk.sent_tokenize(text)
                
                if len(sentences) < 2:
                    st.warning(f"Skipping '{label}': Needs at least 2 sentences.")
                    continue

                # 2. Embed (Uses your CUDA-enabled utils.embed)
                embeddings = utils.embed(sentences)
                
                embeddings_list.append(embeddings)
                all_sentences.extend(sentences)
                all_sources.extend([label] * len(sentences))

            if embeddings_list:
                # 3. PaCMAP Reduction
                all_embeddings = np.vstack(embeddings_list)
                pacmap_results = utils.pacmap_reduce(all_embeddings, dimensions, neighbors)

                # 4. Create DataFrame
                if dimensions == 3:
                    df = utils.create_3d_dataframe(pacmap_results, all_sentences, multiplier, all_sources)
                else:
                    df = utils.create_4d_dataframe(pacmap_results, all_sentences, multiplier, all_sources)

                # 5. Extract Extrema Logic
                x_pos = df.loc[df['dim1'].idxmax()]
                x_neg = df.loc[df['dim1'].idxmin()]
                y_pos = df.loc[df['dim2'].idxmax()]
                y_neg = df.loc[df['dim2'].idxmin()]
                z_pos = df.loc[df['dim3'].idxmax()]
                z_neg = df.loc[df['dim3'].idxmin()]

                # 6. Visualization Layout
                graph_col, info_col = st.columns([3, 1])

                with graph_col:
                    fig = px.scatter_3d(
                        df, x='dim1', y='dim2', z='dim3',
                        color='dim4' if dimensions == 4 else 'source',
                        symbol='source',
                        hover_data=['sentences', 'source'],
                        title='Semantic Manifold'
                    )
                    fig.update_layout(legend=dict(orientation="h", y=-0.1))
                    st.plotly_chart(fig, use_container_width=True)

                with info_col:
                    if show_extrema:
                        st.subheader("Boundary Logic")
                        st.write("**X-Max (Positive):**")
                        st.caption(f"[{x_pos['source']}] {x_pos['sentences']}")
                        st.write("**X-Min (Negative):**")
                        st.caption(f"[{x_neg['source']}] {x_neg['sentences']}")
                        st.divider()
                        st.write("**Y-Max:**")
                        st.caption(f"[{y_pos['source']}] {y_pos['sentences']}")
                        st.write("**Z-Max:**")
                        st.caption(f"[{z_pos['source']}] {z_pos['sentences']}")

                # 7. Matrix View
                st.divider()
                st.subheader("Dimension Correlation Matrix")
                fig_matrix = px.scatter_matrix(df, dimensions=['dim1', 'dim2', 'dim3'], color='source')
                st.plotly_chart(fig_matrix, use_container_width=True)
            else:
                st.error("No valid sentences found to embed.")