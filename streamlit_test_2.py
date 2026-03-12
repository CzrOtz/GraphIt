import streamlit as st
import Dynamic_comparisons_2 as dc

st.title("my first streamlit app")

st.markdown(
    """Im learning how to use this"""
)




if 'num_texts' not in st.session_state:
    st.session_state.num_texts = 2

col_inputs = st.columns(st.session_state.num_texts)
text_data = []
labels = []

for i in range(st.session_state.num_texts):
    with col_inputs[i]:
        label = st.text_input(f"Label {i+1}", value=f"Doc {i+1}", key=f"label_{i}")
        content = st.text_area(f"Paste Text {i+1}", height=300, key=f"text_{i}")
        if content:
            text_data.append(content)
            labels.append(label)

col1, col2, col3 = st.columns([3, 2, 1])

st.divider()

with col3:
    if st.button("Add Another Text Box"):
        st.session_state.num_texts += 1
        st.rerun()

with col2:
    if st.button("remove text box"):
        if st.session_state.num_texts > 1:
            st.session_state.num_texts -= 1
            st.rerun()

with col1:
    if st.button("Process Texts"):
        st.write("Processing the following texts:")
        pacMap_df = dc.pacMap_dataframe(text_data, labels, 3, 15, 1)
        scatter_fig = dc.scatter_plot(pacMap_df, 'dim1', 'dim2', 'dim3', 3)
        matrix_fig = dc.scatter_matrix(pacMap_df, 3)
        st.plotly_chart(scatter_fig, use_container_width=True)
        st.plotly_chart(matrix_fig, use_container_width=True)



