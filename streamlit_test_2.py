# python -m streamlit run streamlit_test_2.py
import streamlit as st
import Dynamic_comparisons_2 as dc



st.set_page_config(layout="wide")

st.title("Semantic Similarity Lab")


with st.expander("About this tool"):
    st.write(
        """This tool allows you to visualize the semantic similarity of texts using various embedding models and dimensionality reduction algorithms. 
        You can input multiple texts, select an embedding model, and choose a dimensionality reduction technique to see how the texts relate to each other in a reduced-dimensional space. You are able to visualize up to 6 dimensions. 
        Please note, visualization options will vary based on the number of dimensions you choose. For 3 dimensions, scatter, line, and matrix plots are available. For 4 dimensions, scatter and matrix plots are available. For 5 dimensions, 
        only scatter plot is available. For 6 dimensions, only cone plot is available. t-SNE is limited to 3 dimensions for visualization purposes using the barnes-hut method, so if you select t-SNE, the dimension count will be set to 3 and the other dimension options will be hidden."""
    )

#SIDE BAR AREA
models = [
"sentence-transformers/all-MiniLM-L6-v2",
"sentence-transformers/all-MiniLM-L12-v2",
"sentence-transformers/paraphrase-MiniLM-L6-v2",
"sentence-transformers/paraphrase-MiniLM-L12-v2",
"sentence-transformers/all-mpnet-base-v2",
"sentence-transformers/paraphrase-mpnet-base-v2",
"sentence-transformers/all-distilroberta-v1",
"sentence-transformers/multi-qa-mpnet-base-dot-v1",
"sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
"sentence-transformers/multi-qa-distilbert-cos-v1",
"sentence-transformers/paraphrase-distilroberta-base-v1",
"sentence-transformers/paraphrase-albert-small-v2",
"sentence-transformers/distiluse-base-multilingual-cased-v2",
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
"sentence-transformers/LaBSE",
"BAAI/bge-small-en-v1.5",
"BAAI/bge-base-en-v1.5",
"BAAI/bge-large-en-v1.5",
"intfloat/e5-small-v2",
"intfloat/e5-base-v2",
"intfloat/e5-large-v2"
]

model_selection = [
"all-MiniLM-L6-v2",
"all-MiniLM-L12-v2",
"paraphrase-MiniLM-L6-v2",
"paraphrase-MiniLM-L12-v2",
"all-mpnet-base-v2",
"paraphrase-mpnet-base-v2",
"all-distilroberta-v1",
"multi-qa-mpnet-base-dot-v1",
"multi-qa-MiniLM-L6-cos-v1",
"multi-qa-distilbert-cos-v1",
"paraphrase-distilroberta-base-v1",
"paraphrase-albert-small-v2",
"distiluse-base-multilingual-cased-v2",
"paraphrase-multilingual-MiniLM-L12-v2",
"LaBSE",
"bge-small-en-v1.5",
"bge-base-en-v1.5",
"bge-large-en-v1.5",
"e5-small-v2",
"e5-base-v2",
"e5-large-v2"
]



embedding_model = st.sidebar.selectbox("Embedding Model", options=model_selection, index=0)
embedding_model = models[model_selection.index(embedding_model)]

    

reduction_algorithm = st.sidebar.selectbox(
    "Select dimensionality reduction algorithm",
    ['PaCMAP', 'UMAP', 'PCA', 'tSNE'],
    index=0
)

st.sidebar.title("controls")


# dimension_count = st.sidebar.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)





def pacMap_settings_func() -> dict: 
    with st.sidebar.expander("PaCMAP Settings"):
        neighbor_count = st.number_input("Neighbor count", min_value=2, value=15)
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        MN_ratio = st.number_input("MN_ratio", value=0.5)
        FP_ratio = st.number_input("FP_ratio", value=2.0)
        lr = st.number_input("Learning Rate", value=1.0)
        num_iters = st.number_input("Iterations", value=1000)
        distance = st.selectbox("Distance Metric", ["angular", "euclidean", "manhattan"], index=0)
        random_state = st.number_input("Random State", value=42)
        init = st.selectbox("Initialization Method", ["pca", "random"])
        apply_pca = st.checkbox("Apply PCA", value=True)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_neighbors": neighbor_count,
            "n_components": dimension_count,
            "num_iters": num_iters,
            "random_state": random_state,
            "MN_ratio": MN_ratio,
            "FP_ratio": FP_ratio,
            "lr": lr,
            "apply_pca": apply_pca,
            "distance": distance,
            "init": init,
            "multiplier": multiplier
        }
    
def umap_settings_func() -> dict:
    with st.sidebar.expander("UMAP Settings"):
        neighbor_count = st.number_input("Neighbor count", min_value=2, value=15)
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        distance = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0)
        random_state = st.number_input("Random State", value=42)
        min_dist = st.number_input("Min Dist", value=0.1)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_neighbors": neighbor_count,
            "n_components": dimension_count,
            "random_state": random_state,
            "distance": distance,
            "min_dist": min_dist,
            "multiplier": multiplier
        }
    
def pca_settings_func() -> dict:
    with st.sidebar.expander("PCA Settings"):
        dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        svd_solver = st.selectbox("SVD Solver", ["auto", "full", "arpack", "randomized"], index=0)

        if svd_solver == "randomized":
            random_state = st.number_input("Random State", value=42)
        else:
            random_state = None

        whiten = st.checkbox("Whiten", value=False)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_components": dimension_count,
            "svd_solver": svd_solver,
            "whiten": whiten,
            "random_state": random_state,
            "multiplier": multiplier
        }

def tsne_settings_func() -> dict:
    with st.sidebar.expander("t-SNE Settings"):
        # dimension_count = st.selectbox("Dimension count/n components", [3, 4, 5, 6], index=0)
        st.write("t-SNE only supports 3 dimensions or less for visualization purposes, so the dimension count is set to 3 in this application.")
        perplexity = st.number_input("Perplexity", value=3, min_value=1)
        learning_rate = st.number_input("Learning Rate", value=200.0)
        max_iter = st.number_input("Number of Iterations", value=1000)
        random_state = st.number_input("Random State", value=42)
        multiplier = st.number_input("Scale multiplier", min_value=1, value=1)

        return {
            "n_components": 3,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "random_state": random_state,
            "multiplier": multiplier
        }

if reduction_algorithm == 'PaCMAP':
    reducer_settings = pacMap_settings_func()
if reduction_algorithm == 'UMAP':
    reducer_settings = umap_settings_func()
if reduction_algorithm == 'tSNE':
    reducer_settings = tsne_settings_func()
if reduction_algorithm == 'PCA':
    reducer_settings = pca_settings_func()

print("reducer:", repr(reducer_settings))

multilingual_models = [
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE"
]
multilingual_indicator = ""

if embedding_model in multilingual_models:
    multilingual_indicator = "is"
else:
    multilingual_indicator = "is not"

st.markdown(f"Current model: **{embedding_model}**. This model **{multilingual_indicator}** multilingual.")
st.markdown(f"Current reduction algorithm: **{reduction_algorithm}**")

#conditionals for the plot

#color scale for 4d and 5d concerning scatter, matrix, and cone plot
#3d does not have a color scale option

if reducer_settings['n_components'] >= 4:
    color_scale = st.sidebar.selectbox(
        "Color scale",
        ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo'],
        index=0
    )
else:
    color_scale = None

#this only concerns only cone plot
if reducer_settings['n_components']  == 6:
    size_mode = st.sidebar.selectbox(
        "Size mode",
        ['scaled', 'absolute'],
        index=0
    )

    size_ref = st.sidebar.number_input("Size reference", value=5)


###### TEXT CAPTURE AREA START

st.divider()

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

#TEXT CAPTURE AREA END

#text boxes start #######################

st.divider()

add_text_col, remove_text_col = st.columns(2)

with add_text_col:
    if st.button("Add Another Text Box"):
        st.session_state.num_texts += 1
        st.rerun()

with remove_text_col:
    if st.button("Remove Text Box"):
        if st.session_state.num_texts > 1:
            st.session_state.num_texts -= 1
            st.rerun()

st.divider()

# text boxes end #######################



def process_and_plot(reduction_algorithm):
    if st.button("Process Texts") and len(text_data) > 0:
        with st.spinner("Processing texts..."):

            data_frame = dc.produce_dataframe(text_data, labels, reduction_algorithm, reducer_settings, embedding_model)

            
            metrics = dc.metrics(data_frame)

            with st.expander("Metrics"):
                st.write("**Description:**")
                st.write(metrics['description'])
                st.write("**Missing Values:**")
                st.write(metrics['missing_values'])
                st.write("**Unique Sources:**")
                st.write(metrics['unique_sources'])
                st.write("**Unique Sentences:**")
                st.write(metrics['unique_sentences'])
                st.write("**Sentence Counts per Source:**")
                st.write(metrics['sentence_counts_per_source'])
                st.write("**Centroid per Source:**")
                st.write(metrics['centroid_per_source'])
                st.write("**Standard Deviation per Source:**")
                st.write(metrics['std_per_source'])
                st.write("**Total Sentences:**")
                st.write(metrics['total_sentences'])
                st.write("**Source List:**")
                st.write(metrics['source_list'])
                

            if reducer_settings['n_components']  == 6:
                
                cone_plot = dc.cone_plot(data_frame, color_scale, size_mode, size_ref, reducer_settings['n_components'], reduction_algorithm, embedding_model)
                st.plotly_chart(cone_plot, width="stretch")
                st.warning("scatter plot, matrix, and line plot not available for 6 dimensions")
            
            if reducer_settings['n_components'] == 5:


                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)

                st.plotly_chart(scatter_fig, width="stretch")

                st.warning("line plot and matrix not available for 5 dimensions")
                st.warning("cone plot not available for dimensions 5 through 3")
                st.warning("matrix plot not available for 5 dimensions and lower")

            if reducer_settings['n_components'] == 4:

                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)
                matrix_fig = dc.scatter_matrix(data_frame, reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)

                st.plotly_chart(scatter_fig, width="stretch")
                st.plotly_chart(matrix_fig, width="stretch")

                st.warning("cone plot not available for dimensions 5 through 3")
                st.warning("line plot not available for 4 dimensions")

            if reducer_settings['n_components'] == 3:
                

                scatter_fig = dc.scatter_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)
                matrix_fig = dc.scatter_matrix(data_frame, reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)
                line_fig = dc.line_plot(data_frame, 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], reduction_algorithm, embedding_model)

                st.plotly_chart(scatter_fig, width="stretch")
                st.plotly_chart(matrix_fig, width="stretch")
                st.plotly_chart(line_fig, width="stretch")

                st.warning("cone plot not available for dimensions 5 through 3")
        




    

process_and_plot(reduction_algorithm)
  





