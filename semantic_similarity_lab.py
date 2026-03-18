# python -m streamlit run semantic_similarity_lab.py
from multiprocessing import reduction
import plotly.express as px

import utils as ut
import streamlit as st
import Dynamic_comparisons_2 as dc

#add this timorrow
#sklearn.metrics.pairwise.cosine_similarity — takes two arrays of embeddings and returns a similarity matrix

st.set_page_config(layout="wide")

st.title("Semantic Similarity Lab")


with st.expander("About this tool"):
    st.write(
        """This tool allows you to visualize the semantic similarity of texts using various embedding models and dimensionality reduction algorithms. 
        You can input multiple texts, select an embedding model, and choose a dimensionality reduction technique to see how the texts relate to each other in a reduced-dimensional space."""
    )

    st.write("""You are able to visualize up to 6 dimensions. 
        Please note, visualization options will vary based on the number of dimensions you choose. For 3 dimensions, scatter, line, and matrix plots are available. For 4 dimensions, scatter and matrix plots are available. For 5 dimensions, 
        only scatter plot is available. For 6 dimensions, only cone plot is available. t-SNE is limited to 3 dimensions for visualization purposes using the barnes-hut method, so if you select t-SNE, the dimension count will be set to 3 and the other dimension options will be hidden."""
    )

    st.write(
        """ more text later"""
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



with st.sidebar.expander("Grand Tour Settings"):
    gt_color_scale = st.selectbox(
        "Grand Tour Color Scale",
        ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo'],
        index=0
    )

    gt_frame_duration = st.slider("Frame Duration", value=500, min_value=100, max_value=2000, step=50)
    gt_transition_duration = st.slider("Transition Duration", value=450, min_value=100, max_value=2000, step=50)
    gt_easing = st.selectbox("Easing Function", ["cubic-in-out", "quadratic-in-out", "linear", "sine-in-out", "exp-in-out", "circle-in-out", "back-in-out", "elastic-in-out", "bounce-in-out"],index=0)



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
        "3D plot color scale",
        ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo'],
        index=0
    )
else:
    color_scale = None

#this only concerns only cone plots
if reducer_settings['n_components']  == 6:
    st.sidebar.write("Cone Plot Settings")
    size_mode = st.sidebar.selectbox(
        "Size mode",
        ['scaled', 'absolute'],
        index=0
    )

    size_ref = st.sidebar.number_input("Size reference", value=5)


###### TEXT CAPTURE AREA START

st.divider()

with st.expander("Clean Text Settings"):
    fix_unicode = st.checkbox("Fix Unicode Characters", value=True)
    to_ascii = st.checkbox("Convert to ASCII", value=True)
    lower = st.checkbox("Convert to Lowercase", value=False)
    no_urls = st.checkbox("Remove URLs", value=True)
    no_emails = st.checkbox("Remove Emails", value=True)
    no_numbers = st.checkbox("Remove Numbers", value=False)
    lang = st.selectbox("Language for cleaning", options=['en', 'es', 'fr', 'de', 'it', 'pt'], index=0)
    remove_brackets = st.checkbox("Remove Square Brackets and everything in them", value=True)
    remove_curly_braces = st.checkbox("Remove Curly Braces and everything in them", value=True)
    remove_parentheses = st.checkbox("Remove Parentheses and everything in them", value=True)
    remove_extra_whitespace = st.checkbox("Remove Extra Whitespace", value=True)
    remove_special_characters = st.checkbox("Remove Special Characters", value=False)


clean_text_settings = {
    "fix_unicode": fix_unicode,
    "to_ascii": to_ascii,
    "lower": lower,
    "no_urls": no_urls,
    "no_emails": no_emails,
    "no_numbers": no_numbers,
    "lang": lang,
    "remove_brackets": remove_brackets,
    "remove_curly_braces": remove_curly_braces,
    "remove_parentheses": remove_parentheses,
    "remove_extra_whitespace": remove_extra_whitespace,
    "remove_special_characters": remove_special_characters
}

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





def GeneratePlots(data_frame):
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
        

def cosine_similarity_functions():
        if len(text_data) < 2:
            st.warning("Cosine similarity requires at least two separate text sources")
            return
        show_cosine_similarity = dc.run_cosine_similarity(text_data, labels, embedding_model, clean_text_settings)
        st.write("Cosine Similarity Matrix between the two sets of embeddings:")
        st.write(show_cosine_similarity)
        fig = px.bar(
            show_cosine_similarity, 
            x=show_cosine_similarity.index, 
            y="similarity", 
            title=f"Raw Similarity Scores using {embedding_model}",
            hover_data=["sentence_a", "source_a", "nearest_sentence_b", "source_b"],
            )
        
        st.plotly_chart(fig)

def grand_tour_projection_func():
    grand_tour = dc.grand_tour_projection(text_data, labels, embedding_model, clean_text_settings)
    st.write("Grand Tour Projection:")
    st.write("This is a dynamic visualization that continuously rotates the high-dimensional data to help you understand the structure of the data in its original dimensionality. It can reveal patterns and relationships that may not be apparent in static 2D or 3D projections.")
    gt_scatter_plot_animated = dc.grand_tour_scatter_plot(grand_tour, embedding_model, gt_color_scale, gt_frame_duration, gt_transition_duration, gt_easing)
    st.plotly_chart(gt_scatter_plot_animated, width="stretch")
    st.write(grand_tour)

def metrics(data_frame):
    metrics = dc.metrics(data_frame)
    st.write("**Description:**")
    st.write(metrics['description'])
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
    

process_option = st.radio("Process Options", options=["Show Post reduction only", "Show Pre-reduction only", "Show Both"], index=0)
    
if st.button("Process Text") and len(text_data) > 0:
    
    with st.expander("post cleaning text review"):
        
        st.write("If you observe unwanted charcaters, or any issues, please click on the checkboxes above to adjust and remove unwanted characters, then click the 'Process Text' button again to see the updated cleaned text.")
        st.write("Note, it may be requirted to clean te text further in an external editor if there are specific formatting issues that need to be addressed, such as removing section headers, footnotes, or other non-sentence elements that may interfere with the embedding and analysis process.")
        st.divider()
        st.write("This is the text after cleaning")
        st.divider()
        for text in text_data:
            st.write(ut.clean_text(text, clean_text_settings))

    if process_option == "Show Pre-reduction only" or process_option == "Show Both":
        st.divider()
        st.markdown("### Pre-reduction Analysis")
        st.write("Before we reduce the dimensionality of our embeddings, we can analyze the raw cosine similarity between the texts and also explore a Grand Tour projection to get an initial sense of the structure of the data in its original high-dimensional space.")
        st.divider()
        with st.spinner(f"Processing cosine similarity with {embedding_model}..."):
            cosine_similarity_functions()
        with st.spinner(f"Generating Grand Tour with {embedding_model}..."):
            grand_tour_projection_func()

        st.divider()
    if process_option == "Show Post reduction only" or process_option == "Show Both":
        st.markdown("### Dimensionality Reduction and Visualization")
        st.write("Now we will reduce the dimensionality of our embeddings using the selected reduction algorithm and visualize the results. The visualizations will help us understand how the texts relate to each other in the reduced-dimensional space, and we can also calculate various metrics to further analyze the structure of the data after reduction.")
        st.divider()

        with st.spinner(f"reducing dimensions using {reduction_algorithm} and generating plots..."):
            data_frame_reduced_dimmension = dc.produce_dataframe(text_data, labels, reduction_algorithm, reducer_settings, embedding_model, clean_text_settings)
        with st.spinner(f"Generating Plots..."):
            GeneratePlots(data_frame_reduced_dimmension)
        with st.spinner(f"Calculating metrics..."):      
            metrics(data_frame_reduced_dimmension)



       
    





