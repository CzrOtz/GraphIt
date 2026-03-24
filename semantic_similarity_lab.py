import plotly.express as px
import numpy as np
import utils as ut
import streamlit as st
import process_functions as dc
import pandas as pd


st.set_page_config(layout="wide", page_title="GraphIt", page_icon="🧪")

st.title("GraphIt: High-Dimensional Semantic Similarity Laboratory")



with st.expander("📖 About GraphIt: Semantic Similarity Lab", expanded=True):
 
    st.markdown("""
    **GraphIt** is a multi-dimensional semantic laboratory for researchers and data scientists who need to *see* how language models organize meaning.
    
    Paste raw text from up to **10 sources**. GraphIt tokenizes each source into sentences, vectorizes them using your chosen Hugging Face sentence transformer, 
    and projects the resulting high-dimensional embeddings into an interactive 3D–6D coordinate space — exposing semantic clusters, 
    cross-document overlap, centroid drift, and latent thematic geometry that standard similarity scores cannot surface.
    """)
 
    st.divider()
 
    # ── Visualization Capabilities ──────────────────────────────────────────
    st.markdown("#### 📐 Visualization Capabilities")
    st.markdown("""
    GraphIt maps reduction dimensions to visual channels, encoding up to **6 semantic axes simultaneously**:
 
    | Dimensions | Available Plot Types | Encoding Logic |
    | :--- | :--- | :--- |
    | **3D** | Scatter · Line · Scatter Matrix | X, Y, Z spatial coordinates |
    | **4D** | Scatter · Scatter Matrix | X, Y, Z + continuous color scale (`dim4`) |
    | **5D** | Scatter | X, Y, Z + color (`dim4`) + marker size (`dim5`, forced absolute) |
    | **6D** | Cone Plot | X, Y, Z position + U, V, W trajectory vectors |
 
    > **t-SNE** is restricted to **3D** via the Barnes-Hut approximation for tractable performance.  
    > **5D** marker size uses `dim5.abs()` — all size values are forced non-negative before rendering.  
    > **6D** cone `sizemode` and `sizeref` are configurable in the sidebar.
    """)
 
    st.divider()
 
    # ── Pre-Reduction Analytics ──────────────────────────────────────────────
    st.markdown("#### 🔬 Pre-Reduction Analytics")
 
    col_pre1, col_pre2 = st.columns(2)
 
    with col_pre1:
        st.markdown("""
        **Cosine Similarity Matrix**  
        For every pair of sources, each sentence in Source A is matched against every sentence in Source B. 
        The nearest-neighbor sentence from Source B is returned alongside its similarity score, source label, and full sentence text. 
        All pairwise source combinations are computed automatically. Requires **at least 2 sources.**
        """)
 
    with col_pre2:
        st.markdown("""
        **Grand Tour Projection**  
        A 2D animated scatter that cycles through adjacent dimension pairs of the *raw* embedding space — before any reduction distorts global relationships. 
        Reveals structural geometry in the original high-dimensional space. 
        Fully configurable: point size, frame duration, transition duration, and easing function.
        """)
 
    st.divider()
 
    # ── Operational Workflow ─────────────────────────────────────────────────
    st.markdown("#### 🛠️ Operational Workflow")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("""
        **1 · Engine Configuration**  
        Select an **Embedding Model** and **Reduction Algorithm** in the sidebar. 
        Each algorithm exposes its full hyperparameter surface — tune *Perplexity* (t-SNE), 
        *MN Ratio / FP Ratio* (PaCMAP), *Min Dist* (UMAP), or *SVD Solver* (PCA) 
        to control local vs. global structure preservation.  
        The active model's Hugging Face Model Card is linked directly in the sidebar.
        Multilingual support is **automatically detected** and labeled for the selected model.
 
        **2 · Data Ingestion**  
        Paste text into the document containers **(1–10 sources)**. Label each source. 
        A minimum of **3 sentences per source** is required for all operations. 
        A minimum of **2 sources** is required for cosine similarity analysis.
        """)
 
    with col2:
        st.markdown("""
        **3 · Sanitization**  
        Expand **Clean Text Settings** to configure 12 preprocessing options — 
        Unicode normalization, URL/email removal, bracket stripping, whitespace collapsing, and more. 
        A post-cleaning preview renders before embedding begins so you can verify output before committing.
        Complex formatting should be resolved in an external editor prior to pasting.
 
        **4 · Execution & Analytics**  
        Select a processing scope and click **Process Text**:  
        — **Pre-reduction** → cosine similarity matrix and Grand Tour animation  
        — **Post-reduction** → manifold projection, spatial plots, centroid metrics (graph included), standard deviation (graph included), magnitude and other metrics
        — **Both** → full pipeline  
        
        > ⚠️ **t-SNE:** Perplexity must be less than your total sentence count across all sources. 
        The application validates this at runtime and will halt with a descriptive error if violated.
        """)
 
    st.divider()
 
    # ── Open Source CTA ──────────────────────────────────────────────────────
    st.markdown("#### 🚀 Open Source & Collaboration")
    st.write("If you find an issue, or want a custom feature, please contribute or feel free to fork the repository to tailor it to your needs! GraphIt is open source on GitHub:")
    
    col_cta1, col_cta2, col_cta3 = st.columns(3)
 
    with col_cta1:
        st.markdown("🍴 [**Fork the Repository**](https://github.com/CzrOtz/GraphIt)  \nCustomize it for your use case.")
 
    with col_cta2:
        st.markdown("🔧 [**Submit a Pull Request**](https://github.com/CzrOtz/GraphIt/pulls)  \nContribute a feature or fix.")
 
    with col_cta3:
        st.markdown("🐛 [**Report an Issue**](https://github.com/CzrOtz/GraphIt/issues)  \nFound a bug? Let us know.")
    
    st.divider()
    st.markdown("Built by **Cesar Ortiz** · [GitHub](https://github.com/CzrOtz)")

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

st.sidebar.title("Controls")

embedding_model = st.sidebar.selectbox("Embedding Model", options=model_selection, index=0)
embedding_model = models[model_selection.index(embedding_model)]
hf_url = f"https://huggingface.co/{embedding_model}"
st.sidebar.markdown(f"🔗 [View Model Card on Hugging Face]({hf_url})")

    

reduction_algorithm = st.sidebar.selectbox(
    "Select dimensionality reduction algorithm",
    ['PaCMAP', 'UMAP', 'PCA', 'tSNE'],
    index=0
)

algo_links = {
    "PaCMAP": "https://github.com/YingfanWang/PaCMAP",
    "UMAP": "https://umap-learn.readthedocs.io/",
    "PCA": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
    "tSNE": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
}

st.sidebar.markdown(f"🔬 [Read {reduction_algorithm} Documentation]({algo_links[reduction_algorithm]})")




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
    gt_point_size = st.slider("Cordinae Point Size", value=10, min_value=1, max_value=15, step=1)
    gt_frame_duration = st.slider("Animation Frame Duration", value=500, min_value=100, max_value=2000, step=50)
    gt_transition_duration = st.slider("Animation Transition Duration", value=450, min_value=100, max_value=2000, step=50)
    gt_easing = st.selectbox("Animation Easing Function", ["cubic-in-out", "quadratic-in-out", "linear", "sine-in-out", "exp-in-out", "circle-in-out", "back-in-out", "elastic-in-out", "bounce-in-out"],index=0)



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

    size_ref = st.sidebar.number_input("Size reference", value=2)


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

if st.session_state.num_texts < 10:
    with add_text_col:
        if st.button("Add Another Text Box"):
            st.session_state.num_texts += 1
            st.rerun()
else:    
    st.warning("Maximum number of text boxes reached")

with remove_text_col:
    if st.session_state.num_texts > 1:
        if st.button("Remove Text Box"):
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
    gt_scatter_plot_animated = dc.grand_tour_scatter_plot(grand_tour, embedding_model, gt_frame_duration, gt_transition_duration, gt_easing, gt_point_size)
    st.plotly_chart(gt_scatter_plot_animated, width="stretch")
    st.write(grand_tour)

def centroid_metrics_func(data_frame):
    if reducer_settings["n_components"] < 6:
        centroid_scatter = dc.scatter_plot_v2(data_frame.reset_index(), 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model)
    else:
        centroid_scatter = dc.cone_plot(data_frame, color_scale, size_mode, size_ref, reducer_settings['n_components'], reduction_algorithm, embedding_model)

    return centroid_scatter

def metrics(data_frame):

    st.write("**Metrics and Analysis:**")

    with st.expander("Dimensionally Reduced DataFrame"):
        st.write("This is the DataFrame containing the reduced-dimensionality embeddings for each sentence, along with their corresponding sources and original sentences. You can use this DataFrame to explore the relationships between the sentences in the reduced-dimensional space and to calculate various metrics based on their positions in this space.")
        st.write(data_frame)

    dims = ["dim1", "dim2", "dim3"] + [f"dim{i}" for i in range(4, len(data_frame.columns) + 1) if f'dim{i}' in data_frame.columns]

    extremes = {}

    for dim in dims:
        extremes[f'max_{dim}'] = [data_frame[dim].idxmax(), data_frame[dim].max(), data_frame.loc[data_frame[dim].idxmax(), 'source'], data_frame.loc[data_frame[dim].idxmax(), 'sentences']]
        extremes[f'min_{dim}'] = [data_frame[dim].idxmin(), data_frame[dim].min(), data_frame.loc[data_frame[dim].idxmin(), 'source'], data_frame.loc[data_frame[dim].idxmin(), 'sentences']]

    st.write("**Extremes in Each Dimension:**")
    extrmes_df = pd.DataFrame(extremes).T
    extrmes_df.columns = ['index', 'value', 'source', 'sentence']
    st.write(extrmes_df)

    copy_df = data_frame.copy()
    copy_df['magnitude'] = np.sqrt(copy_df[dims].pow(2).sum(axis=1))

    max_magnitude = copy_df.loc[copy_df['magnitude'].idxmax()]
    min_magnitude = copy_df.loc[copy_df['magnitude'].idxmin()]
    mid_magnitude = copy_df.loc[(copy_df['magnitude'] - copy_df['magnitude'].mean()).abs().idxmin()]

    magnitude_df = pd.DataFrame([
        {
            "type": "max_magnitude",
            "magnitude": max_magnitude['magnitude'],
            "source": max_magnitude['source'],
            "sentence": max_magnitude['sentences'],
            "index": max_magnitude.name
        },
        {
            "type": "min_magnitude",
            "magnitude": min_magnitude['magnitude'],
            "source": min_magnitude['source'],
            "sentence": min_magnitude['sentences'],
            "index": min_magnitude.name
        },
        {
            "type": "mid_magnitude",
            "magnitude": mid_magnitude['magnitude'],
            "source": mid_magnitude['source'],
            "sentence": mid_magnitude['sentences'],
            "index": mid_magnitude.name
        }
    ])
    st.write("**Magnitude Metrics:**")
    st.write(magnitude_df)

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
    with st.expander("Centroid Scatter Plot"):
        if reducer_settings["n_components"] < 5:
            centroid_scatter = dc.scatter_plot_v2(metrics["centroid_per_source"].reset_index(), 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model, False)
            st.plotly_chart(centroid_scatter, width="stretch")
        else:
            st.warning("Centroid scatter plot not available for 5 and 6 dimensions")

    st.write("**Standard Deviation per Source:**")
    st.write(metrics['std_per_source'])
    with st.expander("Standard Deviation Scatter Plot"):
        if reducer_settings["n_components"] < 5:
            
            centroid_scatter = dc.scatter_plot_v2(metrics["std_per_source"].reset_index(), 'dim1', 'dim2', 'dim3', reducer_settings['n_components'], color_scale, reduction_algorithm, embedding_model, True)
            st.plotly_chart(centroid_scatter, width="stretch")
        else:
            st.warning("Standard deviation scatter plot not available for 5 and 6 dimensions")

    st.write("**Total Sentences:**")
    st.write(metrics['total_sentences'])
    st.write("**Source List:**")
    st.write(metrics['source_list'])



process_option = st.radio("Process Options", options=["Show Post reduction only", "Show Pre-reduction only", "Show Both"], index=0)
process_text = True

  
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




       
    





