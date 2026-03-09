import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np

user_directories = ['text1.txt', 'text2.txt', 'text3.txt','control.txt', 'unrelated.txt', 'HumanText.txt', 'unrelated2.txt']
embeddings_list = []
all_sentences = []
all_sources = []
dimensions = 3
neighbors = 5
multiplier = 1
x_cordinate = 'dim1'
y_cordinate = 'dim2'
z_cordinate = 'dim3'
color_cordinate = lambda x: 'dim4' if x == 4 else 'source'
symbol_cordinate = lambda x: 'source' if x == 4 else None


for files in user_directories:
    text = utils.open_and_clean_text(files)
    sentence = nltk.sent_tokenize(text)
    embeddings = utils.embed(sentence)
    embeddings_list.append(embeddings)
    all_sentences.extend(sentence)
    all_sources.extend([files] * len(sentence))

all_embeddings = np.vstack(embeddings_list)
umap_reduced_embeddings = utils.umap_reduce(all_embeddings, dimensions, neighbors)

if dimensions == 3:
    df_umap_combined = utils.create_3d_dataframe(umap_reduced_embeddings, all_sentences, multiplier, all_sources)
elif dimensions == 4:   
    df_umap_combined = utils.create_4d_dataframe(umap_reduced_embeddings, all_sentences, multiplier, all_sources)



fig_umap = px.scatter_3d(
    df_umap_combined,
    x=x_cordinate,
    y=y_cordinate,
    z=z_cordinate,
    color = color_cordinate(dimensions),
    symbol = symbol_cordinate(dimensions),
    hover_data=['sentences', 'source'],
    title='UMAP 3D Scatter Plot'
)

fig_umap.show()

# # --- OLD UMAP ENGINE ---
# import umap
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='cosine')
# umap_reduced = reducer.fit_transform(all_embeddings)

# # --- NEW PACMAP ENGINE ---
# import pacmap
# reducer = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
# # Note: PaCMAP uses Euclidean by default, but you can pre-calculate a 
# # cosine distance matrix if you want that specific text-alignment.
# umap_reduced = reducer.fit_transform(all_embeddings)