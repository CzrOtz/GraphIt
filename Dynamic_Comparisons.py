import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np


essays = ['Human2.txt', 'AI2.txt']

embeddings_list = []
all_sentences = []
all_sources = []
dimensions = 3
neighbors = 15
multiplier = 15
x_cordinate = 'dim1'
y_cordinate = 'dim2'
z_cordinate = 'dim3'
color_cordinate = lambda x: 'dim4' if x == 4 else 'source'



for files in essays:
    text = utils.open_and_clean_text(files)
    sentence = nltk.sent_tokenize(text)
    embeddings = utils.embed(sentence)
    embeddings_list.append(embeddings)
    all_sentences.extend(sentence)
    all_sources.extend([files] * len(sentence))

all_embeddings = np.vstack(embeddings_list)
pacmap_reduced_embeddings = utils.pacmap_reduce(all_embeddings, dimensions, neighbors)

if dimensions == 3:
    df_pacmap_combined = utils.create_3d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)
elif dimensions == 4:   
    df_pacmap_combined = utils.create_4d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)

#extract the point in the x cordinate that has the highest value
x_positive_axis_extrema = df_pacmap_combined[[x_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[x_cordinate].idxmax()]
x_negative_axis_extrema = df_pacmap_combined[[x_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[x_cordinate].idxmin()]
y_positive_axis_extrema = df_pacmap_combined[[y_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[y_cordinate].idxmax()]
y_negative_axis_extrema = df_pacmap_combined[[y_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[y_cordinate].idxmin()]
z_positive_axis_extrema = df_pacmap_combined[[z_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[z_cordinate].idxmax()]
z_negative_axis_extrema = df_pacmap_combined[[z_cordinate, 'sentences', 'source']].loc[df_pacmap_combined[z_cordinate].idxmin()]


fig_pacmap = px.scatter_3d(
    df_pacmap_combined,
    x=x_cordinate,
    y=y_cordinate,
    z=z_cordinate,
    color = color_cordinate(dimensions),
    symbol = 'source',
    hover_data=['sentences', 'source'],
    title='PaCMAP 3D Scatter Plot'
)

# Move the legend to the bottom-center and horizontal
fig_pacmap.update_layout(
    legend=dict(
        orientation="h",        # Horizontal legend
        yanchor="bottom",
        y=-0.1,                # Pull it below the plot
        xanchor="center",
        x=0.5
    ),
    # You can also move the Color Bar if you want
    coloraxis_colorbar=dict(
        title="3rd Dim Variance",
        thicknessmode="pixels", thickness=15,
        lenmode="pixels", len=200,
        yanchor="top", y=1,
        ticks="outside"
    )
)

fig_matrix = px.scatter_matrix(
    df_pacmap_combined,
    dimensions=['dim1', 'dim2', 'dim3'],
    color='source', 
    symbol='source'
)



fig_matrix.show()

fig_pacmap.show()

