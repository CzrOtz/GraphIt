import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np

test = ["The town woke up every morning to the sound of the old clock tower. Its bells echoed across the river and through the narrow streets. Most people ignored it, but Daniel always listened carefully. To him, the sound meant another chance to try again. He worked in a tiny repair shop that smelled like oil and dust. Broken radios, watches, and toys filled the shelves. Daniel liked fixing things because it felt like reversing time. Every screw tightened was a small victory. One afternoon a child brought in a shattered music box. It took Daniel hours, but he rebuilt it piece by piece. When the melody finally played again, the child smiled so wide that the entire day felt worth it",
        "Far out in the desert stood a lonely research station. Wind constantly pushed sand against the metal walls. Inside, a scientist named Mira studied signals from distant space. Most nights the equipment produced nothing but static. Still, she listened patiently through her headphones. One evening a strange rhythm appeared in the noise. It was faint but clearly organized. Mira replayed the signal again and again to be sure. The pattern repeated every thirty seconds like a heartbeat. She felt a mix of excitement and fear. If the signal was real, it meant something out there was speaking. Mira leaned back in her chair, staring at the dark sky beyond the window. For the first time in years, the desert station did not feel lonely."]

story_names = ["Story 1", "Story 2"]

def pacMap_dataframe(text_passages: list, sources: list, dimensions: int, neighbors: int, multiplier: int) -> pd.DataFrame:
    embeddings_list = []
    all_sentences = []
    all_sources = []

    for i, source in zip(text_passages, sources):
        text = utils.clean_text(i)
        sentence = nltk.sent_tokenize(text)
        embeddings = utils.embed(sentence)
        embeddings_list.append(embeddings)
        all_sentences.extend(sentence)
        all_sources.extend([source] * len(sentence))

    all_embeddings = np.vstack(embeddings_list)
    pacmap_reduced_embeddings = utils.pacmap_reduce(all_embeddings, dimensions, neighbors)

    if dimensions == 3:
        df_pacmap_combined = utils.create_3d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)
    elif dimensions == 4:   
        df_pacmap_combined = utils.create_4d_dataframe(pacmap_reduced_embeddings, all_sentences, multiplier, all_sources)

    return df_pacmap_combined


def scatter_plot(data_frame: pd.DataFrame, x_cordinate: str, y_cordinate: str, z_cordinate: str, dimensions: int) -> px.scatter_3d:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_pacmap = px.scatter_3d(
        data_frame,
        x=x_cordinate,
        y=y_cordinate,
        z=z_cordinate,
        color = color_cordinate,
        symbol = 'source',
        hover_data=['sentences', 'source'],
        title='PaCMAP 3D Scatter Plot'
    )

    # Move the legend to the bottom-center and horizontal
    fig_pacmap.update_layout(
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor='center')
    )
    
    return fig_pacmap

def scatter_matrix(data_frame: pd.DataFrame, dimensions: int) -> px.scatter_matrix:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_matrix = px.scatter_matrix(
    data_frame,
    dimensions=['dim1', 'dim2', 'dim3'],
    color=color_cordinate, 
    symbol='source'
    )
    
    return fig_matrix


# test_df = pacMap_dataframe(test, story_names, 3, 10, 1)
# fig = scatter_plot(test_df, 'dim1', 'dim2', 'dim3', 3)
# fig_matrix = scatter_matrix(test_df, 3)

# fig.show()
# fig_matrix.show()