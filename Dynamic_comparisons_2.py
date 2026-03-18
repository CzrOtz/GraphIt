import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import streamlit as st



def produce_dataframe(text_passages: list, sources: list, reducer: str, settings: dict, embedding_model_name: str, clean_settings: dict) -> pd.DataFrame:
    embeddings_list = []
    all_sentences = []
    all_sources = []

    for i, source in zip(text_passages, sources):
        text = utils.clean_text(i, clean_settings)
        sentence = nltk.sent_tokenize(text)
        embeddings = utils.embed(sentence, embedding_model_name)
        embeddings_list.append(embeddings)
        all_sentences.extend(sentence)
        all_sources.extend([source] * len(sentence))

    all_embeddings = np.vstack(embeddings_list)


    if reducer == 'PaCMAP':
        reduced_embeddings = utils.pacmap_reduce_fully_tunable(all_embeddings, settings)
    if reducer == 'UMAP':
        reduced_embeddings = utils.umap_reduce_fully_tunable(all_embeddings, settings)
    if reducer == 'PCA':
        reduced_embeddings = utils.pca_reduce_fully_tunable(all_embeddings, settings)
    if reducer == 'tSNE':
        reduced_embeddings = utils.tsne_reduce_fully_tunable(all_embeddings, settings)
    
    
    if settings['n_components'] == 3:
        df_combined = utils.create_3d_dataframe(reduced_embeddings, all_sentences, settings['multiplier'], all_sources)
    if settings['n_components'] == 4:   
        df_combined = utils.create_4d_dataframe(reduced_embeddings, all_sentences, settings['multiplier'], all_sources)

    if settings['n_components'] == 5:
        df_combined = utils.create_5d_dataframe(reduced_embeddings, all_sentences, settings['multiplier'], all_sources)

    if settings['n_components'] == 6:
        df_combined = utils.create_6d_dataframe(reduced_embeddings, all_sentences, settings['multiplier'], all_sources)

    return df_combined

def run_cosine_similarity(text_passages: list, sources: list, embedding_model_name: str, clean_settings: dict) -> pd.DataFrame:
    embeddings_list = []
    sentences_list = []

    for text_passage, source in zip(text_passages, sources):
        text = utils.clean_text(text_passage, clean_settings)
        sentences = nltk.sent_tokenize(text)
        embeddings = utils.embed(sentences, embedding_model_name)
        embeddings_list.append(embeddings)
        sentences_list.append(sentences)

    results = []

    for (idx_a, idx_b) in combinations(range(len(embeddings_list)), 2):
        matrix = cosine_similarity(embeddings_list[idx_a], embeddings_list[idx_b])

        for i, row in enumerate(matrix):
            j = np.argmax(row)
            score = row[j]
            results.append({
                "source_a": sources[idx_a],
                "sentence_a": sentences_list[idx_a][i],
                "source_b": sources[idx_b],
                "nearest_sentence_b": sentences_list[idx_b][j],
                "similarity": round(float(score), 4)
            })

    return pd.DataFrame(results)
    
    #embeddings_list[i]
    # corresponds to the source
    #embedings_list[i][j]
    #corresponds to the sentence


    
def grand_tour_projection(text_passages: list, sources: list, embedding_model_name: str, clean_settings: dict) -> pd.DataFrame:
    embeddings_list = []
    all_sentences = []
    all_sources = []

    for i, source in zip(text_passages, sources):
        text = utils.clean_text(i, clean_settings)
        sentence = nltk.sent_tokenize(text)
        embeddings = utils.embed(sentence, embedding_model_name)
        embeddings_list.append(embeddings)
        all_sentences.extend(sentence)
        all_sources.extend([source] * len(sentence))

    all_embeddings = np.vstack(embeddings_list)

    df = pd.DataFrame(all_embeddings)
    df['sentences'] = all_sentences
    df['source'] = all_sources
    return df


def metrics(data_frame: pd.DataFrame) -> pd.DataFrame:
    
    copy_df = data_frame.copy()

    metrics = {
        "description": copy_df.describe(),
        "unique_sources": copy_df['source'].nunique(),
        "unique_sentences": copy_df['sentences'].nunique(),
        "sentence_counts_per_source": copy_df.groupby('source')['sentences'].count(),
        "centroid_per_source": copy_df.groupby('source').mean(numeric_only=True),
        "std_per_source": copy_df.groupby('source').std(numeric_only=True),
        "total_sentences": len(copy_df),
        "source_list": copy_df['source'].unique(),
    }

    return metrics


def scatter_plot(data_frame: pd.DataFrame, x_cordinate: str, y_cordinate: str, z_cordinate: str, dimensions: int, color_scale: str, reducer: str, embedding_model_name: str) -> px.scatter_3d:

    if dimensions == 5:
        color_cordinate = "dim4" 
        size_cordinate = "dim5"
    else:
        size_cordinate = None
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_pacmap = px.scatter_3d(
        data_frame,
        x=x_cordinate,
        y=y_cordinate,
        z=z_cordinate,
        color = color_cordinate,
        color_continuous_scale=color_scale,
        symbol = 'source',
        size=size_cordinate,
        hover_data=['sentences', 'source'],
        title=f'3D Scatter Plot - Reducer: {reducer} | Embedding Model: {embedding_model_name}',
        height=800,
        width=1200
    )

    fig_pacmap.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
        )

    if dimensions == 4:
        fig_pacmap.update_layout(
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
            margin=dict(r=80)
        )
        fig_pacmap.update_traces(
            marker=dict(
                colorbar=dict(
                    x=1.15,
                    y=0.5,
                    len=0.75,
                    thickness=18
                )
            )
        )   

    return fig_pacmap

def line_plot(data_frame: pd.DataFrame, x_cordinate: str, y_cordinate: str, z_cordinate: str, dimensions: int, reducer: str, embedding_model_name: str) -> px.scatter_3d:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_pacmap = px.line_3d(
        data_frame,
        x=x_cordinate,
        y=y_cordinate,
        z=z_cordinate,
        color = color_cordinate,
        symbol = 'source',
        hover_data=['sentences', 'source'],
        title=f'3D Line Plot - Reducer: {reducer} | Embedding Model: {embedding_model_name}',
        height=800,
        width=1200
    )

    fig_pacmap.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
        )

    if dimensions == 4:
        fig_pacmap.update_layout(
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
            margin=dict(r=80)
        )
        fig_pacmap.update_traces(
            marker=dict(
                colorbar=dict(
                    x=1.15,
                    y=0.5,
                    len=0.75,
                    thickness=18
                )
            )
        )   

    return fig_pacmap

def scatter_matrix(data_frame: pd.DataFrame, dimensions: int, color_scale: str, reducer: str, embedding_model_name: str) -> px.scatter_matrix:
    if dimensions == 4: color_cordinate = "dim4"
    if dimensions == 3: color_cordinate = "source"

    fig_matrix = px.scatter_matrix(
    data_frame,
    dimensions=['dim1', 'dim2', 'dim3'],
    color=color_cordinate,
    color_continuous_scale=color_scale, 
    symbol='source',
    title=f'Scatter Matrix - Reducer: {reducer} | Embedding Model: {embedding_model_name}',
    height=1000,
    width=1200,
    )

    fig_matrix.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.05,
            xanchor="left",
            x=1
        ),
        margin=dict(t=100, r=100, b=100, l=100)
    )

    if dimensions == 4:
        fig_matrix.update_layout(
            legend=dict(x=1, y=1.1, xanchor="left", yanchor="top"),
            coloraxis_colorbar=dict(x=1.08, y=0.5, len=0.8),
        )



    
    return fig_matrix
    




def cone_plot(data_frame: pd.DataFrame, color_scale: str, sizem_mode: str, size_ref: float, dimensions: int, reducer: str, embedding_model_name: str) -> go.Figure:


    fig = go.Figure(data=go.Cone(
        x=data_frame['dim1'],
        y=data_frame['dim2'],
        z=data_frame['dim3'],
        u=data_frame['dim4'],
        v=data_frame['dim5'],
        w=data_frame['dim6'],
        colorscale=color_scale,
        showscale= True,
        colorbar= dict(title="Magnitude"),
        sizemode=sizem_mode,
        sizeref=size_ref,
        customdata=data_frame[['source', 'sentences']].to_numpy(),
        hovertemplate=(
            "Source: %{customdata[0]}<br>"
            "Sentence: %{customdata[1]}<br>"
            "x: %{x:.2f}<br>"
            "y: %{y:.2f}<br>"
            "z: %{z:.2f}<br>"
            "u: %{u:.2f}<br>"
            "v: %{v:.2f}<br>"
            "w: %{w:.2f}<br>"
        )
    ))

    fig.update_layout(
        height=1200,
        width=1500,
        title=f'3D Cone Plot - Reducer: {reducer} | Embedding Model: {embedding_model_name}',
        paper_bgcolor="#161b22",
        scene=dict(
            xaxis_title="dim1",
            yaxis_title="dim2",
            zaxis_title="dim3"
        )
    )

    return fig

# def grand_tour_scatter_plot(data_frame: pd.DataFrame, embedding_model_name: str) -> px.scatter_3d:

#     frames = []
#     n_dims = len([col for col in data_frame.columns if isinstance(col, int)])
#     for i in range(n_dims - 2):
#         temp = data_frame[['sentences', 'source']].copy()
#         temp['x'] = data_frame[i].values
#         temp['y'] = data_frame[i + 1].values
#         temp['z'] = data_frame[i + 2].values
#         temp['frame'] = i
#         frames.append(temp)

#     animation_df = pd.concat(frames, ignore_index=True)

#     low = animation_df.select_dtypes(include='number').drop(columns=['frame']).min().min()
#     high = animation_df.select_dtypes(include='number').drop(columns=['frame']).max().max()

#     fig = px.scatter_3d(
#         animation_df,
#         x='x',
#         y='y',
#         z='z',
#         animation_frame='frame',
#         color='source',
#         symbol='source',
#         hover_data=['sentences', 'source'],
#         title=f'3D Scatter Plot Embedding Model: {embedding_model_name}',
#         height=800,
#         width=1200,
#         range_x=[low, high],
#         range_y=[low, high],
#         range_z=[low, high],
#     )

#     fig.update_layout(
#         paper_bgcolor="#161b22",
#         plot_bgcolor="#161b22"
#     )

#     fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 50
#     fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 30
#     # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['redraw'] = False
    

#     return fig

# @st.cache_data
def grand_tour_scatter_plot(data_frame: pd.DataFrame, embedding_model_name: str, color_scale: str, frame_duration: int, transition_duration: int, easing: str) -> px.scatter:

    frames = []
    n_dims = len([col for col in data_frame.columns if isinstance(col, int)])
    for i in range(n_dims - 2):
        temp = data_frame[['sentences', 'source']].copy()
        temp['x'] = data_frame[i].values
        temp['y'] = data_frame[i + 1].values
        temp['z'] = data_frame[i + 2].values
        temp['frame'] = i
        frames.append(temp)

    animation_df = pd.concat(frames, ignore_index=True)

    low = animation_df.select_dtypes(include='number').drop(columns=['frame']).min().min()
    high = animation_df.select_dtypes(include='number').drop(columns=['frame']).max().max()

    fig = px.scatter(
        animation_df,
        x='x',
        y='y',
        animation_frame='frame',
        animation_group='sentences',
        color='z',
        symbol='source',
        hover_data=['sentences', 'source'],
        title=f'Grand Tour Embedding Model: {embedding_model_name}',
        height=800,
        width=1200,
        range_x=[low, high],
        range_y=[low, high],
        size_max=20,
        color_continuous_scale=color_scale
    )

    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
            legend=dict(
                orientation="h",
                x=0,
                y=0.95,
                xanchor="left",
                yanchor="bottom"
            )
    )



    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = transition_duration
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['redraw'] = False
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['easing'] = easing
    fig.layout.updatemenus[0].buttons[0].args[1]['fromcurrent'] = True

    return fig
