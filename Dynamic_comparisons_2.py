import plotly.express as px
import nltk
import utils
import pandas as pd
import numpy as np
import plotly.graph_objects as go




def produce_dataframe(text_passages: list, sources: list, reducer: str, settings: dict, embedding_model_name: str) -> pd.DataFrame:
    embeddings_list = []
    all_sentences = []
    all_sources = []

    for i, source in zip(text_passages, sources):
        text = utils.clean_text(i)
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

def metrics(data_frame: pd.DataFrame) -> pd.DataFrame:
    
    copy_df = data_frame.copy()

    metrics = {
        "description": copy_df.describe(),
        "missing_values": copy_df.isnull().sum(),
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

