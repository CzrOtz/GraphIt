import re
from cleantext import clean
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
import pacmap
import sys
import streamlit as st

def open_and_clean_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    cleaned_text = clean(raw_text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_numbers=False,
        lang='en',
    )

    cleaned_text = cleaned_text.replace("\\'", "'").replace('\\"', '"')
    cleaned_text = cleaned_text.replace("\\n", " ").replace("\\t", " ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def clean_text(text, settings):
    cleaned_text = clean(text,
        fix_unicode=settings['fix_unicode'],
        to_ascii=settings['to_ascii'],
        lower=settings['lower'],
        no_urls=settings['no_urls'],
        no_emails=settings['no_emails'],
        no_numbers=settings['no_numbers'],
        lang=settings['lang']
    )

    cleaned_text = cleaned_text.replace("\\'", "'").replace('\\"', '"')
    cleaned_text = cleaned_text.replace("\\n", " ").replace("\\t", " ")

    if settings['remove_brackets']:
        cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)

    if settings['remove_curly_braces']:
        cleaned_text = re.sub(r'\{.*?\}', '', cleaned_text)

    if settings['remove_parentheses']:
        cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)

    if settings['remove_special_characters']:
        cleaned_text = re.sub(r'[#*~^|<>]', '', cleaned_text)

    if settings['remove_extra_whitespace']:
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text



@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name, device='cuda')

def embed(tokenized_sentences, embedding_model_name):
    if len(tokenized_sentences) < 3:
        # raise ValueError("At least 3 sentences are required for embedding.")
        st.error("At least 3 sentences are required per text field for processing. Please add more sentences or remove an unused text field.")
        st.stop()
    model = load_embedding_model(embedding_model_name)
    embeddings = model.encode(tokenized_sentences, show_progress_bar=True)
    return embeddings



def pca_reduce_fully_tunable(embeddings, settings):
    
        
    pca = PCA(
        n_components=settings['n_components'],
        svd_solver=settings['svd_solver'],
        whiten=settings['whiten'],
        random_state=settings['random_state']
    )
    return pca.fit_transform(embeddings)



def umap_reduce_fully_tunable(embeddings, settings):
    
    umap_reduce = umap.UMAP(
        n_components=settings['n_components'],
        n_neighbors=settings['n_neighbors'],
        metric=settings['distance'],
        random_state=settings['random_state'],
        min_dist=settings['min_dist']
    )
    umap_reduced_embeddings = umap_reduce.fit_transform(embeddings)
    return umap_reduced_embeddings



def tsne_reduce_fully_tunable(embeddings, settings):
    if len(embeddings) >= settings['perplexity']:
        st.error(f"The number of total sences is {len(embeddings)}. Current perplexity is {settings['perplexity']}. Perplexity must be less than the total samples/sences. Please adjust the perplexity, or add more sentences to the text fields.")
        st.stop()

    tsne = TSNE(
        n_components=settings['n_components'],
        perplexity=settings['perplexity'],
        learning_rate=settings['learning_rate'],
        max_iter=settings['max_iter'],
        random_state=settings['random_state']
    )
    tsne_reduced_embeddings = tsne.fit_transform(embeddings)
    return tsne_reduced_embeddings



def pacmap_reduce_fully_tunable(embeddings, settings):
    

    reducer = pacmap.PaCMAP(
        n_components=settings['n_components'],
        n_neighbors=settings['n_neighbors'],
        MN_ratio=settings['MN_ratio'],
        FP_ratio=settings['FP_ratio'],
        lr=settings['lr'],
        num_iters=settings['num_iters'],
        distance=settings['distance'],
        random_state=settings['random_state'],
        apply_pca=settings['apply_pca']
    )
    pacmap_reduced_embeddings = reducer.fit_transform(embeddings,init=settings['init'])
    return pacmap_reduced_embeddings


def create_3d_dataframe(reduced_embeddings, sentences, multiplier,source):
    df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2', 'dim3'])
    df['sentences'] = sentences
    df['source'] = source
    df['dim1'] *= multiplier
    df['dim2'] *= multiplier
    df['dim3'] *= multiplier
    return df

def create_4d_dataframe(reduced_embeddings, sentences, multiplier, source):
    df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4'])
    df['sentences'] = sentences
    df['source'] = source
    df['dim1'] *= multiplier
    df['dim2'] *= multiplier
    df['dim3'] *= multiplier
    df['dim4'] *= multiplier
    return df

def create_5d_dataframe(reduced_embeddings, sentences, multiplier, source):
    df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])
    df['sentences'] = sentences
    df['source'] = source
    df['dim1'] *= multiplier
    df['dim2'] *= multiplier
    df['dim3'] *= multiplier
    df['dim4'] *= multiplier
    df['dim5'] *= multiplier
    df['dim5'] = df['dim5'].abs()
    return df
    
def create_6d_dataframe(reduced_embeddings, sentences, multiplier, source):
    df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6'])
    df['sentences'] = sentences
    df['source'] = source
    df['dim1'] *= multiplier
    df['dim2'] *= multiplier
    df['dim3'] *= multiplier
    df['dim4'] *= multiplier
    df['dim5'] *= multiplier
    df['dim6'] *= multiplier
    return df