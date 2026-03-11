import re
from cleantext import clean
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
import pacmap
import sys

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

def clean_text(text):
    cleaned_text = clean(text,
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


def embed(tokenized_sentences):
    if len(tokenized_sentences) < 3:
        print(tokenized_sentences)
        sys.exit("Error: The number of sentences must be greater than 2 for meaningful dimensionality reduction.")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    embeddings = model.encode(tokenized_sentences, show_progress_bar=True)
    return embeddings

def pca_reduce(embeddings, dimensions):
    pca = PCA(n_components=dimensions)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def umap_reduce(embeddings, dimensions, neighbors):
    umap_reduce = umap.UMAP(n_components=dimensions, n_neighbors=neighbors, metric='cosine', random_state=42, min_dist=0.1)
    umap_reduced_embeddings = umap_reduce.fit_transform(embeddings)
    return umap_reduced_embeddings

def tsne_reduce(embeddings, dimensions):
    tsne = TSNE(n_components=dimensions)
    tsne_reduced_embeddings = tsne.fit_transform(embeddings)
    return tsne_reduced_embeddings

def pacmap_reduce(embeddings, dimensions, neighbors):
    reducer = pacmap.PaCMAP(n_components=dimensions, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2, random_state=42, distance='angular', num_iters=1000, lr=1.0)
    pacmap_reduced_embeddings = reducer.fit_transform(embeddings,init='pca')
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
    