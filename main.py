import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
from gensim.models import Word2Vec
from fastapi import FastAPI
import numpy as np
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import concurrent.futures
import joblib

MAX_EMBEDDING_LENGTH = 500  # Maximum length for embeddings

nlp = spacy.load('en_core_web_sm')

import os

torch.cuda.empty_cache()

directory = os.getcwd()
app = FastAPI()

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased')
JD_CACHE_FILE = "jd_cache.joblib"

def preprocessing(pdf):
    with pdfplumber.open(pdf) as pdf_file:
        pages = pdf_file.pages
        text_pages = [page.extract_text() for page in pages]

    text = ' '.join(text_pages)
    file_clear = text.replace("\n", "")


    # Tokenize the text using SpaCy tokenizer
    doc = nlp(file_clear)

    # Initialize a list to store preprocessed tokens
    preprocessed_tokens = []

    for token in doc:
        # Convert token to lowercase
        token_text = token.text.lower()

        # Remove punctuation and non-alphabetic tokens
        if token_text not in string.punctuation and token.is_alpha:
            # Remove stopwords and perform lemmatization
            if token_text not in STOP_WORDS:
                token_lemma = token.lemma_
                preprocessed_tokens.append(token_lemma)

    # Lemmatization
    preprocessed_text = ' '.join(preprocessed_tokens)
    return preprocessed_text

def rank_cosine(jd, cv):
    Match_Test = [jd, cv]
    vector = TfidfVectorizer(
        ngram_range=(1, 3),                 # Include both unigrams and bigrams
        sublinear_tf=True,                  # Use sublinear term frequency scaling
        max_features=1000                   # Set the maximum number of features
    )
    count_matrix = vector.fit_transform(Match_Test)
    MatchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    MatchPercentage = round(MatchPercentage, 2)
    return MatchPercentage

def rank_wmd(jd, cv):
    # Prepare documents as lists of words
    jd_doc = jd.split()
    cv_doc = cv.split()

    # Build Word2Vec models for the documents
    model = Word2Vec([jd_doc, cv_doc], min_count=1)

    # Calculate the WMD
    distance = model.wv.wmdistance(jd_doc, cv_doc, lambda x, y: model.wv.distance(x, y))

    # Convert distance to match percentage
    match_percentage = round(100 - distance, 2)
    return match_percentage

def bert_embedding(text):
    # Tokenize input text
    tokens = TOKENIZER.tokenize(text)

    # Split tokens into segments of maximum sequence length
    max_length = MAX_EMBEDDING_LENGTH - 2  # Account for [CLS] and [SEP] tokens
    segment_tokens = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]

    # Initialize a list to store segment embeddings
    segment_embeddings = []

    for segment in segment_tokens:
        # Add special tokens for BERT input
        segment_with_special_tokens = ['[CLS]'] + segment + ['[SEP]']

        # Convert tokens to tensor
        indexed_tokens = TOKENIZER.convert_tokens_to_ids(segment_with_special_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Obtain BERT embeddings
        with torch.no_grad():
            MODEL.eval()
            outputs = MODEL(tokens_tensor)
            embeddings = outputs[0].squeeze(0).numpy()  # Reshape embeddings to 2D

        segment_embeddings.append(embeddings)

    # Concatenate segment embeddings
    embeddings = np.concatenate(segment_embeddings, axis=0)

    return embeddings

def rank_bert(jd, cv):
    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(jd, cv)[0][0]
    match_percentage = round(similarity * 100, 2)
    return match_percentage

def rank_combined(cosine_score, wmd_score, bert_score, wmd_weight=0.35, bert_weight=0.35):
    # Calculate combined score using weighted average
    combined_score = (wmd_weight * wmd_score) + (bert_weight * bert_score) + ((1 - (wmd_weight+bert_weight)) * cosine_score)
    combined_score = round(combined_score, 2)
    return combined_score

def process_cv(jd_preprocessing, jd_embeddings, cv_file):
    cv = directory + "/CV/" + cv_file
    cv_preprocessing = preprocessing(cv)
    cv_embeddings = bert_embedding(cv_preprocessing)
    cosine_score = rank_cosine(jd_preprocessing, cv_preprocessing)
    wmd_score = rank_wmd(jd_preprocessing, cv_preprocessing)
    bert_score = rank_bert(jd_embeddings, cv_embeddings)
    score = rank_combined(cosine_score, wmd_score, bert_score, 0.40, 0.50)
    return score

def load_cached_jd_data():
    if os.path.isfile(JD_CACHE_FILE):
        return joblib.load(JD_CACHE_FILE)
    else:
        return None

def save_jd_data_to_cache(data):
    joblib.dump(data, JD_CACHE_FILE)

def get_jd_data():
    cached_data = load_cached_jd_data()
    if cached_data is not None:
        return cached_data
    else:
        jd = directory + "/JD/BusinessAnalyst.pdf"
        jd_preprocessing = preprocessing(jd)
        jd_embeddings = bert_embedding(jd_preprocessing)
        jd_data = (jd_preprocessing, jd_embeddings)
        save_jd_data_to_cache(jd_data)
        return jd_data

@app.get("/rank")
def get_score():
    jd_preprocessing, jd_embeddings = get_jd_data()
    folder = os.listdir(directory + "/CV")

    score_list = []
    batch_size = 4  # Choose an appropriate batch size

    with concurrent.futures.ProcessPoolExecutor() as executor:
        batch = []
        for cv_file in folder:
            batch.append(cv_file)
            if len(batch) == batch_size:
                futures = [executor.submit(process_cv, jd_preprocessing, jd_embeddings, cv_file) for cv_file in batch]
                for future, cv_file in zip(futures, batch):
                    try:
                        score = future.result()
                        score_list.append({cv_file: score})
                    except Exception as e:
                        print(f"Error processing {cv_file}: {e}")
                batch = []

        if batch:
            # Process the remaining files in the last batch
            futures = [executor.submit(process_cv, jd_preprocessing, jd_embeddings, cv_file) for cv_file in batch]
            for future, cv_file in zip(futures, batch):
                try:
                    score = future.result()
                    score_list.append({cv_file: score})
                except Exception as e:
                    print(f"Error processing {cv_file}: {e}")

    return sorted(score_list, key=lambda x: list(x.values())[0], reverse=True)