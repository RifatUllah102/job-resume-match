from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

class Ranker:
    def rank_cosine(self, jd, cv):
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

    def rank_wmd(self, jd, cv):
        # Prepare documents as lists of words
        jd_doc = jd.split()
        cv_doc = cv.split()

        # Build Word2Vec models for the documents
        model = Word2Vec(
            [jd_doc, cv_doc],
            min_count=1,
            window=3,
            vector_size=300,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20
        )

        # Calculate the WMD
        distance = model.wv.wmdistance(jd_doc, cv_doc, lambda x, y: model.wv.distance(x, y))

        # Convert distance to match percentage
        match_percentage = round(100 - distance, 2)
        return match_percentage

    def rank_bert(self, jd, cv):
        # Calculate cosine similarity between embeddings
        similarity = cosine_similarity(jd, cv)[0][0]
        match_percentage = round(similarity * 100, 2)
        return match_percentage

    def rank_combined(self, cosine_score, wmd_score, bert_score, wmd_weight=0.35, bert_weight=0.35):
        # Calculate combined score using weighted average
        combined_score = (wmd_weight * wmd_score) + (bert_weight * bert_score) + ((1 - (wmd_weight+bert_weight)) * cosine_score)
        combined_score = round(combined_score, 2)
        return combined_score
