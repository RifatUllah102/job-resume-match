from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from numpy.linalg import norm
from packages.resume_parser import CVParser

skill_corpus_file = "skill_corpus.pkl"

class Ranker:
    def __init__(self):
        try:
            self.word2vec = Word2Vec(min_count=20,
                window=3,
                vector_size=300,
                sample=6e-5,
                alpha=0.03,
                min_alpha=0.0007,
                negative=20
            )
            self.parser = CVParser(skill_corpus_file)
        except Exception as e:
            print("Error occurred while initializing the Ranker:", str(e))

    def z_score_normalization(self, scores):
        try:
            mean = np.mean(scores)
            std = np.std(scores)
            normalized_scores = (scores - mean) / std
            return normalized_scores
        except Exception as e:
            print("Error occurred during z-score normalization:", str(e))
            return None

    def rank_keyword(self, cv, keyword):
        try:
            match_keyword = self.parser.extract_keyword(cv, keyword)
            rank = (len(match_keyword) / len(keyword)) * 100
            return round(rank, 2)
        except Exception as e:
            print("Error occurred during keyword ranking:", str(e))
            return None

    def rank_jaccard_keyword(self, cv, keyword):
        try:
            # Preprocess the documents
            cv = cv.lower().strip()
            keyword = [x.lower() for x in keyword]
            cv_words = set(cv.split())
            keyword_words = set(keyword)

            # Calculate the intersection and union of word sets
            intersection = cv_words.intersection(keyword_words)
            union = cv_words.union(keyword_words)

            # Calculate Jaccard similarity
            similarity = (len(intersection) / len(union)) *100
            return round(similarity, 2)
        except Exception as e:
            print("Error occurred during Jaccard keyword ranking:", str(e))
            return None

    def rank_jaccard(self, jd, cv):
        try:
            # Preprocess the documents
            jd = jd.lower().strip()
            cv = cv.lower().strip()
            jd_words = set(jd.split())
            cv_words = set(cv.split())

            # Calculate the intersection and union of word sets
            intersection = jd_words.intersection(cv_words)
            union = jd_words.union(cv_words)

            # Calculate Jaccard similarity
            similarity = (len(intersection) / len(union)) * 100
            return round(similarity, 2)
        except Exception as e:
            print("Error occurred during Jaccard ranking:", str(e))
            return None

    def rank_cosine(self, jd, cv):
        try:
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
        except Exception as e:
            print("Error occurred during cosine ranking:", str(e))
            return None

    def rank_wmd(self, jd, cv):
        try:
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
        except Exception as e:
            print("Error occurred during WMD ranking:", str(e))
            return None

    def rank_bert(self, jd, cv):
        try:
            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity(jd, cv)[0][0]
            match_percentage = round(similarity * 100, 2)
            return match_percentage
        except Exception as e:
            print("Error occurred during BERT ranking:", str(e))
            return None

    def rank_doc2vec(self, jd, cv):
        try:
            model = Doc2Vec.load('doc2vec.model')
            v1 = model.infer_vector(cv.split())
            v2 = model.infer_vector(jd.split())
            cosine_similarity = (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2))) * 100
            return round(float(cosine_similarity), 2)
        except Exception as e:
            print("Error occurred during Doc2Vec ranking:", str(e))
            return None

    def rank_combined(
            self,
            cosine_score,
            bert_score,
            doc2vec_score,
            keyword_score,
            wmd_score,
            bert_weight=0.25,
            doc2vec_weight=0.25):

        try:
            # Combine cosine score and keyword score with weights
            combined_cosine_keyword = (0.25 * cosine_score) + (0.75 * keyword_score)

            # Combine BERT score and WMD score with weights
            combined_bert_wmd = (0.90 * bert_score) + (0.10 * wmd_score)

            combined_score = (combined_bert_wmd * bert_weight) + (doc2vec_score * doc2vec_weight) + ((1 - (bert_weight+doc2vec_weight)) * combined_cosine_keyword)
            combined_score = round(combined_score, 2)
            return combined_score
        except Exception as e:
            print("Error occurred during combined ranking:", str(e))
            return None
