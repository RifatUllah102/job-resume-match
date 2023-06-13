import concurrent.futures
import os
import joblib

from .preprocessing import Preprocessor
from .embedding import Embedder
from .ranking import Ranker

class Scorer:
    def __init__(self):
        self.directory = os.getcwd()
        self.JD_CACHE_FILE = "jd_cache.joblib"
        self.preprocessor = Preprocessor()
        self.embedder = Embedder()
        self.ranker = Ranker()

    def process_cv(self, jd_preprocessing, jd_embeddings, cv_file):
        cv = self.directory + "/CV/" + cv_file
        cv_preprocessing = self.preprocessor.preprocess(cv)
        cv_embeddings = self.embedder.bert_embedding(cv_preprocessing)
        cosine_score = self.ranker.rank_cosine(jd_preprocessing, cv_preprocessing)
        wmd_score = self.ranker.rank_wmd(jd_preprocessing, cv_preprocessing)
        bert_score = self.ranker.rank_bert(jd_embeddings, cv_embeddings)
        score = self.ranker.rank_combined(cosine_score, wmd_score, bert_score, 0.30, 0.50)
        return score

    def load_cached_jd_data(self):
        if os.path.isfile(self.JD_CACHE_FILE):
            return joblib.load(self.JD_CACHE_FILE)
        else:
            return None

    def save_jd_data_to_cache(self, data):
        joblib.dump(data, self.JD_CACHE_FILE)

    def get_jd_data(self):
        cached_data = self.load_cached_jd_data()
        if cached_data is not None:
            return cached_data
        else:
            jd = self.directory + "/JD/BusinessAnalyst.pdf"
            # Add HR domain-specific terms to the stopwords
            hr_stopwords = [
                "business analyst",
                "data analysis",
                "requirements gathering",
                "data modeling",
                "process improvement",
                "project management",
                "communication skills",
                "problem-solving",
                "analytical skills",
                "critical thinking",
                "technical skills",
                "teamwork",
                "documentation",
                "presentation skills",
                "MS Excel",
                "SQL",
                "Tableau",
                "Power BI",
                "Jira",
                "Agile methodology",
                "SDLC",
                "data visualization",
                "data interpretation",
                "reporting",
                "freshers",
                "entry-level",
                "graduate",
                "internship",
                "trainee"
            ]

            jd_preprocessing = self.preprocessor.preprocess(jd, hr_stopwords)
            jd_embeddings = self.embedder.bert_embedding(jd_preprocessing)
            jd_data = (jd_preprocessing, jd_embeddings)
            self.save_jd_data_to_cache(jd_data)
            return jd_data

    def get_score(self):
        jd_preprocessing, jd_embeddings = self.get_jd_data()
        folder = os.listdir(self.directory + "/CV")

        score_list = []
        batch_size = 4  # Choose an appropriate batch size

        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch = []
            for cv_file in folder:
                batch.append(cv_file)
                if len(batch) == batch_size:
                    futures = [executor.submit(self.process_cv, jd_preprocessing, jd_embeddings, cv_file) for cv_file in batch]
                    for future, cv_file in zip(futures, batch):
                        try:
                            score = future.result()
                            score_list.append({cv_file: score})
                        except Exception as e:
                            print(f"Error processing {cv_file}: {e}")
                    batch = []

            if batch:
                # Process the remaining files in the last batch
                futures = [executor.submit(self.process_cv, jd_preprocessing, jd_embeddings, cv_file) for cv_file in batch]
                for future, cv_file in zip(futures, batch):
                    try:
                        score = future.result()
                        score_list.append({cv_file: score})
                    except Exception as e:
                        print(f"Error processing {cv_file}: {e}")

        return sorted(score_list, key=lambda x: list(x.values())[0], reverse=True)
