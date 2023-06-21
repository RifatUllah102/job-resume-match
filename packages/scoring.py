import concurrent.futures
import os

from .preprocessing import Preprocessor
from .embedding import Embedder
from .ranking import Ranker

class Scorer:
    def __init__(self):
        self.directory = os.getcwd()
        self.preprocessor = Preprocessor()
        self.embedder = Embedder()
        self.ranker = Ranker()

    def process_cv(self, jd_preprocessing, jd_embeddings, cv_file):
        cv = self.directory + "/CV/" + cv_file
        cv_text = self.preprocessor.read_pdf(cv)
        cv_preprocessing = self.preprocessor.preprocess(cv_text)
        cv_embeddings = self.embedder.embedding(cv_preprocessing)
        cosine_score = self.ranker.rank_cosine(jd_preprocessing, cv_preprocessing)
        wmd_score = self.ranker.rank_wmd(jd_preprocessing, cv_preprocessing)
        bert_score = self.ranker.rank_bert(jd_embeddings, cv_embeddings)
        doc2vec_score = self.ranker.rank_doc2vec(jd_preprocessing, cv_preprocessing)
        score = self.ranker.rank_combined(cosine_score, bert_score, doc2vec_score, 0.45, 0.10)
        return {
            'cosine_score': cosine_score,
            'bert_score': bert_score,
            'wmd_score': wmd_score,
            'doc2vec_score': doc2vec_score,
            'score': score
        }

    def get_jd_data(self):
        jd = self.directory + "/JD/BusinessAnalyst.pdf"
        # Add HR domain-specific terms to the stopwords
        hr_keyword = [
            "business analyst",
            "data analysis",
            "requirements gathering",
            "data modeling",
            "process improvement",
            "project management",
            "communication",
            "problem-solving",
            "analytical skills",
            "critical thinking",
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
            "User Acceptance Testing",
            "UAT",
            "BRD",
            "SRS",
            "UML Diagram",
            "Technical Documentation",
            "Trello",
            "Leadership",
            "Extra-curricular activities"
        ]

        jd_text = self.preprocessor.read_pdf(jd)
        jd_preprocessing = self.preprocessor.preprocess(jd_text, hr_keyword)
        jd_embeddings = self.embedder.embedding(jd_preprocessing)
        jd_data = (jd_preprocessing, jd_embeddings)
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
                            result = future.result()
                            score_list.append({cv_file: result})
                        except Exception as e:
                            print(f"Error processing {cv_file}: {e}")
                    batch = []

            if batch:
                # Process the remaining files in the last batch
                futures = [executor.submit(self.process_cv, jd_preprocessing, jd_embeddings, cv_file) for cv_file in batch]
                for future, cv_file in zip(futures, batch):
                    try:
                        result = future.result()
                        score_list.append({cv_file: result})
                    except Exception as e:
                        print(f"Error processing {cv_file}: {e}")

        return sorted(score_list, key=lambda x: list(x.values())[0]["score"], reverse=True)
