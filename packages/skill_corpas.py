import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import os
import joblib

class SkillCorpus:
    def __init__(self):
        self.technical_skills = pd.read_excel(os.getcwd() + "/data/Technology_Skills.xlsx", sheet_name='Technology Skills', na_values='n/a')
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self.skills_corpus = self.create_corpus()

    def create_corpus(self):
        skills_data = self.technical_skills["Skills"].tolist()
        titles_data = self.technical_skills["Commodity Title"].tolist()
        data = list(zip(skills_data, titles_data))
        skills_corpus = []

        for skill, title in data:
            doc = self.nlp(skill)
            self.matcher.add(title, None, doc)
            skills_corpus.append(skill)

        return skills_corpus

    def save_corpus(self, file_path):
        joblib.dump(self.skills_corpus, file_path)

    @staticmethod
    def load_corpus(file_path):
        return joblib.load(file_path)