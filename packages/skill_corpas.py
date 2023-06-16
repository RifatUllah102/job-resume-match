import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import os

class Skill_Corpus:
    def __init__(self):
        self.technical_skills = pd.read_excel(os.getcwd() + "/data/Technology_Skills.xlsx", sheet_name='Technology Skills', na_values='n/a')
        # Load the spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize a PhraseMatcher
        self.matcher = PhraseMatcher(self.nlp.vocab)

    def create_corpus(self):
        # Extract the skills and commodity titles from the DataFrame
        skills_data = self.technical_skills["Skills"].tolist()
        titles_data = self.technical_skills["Commodity Title"].tolist()

        # Sample dataset
        data = list(zip(skills_data, titles_data))
        # Create a list to store the skills
        skills_corpus = []

        # Iterate over the data and add skills to the PhraseMatcher
        for skill, title in data:
            doc = self.nlp(skill)
            self.matcher.add(title, None, doc)
            skills_corpus.append(skill)

        return skills_corpus

