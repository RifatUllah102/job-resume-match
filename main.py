from fastapi import FastAPI
from packages.scoring import Scorer
from packages.model_builder import ModelBuilder
from packages.preprocessing import Preprocessor
from packages.resume_parser import CVParser
from packages.skill_corpas import Skill_Corpus

import os


app = FastAPI()
scorer = Scorer()
model_builder = ModelBuilder()
preprocessor = Preprocessor()
parser = CVParser()
corpus = Skill_Corpus()

@app.get("/rank")
def get_score():
    return scorer.get_score()

@app.get("/build-model/doc2vec")
def get_score():
    return model_builder.build_doc2vec()

@app.get("/preprocessed")
def get_preprocessed():
    folder = os.listdir(os.getcwd() + "/CV")
    res = []
    for cv_file in folder:
        cv = os.getcwd() + "/CV/" + cv_file
        res.append({
            cv_file: preprocessor.preprocess(cv)
        })
    return res

@app.get("/resume-parser")
def get_resume_parser():
    folder = os.listdir(os.getcwd() + "/CV")
    res = []
    for cv_file in folder:
        cv = os.getcwd() + "/CV/" + cv_file
        res.append({
            cv_file: parser.parse_cv(cv)
        })
    return res

@app.get("/skills")
def get_skills():
    return corpus.create_corpus()