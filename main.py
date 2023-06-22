from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from packages.scoring import Scorer
from packages.model_builder import ModelBuilder
from packages.preprocessing import Preprocessor
from packages.resume_parser import CVParser
from packages.skill_corpas import SkillCorpus
import json
import os

class RankBody(BaseModel):
    jd_path: str
    cv_folder: str
    keyword: List[str]


# File path to store the skill corpus
skill_corpus_file = "skill_corpus.pkl"

app = FastAPI()
scorer = Scorer()
model_builder = ModelBuilder()
preprocessor = Preprocessor()
parser = CVParser(skill_corpus_file)

@app.post("/rank")
def get_score(body: RankBody):
    return scorer.get_score(body.jd_path, body.cv_folder, body.keyword)

@app.get("/build-model/doc2vec")
def get_score():
    return model_builder.build_doc2vec()

@app.get("/preprocessed")
def get_preprocessed():
    folder = os.listdir(os.getcwd() + "/CV")
    res = []
    for cv_file in folder:
        cv = os.getcwd() + "/CV/" + cv_file
        file = preprocessor.read_pdf(cv)
        res.append({
            cv_file: preprocessor.preprocess(file)
        })
    return res

@app.get("/resume-parser")
def get_resume_parser(filter):
    filter = json.loads(filter)
    folder = os.listdir(os.getcwd() + "/CV")
    res = []
    for cv_file in folder:
        cv = os.getcwd() + "/CV/" + cv_file
        res.append({
            cv_file: parser.parse_cv(cv, filter)
        })
    return res

@app.get("/save-skills")
def save_skillCorpus():
    # Create an instance of SkillCorpus
    skill_corpus = SkillCorpus()
    # Save the skill corpus to a file
    skill_corpus.save_corpus(skill_corpus_file)
    return 'OK'
