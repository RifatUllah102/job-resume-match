from fastapi import FastAPI
from packages.scoring import Scorer
from packages.model_builder import ModelBuilder

app = FastAPI()
scorer = Scorer()
model_builder = ModelBuilder()

@app.get("/rank")
def get_score():
    return scorer.get_score()

@app.get("/build-model/doc2vec")
def get_score():
    return model_builder.build_doc2vec()