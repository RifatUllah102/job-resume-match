from fastapi import FastAPI
from packages.scoring import Scorer

app = FastAPI()
scorer = Scorer()

@app.get("/rank")
def get_score():
    return scorer.get_score()