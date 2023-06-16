import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import spacy

nlp = spacy.load('en_core_web_sm')

class ModelBuilder:
    def __init__(self):
        self.directory = os.getcwd()

    def build_doc2vec(self):
        data = pd.read_csv(self.directory + '/data/UpdatedResumeDataSet.csv')
        cv_data = list(data['Resume'])
        tagged_data = [
            TaggedDocument(words=[token.text.lower() for token in nlp(_d)], tags=[str(i)])
            for i, _d in enumerate(cv_data)
        ]
        model = Doc2Vec(
            vector_size=300,
            min_count=3,
            epochs=50,
            window=3,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            dm=1,  # Use "distributed memory" (PV-DM) mode
            hs=1   # Use hierarchical softmax for training efficiency
        )
        model.build_vocab(tagged_data)
        model.train(
            tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )
        model.save('doc2vec.model')
        return 'Model Saved'
