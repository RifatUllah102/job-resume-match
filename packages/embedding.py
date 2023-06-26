import torch
import numpy as np
import logging
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

MAX_EMBEDDING_LENGTH = 500

class Embedder:
    def __init__(self):
        try:
            self.TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
            self.MODEL = BertModel.from_pretrained('bert-base-uncased')
            self.word2vec = Word2Vec(min_count=20,
                                      window=3,
                                      vector_size=300,
                                      sample=6e-5,
                                      alpha=0.03,
                                      min_alpha=0.0007,
                                      negative=20
                                    )
        except Exception as e:
            logging.error("Error occurred while initializing the Embedder: %s", str(e))

    def bert_embedding(self, text):
        try:
            tokens = self.TOKENIZER.tokenize(text)
            segment_tokens = []
            current_segment = []
            current_segment_length = 0

            for token in tokens:
                current_segment.append(token)
                current_segment_length += len(token)

                if current_segment_length >= MAX_EMBEDDING_LENGTH - 2:
                    segment_tokens.append(current_segment)
                    current_segment = []
                    current_segment_length = 0

            if current_segment:
                segment_tokens.append(current_segment)

            segment_embeddings = []

            for segment in segment_tokens:
                segment_with_special_tokens = ['[CLS]'] + segment + ['[SEP]']
                indexed_tokens = self.TOKENIZER.convert_tokens_to_ids(segment_with_special_tokens)
                tokens_tensor = torch.tensor([indexed_tokens])

                with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    self.MODEL.eval()
                    if torch.cuda.is_available():
                        tokens_tensor = tokens_tensor.cuda()
                        self.MODEL = self.MODEL.cuda()
                    outputs = self.MODEL(tokens_tensor)
                    embeddings = outputs[0].squeeze(0).cpu().numpy()

                segment_embeddings.append(embeddings)

            embeddings = np.concatenate(segment_embeddings, axis=0)

            return torch.from_numpy(embeddings), embeddings
        except Exception as e:
            logging.error("Error occurred during BERT embedding: %s", str(e))
            return None, None

    def word2vec_embedding(self, text):
        try:
            tokens = text.split()
            embeddings = [self.word2vec.wv[word] for word in tokens if word in self.word2vec.wv]
            embeddings = np.array(embeddings)
            return torch.from_numpy(embeddings), embeddings
        except Exception as e:
            logging.error("Error occurred during Word2Vec embedding: %s", str(e))
            return None, None

    def embedding(self, text):
        try:
            bert_embeddings, _ = self.bert_embedding(text)
            word2vec_embeddings, _ = self.word2vec_embedding(text)

            if bert_embeddings is None or word2vec_embeddings is None:
                return None

            if torch.cuda.is_available():
                bert_embeddings = bert_embeddings.cuda()
                word2vec_embeddings = word2vec_embeddings.cuda()

            combined_embeddings = torch.cat((bert_embeddings, word2vec_embeddings), dim=1)
            return combined_embeddings
        except Exception as e:
            logging.error("Error occurred during embedding: %s", str(e))
            return None
