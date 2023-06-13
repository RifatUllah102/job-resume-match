from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from gensim.models import Word2Vec

MAX_EMBEDDING_LENGTH = 500  # Maximum length for embeddings

class Embedder:
    def __init__(self):
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

    def bert_embedding(self, text):
        # Tokenize input text
        tokens = self.TOKENIZER.tokenize(text)

        # Split tokens into segments of maximum sequence length
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

        # Initialize a list to store segment embeddings
        segment_embeddings = []

        for segment in segment_tokens:
            # Add special tokens for BERT input
            segment_with_special_tokens = ['[CLS]'] + segment + ['[SEP]']

            # Convert tokens to tensor
            indexed_tokens = self.TOKENIZER.convert_tokens_to_ids(segment_with_special_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])

            # Obtain attention mask
            attention_mask = torch.ones_like(tokens_tensor)

            # Obtain BERT embeddings
            with torch.no_grad():
                self.MODEL.eval()
                outputs = self.MODEL(tokens_tensor, attention_mask=attention_mask)
                embeddings = outputs[0].squeeze(0).numpy()  # Reshape embeddings to 2D

            segment_embeddings.append(embeddings)

        # Concatenate segment embeddings
        embeddings = np.concatenate(segment_embeddings, axis=0)

        return torch.tensor(embeddings), embeddings

    def word2vec_embedding(self, text):
        tokens = text.split()  # Tokenize text into words
        embeddings = [self.word2vec.wv[word] for word in tokens if word in self.word2vec.wv]
        embeddings = np.array(embeddings)
        return torch.tensor(embeddings), embeddings

    def embedding(self, text):
        # Obtain BERT embeddings
        bert_embeddings, _ = self.bert_embedding(text)

        # Obtain Word2Vec embeddings
        word2vec_embeddings, _ = self.word2vec_embedding(text)

        # Concatenate BERT and Word2Vec embeddings
        combined_embeddings = torch.cat((bert_embeddings, word2vec_embeddings), dim=1)
        return combined_embeddings
