from transformers import BertTokenizer, BertModel
import torch
import numpy as np

MAX_EMBEDDING_LENGTH = 500  # Maximum length for embeddings

class Embedder:
    def __init__(self):
        self.TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MODEL = BertModel.from_pretrained('bert-base-uncased')

    def bert_embedding(self, text):
        # Tokenize input text
        tokens = self.TOKENIZER.tokenize(text)

        # Split tokens into segments of maximum sequence length
        max_length = MAX_EMBEDDING_LENGTH - 2  # Account for [CLS] and [SEP] tokens
        segment_tokens = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]

        # Initialize a list to store segment embeddings
        segment_embeddings = []

        for segment in segment_tokens:
            # Add special tokens for BERT input
            segment_with_special_tokens = ['[CLS]'] + segment + ['[SEP]']

            # Convert tokens to tensor
            indexed_tokens = self.TOKENIZER.convert_tokens_to_ids(segment_with_special_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])

            # Obtain BERT embeddings
            with torch.no_grad():
                self.MODEL.eval()
                outputs = self.MODEL(tokens_tensor)
                embeddings = outputs[0].squeeze(0).numpy()  # Reshape embeddings to 2D

            segment_embeddings.append(embeddings)

        # Concatenate segment embeddings
        embeddings = np.concatenate(segment_embeddings, axis=0)

        return embeddings
