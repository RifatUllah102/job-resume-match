import pdfplumber
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, pdf, hr_stopwords=[]):
        with pdfplumber.open(pdf) as pdf_file:
            pages = pdf_file.pages
            text_pages = [page.extract_text() for page in pages]

        text = ' '.join(text_pages)
        file_clear = text.replace("\n", "")

        # Tokenize the text using SpaCy tokenizer
        doc = self.nlp(file_clear)

        # Initialize a list to store preprocessed tokens
        preprocessed_tokens = []

        for token in doc:
            # Convert token to lowercase
            token_text = token.text.lower()

            # Remove punctuation and non-alphabetic tokens
            if token_text not in string.punctuation and token.is_alpha:
                # Remove stopwords and perform lemmatization
                if token_text not in STOP_WORDS:
                    token_lemma = token.lemma_
                    preprocessed_tokens.append(token_lemma)

        STOP_WORDS.update(hr_stopwords)

        # Lemmatization
        preprocessed_text = ' '.join(preprocessed_tokens)
        return preprocessed_text
