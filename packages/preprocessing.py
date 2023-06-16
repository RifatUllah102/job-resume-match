import pdfplumber
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import sent_tokenize
from collections import Counter

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, pdf, hr_stopwords=[]):
        with pdfplumber.open(pdf) as pdf_file:
            pages = pdf_file.pages
            text_pages = [page.extract_text() for page in pages]

        text = ' '.join(text_pages)
        file_clear = text.replace("\n", "")

        # Remove extra whitespaces between words
        file_clear = re.sub(r'\s+', ' ', file_clear)

        # Split merged words using specific rules
        file_clear = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])|(?<=[A-Za-z0-9])(?=[A-Z]{2,})', ' ', file_clear)

        # Tokenize the text into sentences
        sentences = sent_tokenize(file_clear)

        # Initialize a list to store preprocessed sentences
        preprocessed_sentences = []

        for sentence in sentences:
            # Tokenize the sentence using SpaCy tokenizer
            doc = self.nlp(sentence)

            # Initialize a list to store preprocessed tokens
            preprocessed_tokens = []

            for token in doc:
                # Convert token to lowercase
                token_text = token.text.lower()

                # Remove punctuation and non-alphabetic tokens
                if token_text not in string.punctuation and token.is_alpha:
                    # Remove stopwords, perform lemmatization, and filter based on POS
                    if token_text not in STOP_WORDS and token.pos_ not in ['PUNCT', 'SYM']:
                        token_lemma = token.lemma_
                        preprocessed_tokens.append(token_lemma)

            # Add custom stopwords
            STOP_WORDS.update(hr_stopwords)

            # Lemmatization
            preprocessed_sentence = ' '.join(preprocessed_tokens)
            preprocessed_sentences.append(preprocessed_sentence)

        # Join the preprocessed sentences into a single text
        preprocessed_text = ' '.join(preprocessed_sentences)

        return preprocessed_text
