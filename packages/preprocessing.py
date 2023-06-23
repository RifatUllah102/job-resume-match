import pdfplumber
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import sent_tokenize
from tika import parser
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = PorterStemmer()

    def read_pdf_pdfPlumber(self, pdf):
        try:
            with pdfplumber.open(pdf) as pdf_file:
                pages = pdf_file.pages
                text_pages = [page.extract_text() for page in pages]
            return ' '.join(text_pages)
        except Exception as e:
            print("Error occurred while reading PDF with pdfplumber:", str(e))
            return ''

    def read_pdf_tika(self, pdf):
        try:
            file_data = parser.from_file(pdf)
            text = file_data['content']
            return text
        except Exception as e:
            print("Error occurred while reading PDF with Tika:", str(e))
            return ''

    def read_pdf(self, pdf):
        return self.read_pdf_tika(pdf)

    def preprocess(self, text, hr_keyword=[]):
        try:
            # Remove newlines and extra whitespaces
            file_clear = re.sub(r'\s+', ' ', text.replace("\n", " "))

            # Add HR keywords
            file_clear = file_clear + ' ' + ' '.join(hr_keyword)

            # Convert text to lowercase
            file_clear = file_clear.lower()

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
                        # Remove stopwords, perform stemming, and filter based on POS
                        if token_text not in STOP_WORDS and token.pos_ not in ['PUNCT', 'SYM']:
                            token_stem = self.stemmer.stem(token_text)
                            preprocessed_tokens.append(token_stem)

                # Join preprocessed tokens into a sentence
                preprocessed_sentence = ' '.join(preprocessed_tokens)
                preprocessed_sentences.append(preprocessed_sentence)

            # Join preprocessed sentences into a single text
            preprocessed_text = ' '.join(preprocessed_sentences)

            # Remove duplicate words
            preprocessed_text = ' '.join(list(set(preprocessed_text.split())))

            return preprocessed_text
        except Exception as e:
            print("Error occurred during preprocessing:", str(e))
            return ''
