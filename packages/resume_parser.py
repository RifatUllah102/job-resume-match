from tika import parser
import re
import spacy
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from packages.skill_corpas import SkillCorpus

class CVParser:
    def __init__(self, filePath):
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self.skill_corpus = SkillCorpus()
        self.skills = self.skill_corpus.load_corpus(filePath)

    def parse_cv(self, pdf_path, keyword=[]):
        try:
            file_data = parser.from_file(pdf_path)
            text = file_data['content']
            cv = {
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'name': self.extract_name(text),
                'skills': self.extract_skills(text),
                'education': self.extract_education(text),
                'experience': self.extract_experience(text),
                'keyword': self.extract_keyword(text, keyword)
            }
            return cv
        except Exception as e:
            print(f"Error parsing CV: {str(e)}")
            return None

    def extract_email(self, string):
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        return r.findall(string)

    def extract_phone(self, string):
        r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
        phone_numbers = r.findall(string)
        return [re.sub(r'\D', '', num) for num in phone_numbers]

    def extract_name(self, resume_text):
        nlp_text = self.nlp(resume_text)

        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

        self.matcher.add('NAME', [pattern], on_match=None)

        matches = self.matcher(nlp_text)

        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text

    def extract_skills(self, resume_text):
        doc = self.nlp(resume_text)
        matches = self.skill_corpus.matcher(doc)
        matched_skills = []
        for match_id, start, end in matches:
            matched_skills.append(doc[start:end].text.lower())
        return set(matched_skills)

    def extract_grades(self, resume_text):
        grade_regex = r'(?:\d{1,2}\.\d{1,2})'
        return re.findall(grade_regex, resume_text)

    def extract_education(self, resume_text):
        nlp_text = self.nlp(resume_text)
        stop_words = set(stopwords.words('english'))

        education_degrees = [
            'BE', 'B.E.', 'B.E', 'BS', 'B.S', 'B.Sc',
            'ME', 'M.E', 'M.E.', 'MS', 'M.S',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
        ]

        sentences = [sent.text.strip() for sent in nlp_text.sents]
        education = {}

        for index, text in enumerate(sentences):
            for word in text.split():
                word = re.sub(r'[?|$|.|!|,]', r'', word)
                if word.upper() in education_degrees and word not in stop_words:
                    if index < len(sentences) - 1:
                        education[word] = text + ' ' + sentences[index + 1]
                    else:
                        education[word] = text

        extracted_education = []

        for degree, details in education.items():
            year_match = re.search(r'(((20|19)(\d{2})))', details)
            if year_match:
                extracted_education.append((degree, ''.join(year_match.group())))
            else:
                extracted_education.append(degree)

        return extracted_education

    def extract_experience(self, resume_text):
        sub_patterns = [
            '[A-Z][a-z]* [A-Z][a-z]* Private Limited',
            '[A-Z][a-z]* [A-Z][a-z]* Pvt. Ltd.',
            '[A-Z][a-z]* [A-Z][a-z]* Inc.',
            '[A-Z][a-z]* LLC',
        ]
        pattern = '({})'.format('|'.join(sub_patterns))
        experience = re.findall(pattern, resume_text)
        return experience

    def extract_keyword(self, resume_text, keyword):
        if len(keyword) > 0:
            keyword = [x.lower() for x in keyword]

        resume_text = re.sub(r'\s+', ' ', resume_text.replace("\n", " "))
        resume_text = resume_text.lower()
        nlp_text = self.nlp(resume_text)

        # removing stop words and implementing word tokenization
        tokens = [token.text for token in nlp_text if not token.is_stop]

        res = []

        # check for one-grams (example: python)
        for token in tokens:
            if token.lower() in keyword:
                res.append(token)

        # check for bi-grams and tri-grams (example: machine learning)
        for token in nlp_text.noun_chunks:
            token = token.text.lower().strip()
            if token in keyword:
                res.append(token)

        return [i.capitalize() for i in set([i.lower() for i in res])]
