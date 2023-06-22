# Resume Ranking Application

This is an application for ranking resumes (CVs) based on their similarity to a given job description (JD). The application uses various techniques, including natural language processing and machine learning, to analyze and compare resumes with the JD. It provides a score for each resume, indicating its relevance to the job description.

## Table of Contents

* Requirements
* Prerequisites
* Installation
* Usage
* API Endpoint
* Customization
* Contributing
* Acknowledgments
* License
* Contact

## Requirements
This project has the following requirements:

* Java 8 or higher
* Python 3.10 or higher

## Prerequisites
* Python 3.7 or higher
* pip package manager

## Installation

1. Clone the repository to your local machine:

```bash
  git clone https://github.com/RoomeyRahman/job-resume-match.git
```
2. Navigate to the project directory:
```bash
  cd resume-ranking
```
3. Create a virtual environment:
```bash
  python -m venv myenv
```
4. Activate the virtual environment:
* For Windows:
```bash
  myenv\Scripts\activate
```
* For macOS/Linux:
```bash
  source myenv/bin/activate
```
5. Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Usage

1. Prepare the job description (JD) file:

* Save the JD as a PDF file in the JD directory.
* The JD should be named BusinessAnalyst.pdf.

2. Prepare the resumes (CVs):

* Save the resumes as PDF files in the CV directory.
* Each resume should be a separate PDF file.
* Customize HR domain-specific stopwords (optional):

3. Open the preprocessing.py file.
* Modify the hr_stopwords list in the get_jd_data() method to * include or remove HR-specific terms relevant to your job description.
4. Start the application:
```bash
uvicorn main:app --reload
```
5. Access the API endpoint:

* Open your web browser and visit http://localhost:8000/rank.
* The API will return a JSON response containing the ranked scores for each resume.


## API Endpoint

The application provides a single API endpoint:

* Endpoint: /rank
* Method: GET
* Description: Retrieves the ranked scores for each resume based on their similarity to the job description (JD).
* Response: JSON format
Example response:
```json
[
  {
    "resume1.pdf": {
      "cosine_score": 83.24,
      "wmd_score": 71.89,
      "bert_score": 91.53,
      "score": 84.72
    }
  },
  {
    "resume2.pdf": {
      "cosine_score": 79.17,
      "wmd_score": 68.55,
      "bert_score": 89.72,
      "score": 80.67
    }
  },
  {
    "resume3.pdf": {
      "cosine_score": 85.62,
      "wmd_score": 74.23,
      "bert_score": 92.18,
      "score": 86.67
    }
  }
]
```


## Customization

* You can modify the preprocessing steps in preprocessing.py to suit your specific requirements. For example, you can add additional text cleaning or filtering steps.

* Adjust the parameters in embedding.py for the Word2Vec model and BERT model according to your preferences. You can change the vector size, window size, or other hyperparameters to optimize the embeddings.

* Fine-tune the ranking method in ranking.py to adjust the weights or add/remove similarity metrics based on your needs.


## Contributing
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request.


## Acknowledgments
* The application uses the pdfplumber library for extracting text from PDF files.

* The BERT embeddings are obtained using the Hugging Face Transformers library.

* The Word2Vec embeddings are trained using the Gensim library.



## License

his project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.


## Contact

For any questions or inquiries, please contact [roomeyrahman@gmail.com](mailto:roomeyrahman@gmail.como).




