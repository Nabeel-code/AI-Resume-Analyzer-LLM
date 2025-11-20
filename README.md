AI Resume Analyzer – LLM-Powered Evaluation System

A smart, LLM-powered resume analysis tool that extracts content from resumes, evaluates skills, performs ATS scoring, finds gaps, matches with job descriptions, and provides improvement suggestions. Built with Streamlit and Python.

⭐ Features

PDF text extraction and cleaning

LLM-based resume content analysis

ATS scoring engine

Skill-gap detection

JD-to-resume similarity score

Clean Streamlit UI

Ready for deployment

📂 Project Structure
AI-Resume-Analyzer-LLM/
│
├── src/
│   ├── ats_scoring.py
│   ├── jd_matcher.py
│   ├── resume_parser.py
│   ├── text_cleaner.py
│   ├── text_utils.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│   └── screenshots/
│
├── requirements.txt
└── README.md

🧠 Tech Stack

Python

Streamlit

PyPDF2

Scikit-learn

Transformers / LLM APIs

NLTK / spaCy (optional)

🚀 Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Start the app
streamlit run app/streamlit_app.py

🔮 Future Enhancements

Vector-based scoring (embeddings)

Cross-encoder reranking

Job-specific scoring profiles

Batch resume analysis
