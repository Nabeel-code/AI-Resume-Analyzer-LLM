AI Resume Analyzer (LLM + ATS Checker)

A smart, LLM-powered resume analysis tool that evaluates resumes against job descriptions using NLP, embeddings, and machine learning. Built with Python, Streamlit, spaCy, Scikit-learn, and Sentence Transformers.

🚀 Features

Resume Text Extraction
Supports PDF, DOCX, and TXT resume formats.

JD Parsing & Skill Extraction
Extracts required skills, keywords, and role expectations from any job description.

Semantic Similarity using BERT Embeddings
Compares resume content and JD using transformer-based embeddings.

ATS Keyword Match Score
Highlights missing keywords and role-specific gaps.

Recommendation Engine
Suggests improvements to match industry standards.

Interactive UI (Streamlit)
Clean, responsive app UI for quick resume evaluations.

🧠 Tech Stack

Python

Streamlit

spaCy (NLP parsing)

Sentence Transformers

Scikit-learn

PyTorch

docx2txt

PDF parsing

📁 Project Structure
├── app/
│   └── streamlit_app.py
├── src/
│   ├── resume_parser.py
│   ├── jd_parser.py
│   ├── similarity.py
│   └── recommender.py
├── assets/
│   └── screenshots/
│         ├── screenshot_home.png
│         ├── screenshot_analysis.png
│         └── screenshot_report.png
├── utils/
│   ├── text_cleaning.py
│   └── helpers.py
├── requirements.txt
└── README.md

🖼 Screenshots
Home Screen

Resume Analysis Output

Generated Report

▶️ Running Locally
1. Create venv
python -m venv venv
.\venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app/streamlit_app.py

📌 Future Improvements

Add AI-powered resume rewriting

Add PDF export of ATS report

Add support for multiple resume versions

Integrate a vector database for better skill ranking

📜 License

MIT License
