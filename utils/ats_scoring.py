import re

def ats_score(text: str) -> int:
    if not text:
        return 0

    keywords = [
        "python", "machine learning", "data science", "sql", "deep learning",
        "nlp", "computer vision", "aws", "pytorch", "tensorflow",
        "llm", "streamlit", "docker", "git"
    ]

    score = 0
    text = text.lower()

    for kw in keywords:
        if kw in text:
            score += 1

    # Score out of 100
    return int((score / len(keywords)) * 100)
