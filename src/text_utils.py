# utils/text_utils.py
import re
import json
from typing import List, Tuple, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Minimal skill lexicon. Extend this list to taste.
DEFAULT_SKILL_LEXICON = {
    "programming": ["python", "java", "c++", "c#", "r", "scala"],
    "data": ["pandas", "numpy", "sql", "nosql", "spark", "hive", "pyarrow"],
    "ml_frameworks": ["scikit-learn", "sklearn", "tensorflow", "pytorch", "keras"],
    "deployment": ["docker", "kubernetes", "fastapi", "flask", "uvicorn"],
    "cloud": ["aws", "azure", "gcp", "s3", "ec2"],
    "mlop": ["airflow", "mlflow", "dvc", "kubeflow"],
    "nlp": ["transformers", "spacy", "nltk", "sentence-transformers"],
    "visualization": ["plotly", "matplotlib", "seaborn", "powerbi", "tableau"],
    "testing": ["pytest", "unittest"],
    "other": ["sqlalchemy", "rest", "api", "ci/cd", "containerized"]
}

# -------------------------
# Text cleaning / helpers
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    # remove non-printables
    text = re.sub(r"[^a-z0-9\-\_\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_relevant_phrases(text: str, top_n: int = 50) -> List[str]:
    text = clean_text(text)
    words = text.split()
    freq = {}
    for w in words:
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

# -------------------------
# Skill extraction
# -------------------------
def build_skill_set(lexicon: Dict[str, List[str]] = None) -> Dict[str, Set[str]]:
    if lexicon is None:
        lexicon = DEFAULT_SKILL_LEXICON
    out = {}
    for cat, words in lexicon.items():
        out[cat] = set([w.lower() for w in words])
    return out

SKILL_SET = build_skill_set()

def extract_skills(text: str, skill_set: Dict[str, Set[str]] = None) -> Dict[str, List[str]]:
    text = clean_text(text)
    if skill_set is None:
        skill_set = SKILL_SET
    found = {k: [] for k in skill_set.keys()}
    for cat, words in skill_set.items():
        for w in words:
            # exact substring or token match
            if f" {w} " in f" {text} " or text.startswith(w + " ") or text.endswith(" " + w):
                found[cat].append(w)
    # prune empties
    found = {k: sorted(list(set(v))) for k, v in found.items() if v}
    return found

# -------------------------
# Matching / similarity
# -------------------------
def tfidf_similarity(a: str, b: str) -> float:
    a = clean_text(a)
    b = clean_text(b)
    if len(a) < 10 or len(b) < 10:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([a, b])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
        return round(sim * 100, 2)
    except Exception:
        return 0.0

def keyword_overlap_score(resume_text: str, jd_text: str, skill_set: Dict[str, Set[str]] = None) -> Tuple[float, List[str]]:
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)
    if skill_set is None:
        skill_set = SKILL_SET
    # collect skill words found in JD and resume
    jd_skills = set()
    resume_skills = set()
    for cat, words in skill_set.items():
        for w in words:
            if f" {w} " in f" {jd_text} " or jd_text.startswith(w + " ") or jd_text.endswith(" " + w):
                jd_skills.add(w)
            if f" {w} " in f" {resume_text} " or resume_text.startswith(w + " ") or resume_text.endswith(" " + w):
                resume_skills.add(w)
    if not jd_skills:
        return 0.0, []
    matched = sorted(list(jd_skills & resume_skills))
    score = (len(matched) / len(jd_skills)) * 100
    return round(score, 2), matched

# -------------------------
# Combined scoring function
# -------------------------
def combine_scores(semantic_sim: float, tfidf_sim: float, keyword_sim: float,
                   weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)) -> float:
    """
    Inputs are percentages (0-100).
    weights sum to 1.0
    """
    s = (weights[0] * semantic_sim) + (weights[1] * tfidf_sim) + (weights[2] * keyword_sim)
    return round(float(s), 2)
