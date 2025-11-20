from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_match_score(resume_kw, jd_kw):
    intersect = resume_kw.intersection(jd_kw)
    score = (len(intersect) / len(jd_kw)) * 100
    return round(score, 2)
