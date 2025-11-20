import spacy

nlp = spacy.load("en_core_web_sm")

# smart keyword extractor
def extract_relevant_phrases(text):
    doc = nlp(text.lower())
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and len(token.text) > 2:
            keywords.add(token.lemma_)

    return keywords

# match score
def compute_match_score(resume_kw, jd_kw):
    if len(jd_kw) == 0:
        return 0
    score = (len(resume_kw.intersection(jd_kw)) / len(jd_kw)) * 100
    return round(score, 2)

# suggestions
def missing_keywords(resume_kw, jd_kw):
    blacklist = {
        "india", "including", "links", "growth",
        "action", "teams", "enjoys", "opportunity",
        "environment", "culture"
    }

    missing = jd_kw - resume_kw
    final = [w for w in missing if w not in blacklist and len(w) > 3]
    return final
