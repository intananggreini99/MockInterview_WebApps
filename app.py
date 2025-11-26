import os, re
from datetime import datetime

import streamlit as st
import spacy
import torch
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

import psycopg2
from psycopg2.extras import RealDictCursor


# ===========================
# DATABASE CONFIG
# ===========================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5777"))
DB_NAME = os.getenv("DB_NAME", "mock_interview")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mock999")

UPLOAD_DIR = "uploaded_cvs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


def init_db():
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    address TEXT,
                    role TEXT,
                    linkedin TEXT,
                    cv_path TEXT,
                    hard_score NUMERIC,
                    soft_score NUMERIC,
                    tfidf_score NUMERIC,
                    bm25_score NUMERIC,
                    similarity_score NUMERIC,
                    pattern_score NUMERIC,
                    sentiment_label TEXT,
                    sentiment_score NUMERIC,
                    system_score NUMERIC,
                    interview_hard TEXT,
                    interview_soft TEXT,
                    hrd_score NUMERIC,
                    final_score NUMERIC,
                    status VARCHAR(20) DEFAULT 'Invalidated',
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)


# ===========================
# LOAD NLP MODELS
# ===========================
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


@st.cache_resource
def load_sentiment_model():
    model_id = "w11wo/indonesian-roberta-base-sentiment-classifier"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tok, mdl


@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


nlp = load_spacy()
sent_tok, sent_model = load_sentiment_model()
embed_model = load_embedder()
init_db()


# ===========================
# PREPROCESSING
# ===========================
def preprocess(text):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]


# ===========================
# SENTIMENT ANALYSIS
# ===========================
def compute_sentiment(text):
    inputs = sent_tok(text, return_tensors="pt", truncation=True, max_length=200)
    with torch.no_grad():
        out = sent_model(**inputs)
        probs = F.softmax(out.logits, dim=-1)[0]
        label_id = int(torch.argmax(probs))
    label = sent_model.config.id2label[label_id]
    return label, round(float(probs[label_id]) * 100, 2)


# ===========================
# SEMANTIC SIMILARITY
# ===========================
ROLE_IDEAL = {
    "Data Engineer (DE)": "mengelola etl pipeline python sql airflow",
    "Data Analyst (DA)": "analisis data sql excel tableau",
    "Data Scientist (DS)": "machine learning modelling data",
    "Machine Learning Engineer (MLE)": "deploy model docker kubernetes api"
}


def compute_similarity(text, role):
    emb = embed_model.encode([text, ROLE_IDEAL[role]], convert_to_tensor=True)
    sim = util.cos_sim(emb[0], emb[1]).item()
    return round((sim + 1) * 50, 2)


# ===========================
# LEXICAL EVIDENCE
# ===========================
vectorizer = TfidfVectorizer(ngram_range=(1,2))


def tfidf_score(text, role):
    docs = [ROLE_IDEAL[role], text]
    tfidf = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(sim * 100, 2)


def bm25_score(text, role):
    corpus = [ROLE_IDEAL[role].split()]
    bm25 = BM25Okapi(corpus)
    score = bm25.get_scores(text.split())[0]
    return round(min(100, score), 2)


# ===========================
# REGEX PATTERN
# ===========================
PATTERNS = {
    "years": r"\d+\s*(tahun|years|yrs)",
    "prod": r"\bproduction\b",
    "etl": r"\betl\b"
}


def pattern_score(text):
    total = sum(len(re.findall(p, text.lower())) for p in PATTERNS.values())
    return min(100, total * 15)


# ===========================
# RULE BASED HARD SKILL
# ===========================
ROLE_KEYWORDS = {
    "Data Engineer (DE)": {'sql':3, 'python':3, 'etl':3, 'airflow':3},
    "Data Analyst (DA)": {'sql':3, 'excel':2, 'tableau':2},
    "Data Scientist (DS)": {'python':3, 'ml':3, 'regression':2},
    "Machine Learning Engineer (MLE)": {'docker':2, 'api':2, 'kubernetes':2}
}


def rule_hard(tokens, role):
    sc = 0
    max_sc = sum(ROLE_KEYWORDS[role].values())
    for k,w in ROLE_KEYWORDS[role].items():
        if k in tokens:
            sc += w
    return round(100 * sc / max_sc, 2)


# ===========================
# SOFT SKILL LEXICON
# ===========================
SOFT_SKILLS = {
    "communication":["komunikasi","communicate","explain"],
    "teamwork":["team","tim","collaboration"],
    "leadership":["lead","leader","memimpin"]
}


def soft_skill_score(tokens):
    text = " ".join(tokens)
    score = 0
    for s in SOFT_SKILLS.values():
        score += min(3, sum(text.count(w) for w in s))
    return round(100 * score / (3 * len(SOFT_SKILLS)), 2)


TONE_WEIGHT = {"positive":1.1,"neutral":1.0,"negative":0.9}


def tone_multiplier(label):
    return TONE_WEIGHT.get(label.lower(), 1.0)


# ===========================
# SAVE DATABASE
# ===========================
def save_candidate(**data):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO candidates (
                    name,email,address,role,linkedin,cv_path,
                    hard_score,soft_score,tfidf_score,bm25_score,
                    similarity_score,pattern_score,
                    sentiment_label,sentiment_score,
                    system_score,interview_hard,interview_soft
                )
                VALUES (
                    %(name)s,%(email)s,%(address)s,%(role)s,%(linkedin)s,%(cv_path)s,
                    %(hard)s,%(soft)s,%(tfidf)s,%(bm25)s,
                    %(sim)s,%(pattern)s,
                    %(sent_label)s,%(sent_score)s,
                    %(system)s,%(ans_h)s,%(ans_s)s
                )
            """, data)


# ===========================
# MAIN UI
# ===========================
def main():
    st.set_page_config(page_title="Mock Interview - Interviewee", layout="wide")
    st.title("Mock Interview - Interviewee")

    st.sidebar.header("📌 Biodata")
    name = st.sidebar.text_input("Nama")
    email = st.sidebar.text_input("Email / WhatsApp")
    address = st.sidebar.text_area("Alamat")
    role = st.sidebar.selectbox("Bidang", list(ROLE_IDEAL.keys()))
    linkedin = st.sidebar.text_input("LinkedIn")
    cv_file = st.sidebar.file_uploader("Upload CV PDF", type=["pdf"])

    st.subheader("✍ Jawaban Hardskill")
    ans_hard = st.text_area("Pengalaman teknis anda")

    st.subheader("🧠 Jawaban Softskill")
    ans_soft = st.text_area("Pengalaman softskill anda")

    if st.button("🔍 Hitung Skor"):
        full = ans_hard + " " + ans_soft
        tokens = preprocess(full)

        sent_label, sent_score = compute_sentiment(full)
        sem = compute_similarity(full, role)
        tfidf = tfidf_score(full, role)
        bm25 = bm25_score(full, role)
        pat = pattern_score(full)
        rule = rule_hard(tokens, role)

        hard = round(0.30*rule + 0.20*tfidf + 0.20*bm25 + 0.20*sem + 0.10*pat,2)
        soft = round(min(100, soft_skill_score(tokens) * tone_multiplier(sent_label)),2)

        system = round(0.6*hard + 0.2*soft + 0.1*sent_score + 0.1*sem,2)

        cv_path = None
        if cv_file:
            fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M')}_{cv_file.name}"
            cv_path = os.path.join(UPLOAD_DIR, fname)
            with open(cv_path,"wb") as f: f.write(cv_file.read())

        save_candidate(
            name=name,email=email,address=address,role=role,
            linkedin=linkedin,cv_path=cv_path,
            hard=hard,soft=soft,tfidf=tfidf,bm25=bm25,
            sim=sem,pattern=pat,
            sent_label=sent_label,sent_score=sent_score,
            system=system,ans_h=ans_hard,ans_s=ans_soft
        )

        st.success("✅ Data tersimpan")
        st.metric("System Score", system)
        st.metric("Hard Skill", hard)
        st.metric("Soft Skill", soft)


if __name__ == "__main__":
    main()