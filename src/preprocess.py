import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP = set(stopwords.words('portuguese'))
LEMMA = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 2]
    return ' '.join(LEMMA.lemmatize(w) for w in tokens)

def preprocess_applicants(df: pd.DataFrame) -> pd.DataFrame:
    # preenche NaN com string vazia
    pt = df['cv_pt'].fillna('')
    en = df['cv_en'].fillna('')
    # usa pt se não vazio, senão en
    df['raw_cv'] = pt.where(pt.str.strip() != '', en)
    # limpa o texto
    df['resume_clean'] = df['raw_cv'].apply(clean_text)
    return df