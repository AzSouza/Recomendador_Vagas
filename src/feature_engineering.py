from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_skills(df, skill_list):
    for skill in skill_list:
        df[f'skill_{skill}'] = df['resume_clean'].apply(lambda txt: int(skill in txt))
    return df

def vectorize_resumes(df, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['resume_clean'])
    return X, tfidf

def job_applicant_matrix(vagas_df, applicants_df, vectorizer):
    # matrizes TF-IDF para vagas e candidatos
    X_job = vectorizer.transform(vagas_df['descricao_limpa'])
    X_app, _ = vectorize_resumes(applicants_df, max_features=vectorizer.max_features)
    # similaridade cosseno
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(X_app, X_job)
