import streamlit as st
import pandas as pd
import pickle
import re

from src.data_loader import load_all
from src.preprocess import preprocess_applicants, clean_text
from src.feature_engineering import extract_skills, vectorize_resumes, job_applicant_matrix
from src.model import train_model

@st.cache_data
def carregar_dados():
    return load_all('data')

@st.cache_data
def preparar_applicants(applicants_raw):
    return preprocess_applicants(applicants_raw)

@st.cache_data
def preparar_vagas(vagas_raw):
    vagas_df = vagas_raw.copy()
    vagas_df['titulo_limpo'] = vagas_df['informacoes_basicas.titulo_vaga'] \
        .str.replace(r'\s*-\s*\d+$', '', regex=True)
    vagas_df['descricao_raw'] = (
        vagas_df['informacoes_basicas.objetivo_vaga'].fillna('') + ' ' +
        vagas_df['perfil_vaga.principais_atividades'].fillna('') + ' ' +
        vagas_df['perfil_vaga.competencia_tecnicas_e_comportamentais'].fillna('')
    )
    vagas_df['descricao_limpa'] = vagas_df['descricao_raw'].apply(clean_text)
    return vagas_df

@st.cache_resource
def carregar_modelos():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vect = pickle.load(f)
    return model, vect

@st.cache_data
def calcular_similaridade(vagas_df, applicants_df, _vect):
    return job_applicant_matrix(vagas_df, applicants_df, _vect)

def main():
    try:
        st.title("Recomendador de Candidatos")

        data = carregar_dados()
        prospects_df = data['prospects']
        vagas_df = preparar_vagas(data['vagas'])
        applicants_df = preparar_applicants(data['applicants'])

        if st.sidebar.checkbox("Modo Apresentação (limita a 150 candidatos)", value =True):
            applicants_df = applicants_df.head(150)

        escolha = st.sidebar.selectbox(
        "Escolha a vaga", sorted(vagas_df['titulo_limpo'].unique())
)
        vaga = vagas_df[vagas_df['titulo_limpo'] == escolha].iloc[0]
        job_index = vagas_df.index[vagas_df['titulo_limpo'] == escolha][0]

        st.subheader(f"Vaga selecionada: {escolha}")
        st.write(vaga.get('informacoes_basicas.objetivo_vaga', '— Sem descrição —'))

        raw_skills = vaga.get('perfil_vaga.competencia_tecnicas_e_comportamentais', '')
        required = [s.strip().lower() for s in re.split('[,;]', raw_skills) if s.strip()]

        skills = ["python", "aws", "docker", "kubernetes", "terraform"]
        default_skills = [s for s in skills if s in required]

        st.sidebar.subheader("Refine por Skills")
        chosen = st.sidebar.multiselect(
            "Skills obrigatórias", skills, default=default_skills
        )

        applicants_df = extract_skills(applicants_df, skills)

        if st.sidebar.checkbox("Treinar modelo agora"):
            pros_flat = pd.json_normalize(
                data['prospects'].explode('prospects')['prospects']
            )
            merged = pros_flat.merge(
                applicants_df,
                left_on='codigo',
                right_on='infos_basicas.codigo_profissional'
            )
            X, vect = vectorize_resumes(merged)
            y = merged['situacao_candidado'].apply(
                lambda s: 1 if str(s).lower() in ['hired', 'contratado'] else 0
            )
            model = train_model(X.toarray(), y)
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(vect, f)
            st.success("✅ Modelo treinado e salvo em artifacts")
        else:
            model, vect = carregar_modelos()

        sim_matrix = calcular_similaridade(vagas_df, applicants_df, vect)
        scores = sim_matrix[:, job_index]

        if chosen:
            mask = applicants_df[[f'skill_{s}' for s in chosen]].all(axis=1)
        else:
            mask = pd.Series(True, index=applicants_df.index)

        resultado = applicants_df[mask].copy()
        resultado['match_score'] = scores[mask]
        display_df = (
            resultado.sort_values('match_score', ascending=False)
            [['infos_basicas.codigo_profissional', 'informacoes_pessoais.nome', 'match_score']]
            .head(10)
            .reset_index(drop=True)
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Alunos:")
        alunos = [
            "André Azevedo de Souza (RM357748)",
            "Mateus Queiroz (RM357648)",
            "Marcelo Almeida (RM357258)",
        ]
        for aluno in alunos:
            st.sidebar.write(aluno)

        st.write("### Top candidatos")
        st.dataframe(display_df)

    except Exception as e:
        st.error("Erro ao executar o app:")
        st.exception(e)

if __name__ == '__main__':
    main()

