import streamlit as st
import pandas as pd
import pickle
import re

from src.preprocess import preprocess_applicants, clean_text
from src.data_loader import load_all
from src.feature_engineering import extract_skills, vectorize_resumes, job_applicant_matrix
from src.model import train_model


def main():
    st.title("Recomendador de Candidatos")

    # 1. Carrega e prepara os dados
    data = load_all('data')
    vagas_df = data['vagas']
    prospects_df = data['prospects']
    applicants_df = preprocess_applicants(data['applicants'])

    # 2. Limpa títulos removendo sufixo numérico e cria coluna limpa
    vagas_df['titulo_limpo'] = vagas_df['informacoes_basicas.titulo_vaga'] \
        .str.replace(r'\s*-\s*\d+$', '', regex=True)

    # 3. Constrói descrição combinada das vagas para similaridade
    vagas_df['descricao_raw'] = (
        vagas_df['informacoes_basicas.objetivo_vaga'].fillna('') + ' ' +
        vagas_df['perfil_vaga.principais_atividades'].fillna('') + ' ' +
        vagas_df['perfil_vaga.competencia_tecnicas_e_comportamentais'].fillna('')
    )
    vagas_df['descricao_limpa'] = vagas_df['descricao_raw'].apply(clean_text)

    # 4. Seletor de vaga: exibe título sem número
    escolha = st.sidebar.selectbox(
        "Escolha a vaga", vagas_df['titulo_limpo'].tolist()
    )
    vaga = vagas_df[vagas_df['titulo_limpo'] == escolha].iloc[0]
    job_index = vagas_df.index[vagas_df['titulo_limpo'] == escolha][0]
    job_code = vaga['informacoes_basicas.vaga_sap']

    st.subheader(f"Vaga selecionada: {escolha}")
    st.write(vaga.get('informacoes_basicas.objetivo_vaga', '— Sem descrição —'))

    # 5. Skills padrão a partir da descrição da vaga
    raw_skills = vaga.get('perfil_vaga.competencia_tecnicas_e_comportamentais', '')
    required = [s.strip().lower() for s in re.split('[,;]', raw_skills) if s.strip()]

    skills = ["python", "aws", "docker", "kubernetes", "terraform"]
    default_skills = [s for s in skills if s in required]

    st.sidebar.subheader("Refine por Skills")
    chosen = st.sidebar.multiselect(
        "Skills obrigatórias", skills, default=default_skills
    )


    applicants_df = extract_skills(applicants_df, skills)

    # 6. Treino / carregamento do modelo
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
        model = pickle.load(open('model.pkl', 'rb'))
        vect = pickle.load(open('vectorizer.pkl', 'rb'))

    # 7. Calcula similaridade usando a descrição limpa
    sim_matrix = job_applicant_matrix(vagas_df, applicants_df, vect)
    scores = sim_matrix[:, job_index]

    # 8. Filtra por skills escolhidas
    if chosen:
        mask = applicants_df[[f'skill_{s}' for s in chosen]].all(axis=1)
    else:
        mask = pd.Series(True, index=applicants_df.index)

    # 9. Gera ranking com score na última coluna
    resultado = applicants_df[mask].copy()
    resultado['match_score'] = scores[mask]
    display_df = (
        resultado.sort_values('match_score', ascending=False)
        [['infos_basicas.codigo_profissional', 'informacoes_pessoais.nome', 'match_score']]  # score last
        .head(10)
        .reset_index(drop=True)
    )
    st.sidebar.markdown("---")  # Linha de separação
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


if __name__ == '__main__':
    main()
