# Recomendador de Candidatos

Este projeto implementa um **MVP** de recomendação de candidatos para vagas de TI, usando:

- **NLP** (TF-IDF + similaridade cosseno) para casar descrições de vagas com currículos.
- **Streamlit** para criar uma interface web interativa.

---

## Estrutura do Projeto

```
project_root/
├── data/                      # JSONs originais (vagas, prospects, applicants)
│   ├── vagas.json
│   ├── prospects.json
│   └── applicants.json
├── src/                       # Código-fonte
│   ├── data_loader.py        # Carregamento e normalização dos JSONs
│   ├── preprocess.py         # Limpeza de texto dos currículos (cv_pt/cv_en)
│   ├── feature_engineering.py# Vetorização TF-IDF e similaridade de texto
│   ├── model.py              # Treino e avaliação de modelo simples (RandomForest)
│   └── app.py                # Interface Streamlit
├── requirements.txt          # Dependências Python
└── README.md                 # Este arquivo
```

---

## Pré-requisitos

- Python 3.8 ou superior
- Git (opcional)
- Conexão com a internet para baixar recursos NLTK pela primeira vez

---

## Instalação e Setup

1. **Clone** este repositório ou faça download dos arquivos:

   ```bash
   git clone https://seu-repo.git project_root
   cd project_root
   ```

2. **Crie e ative** um ambiente virtual (recomendado):

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   # Para desativaro venv
   deactivate
   ```

3. **Instale** as dependências:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Baixe** recursos do NLTK (executar uma vez):

   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Configure** os arquivos JSON em `data//`. Se usar Git LFS ou DVC, verifique se os ponteiros estão corretos e os dados foram baixados.

---

## Como Executar

### Interface Streamlit

```bash
cd "C:\Trabalhos\Phase_5_Fiap\P-s-Tech\fase 5"
streamlit run src/app.py

```

- Abra `http://localhost:8501` no navegador.
- **Sidebar**:
  - Selecione uma vaga (título limpo, sem sufixo numérico).
  - (Opcional) Marque **Treinar modelo agora** para reapreender a partir do histórico de `prospects`.
  - Filtre por **Skills obrigatórias** (Python, AWS, Docker, etc.).

### Script Manual de Treino

Caso queira gerar `model.pkl` e `vectorizer.pkl` fora do Streamlit:

```bash
python - <<EOF
from src.data_loader import load_all
from src.preprocess import preprocess_applicants
from src.feature_engineering import vectorize_resumes
from src.model import train_model
import pickle

# Carrega dados
data = load_all('data')
pros = data['prospects']
apps = preprocess_applicants(data['applicants'])

# Junta histórico e treina
pros_flat = pros.explode('prospects')
merged = pd.json_normalize(pros_flat['prospects']).merge(
    apps, left_on='codigo', right_on='infos_basicas.codigo_profissional'
)
X, vect = vectorize_resumes(merged)
y = merged['situacao_candidado'].apply(lambda s: 1 if str(s).lower() in ['hired','contratado'] else 0)
model = train_model(X.toarray(), y)

# Salva
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(vect, open('vectorizer.pkl','wb'))
print("Artefatos salvos.")
EOF
```

---

## Como Funciona

1. **data\_loader.py**: normaliza JSONs, convertendo dicionários em listas de registros.
2. **preprocess.py**: limpa `cv_pt` / `cv_en`, aplica stopwords e lemmatização.
3. **feature\_engineering.py**: vetoriza currículos e descrição de vagas via TF-IDF; calcula similaridade cosseno.
4. **model.py**: treina um `RandomForestClassifier` sobre o histórico de candidatos contratados vs. não.
5. **app.py**: UI em Streamlit que combina:
   - Seleção de vaga e extração de descrição.
   - Filtro por skills.
   - Treino/reload de modelo.
   - Ranking de candidatos por `match_score`.

---
