# Datathon Passos Mágicos: Predição de Risco de Lacunas de Aprendizado

## Visão Geral do Projeto

Este projeto, desenvolvido no contexto de um Datathon de Pós-Graduação em Análise de Dados, aborda o desafio de identificar proativamente alunos em risco de desenvolver lacunas de aprendizado. Utilizando um conjunto de indicadores educacionais, o objetivo é fornecer uma ferramenta preditiva que apoie intervenções pedagógicas e psicossociais direcionadas. A principal inovação reside na **reconstrução do target de risco**, superando uma limitação crítica do modelo original para garantir predições consistentes e logicamente coerentes.

## Declaração do Problema

O modelo preditivo inicial apresentava inconsistências significativas, especialmente em cenários extremos (e.g., todos os inputs em 0 ou todos em 10), resultando em predições contraintuitivas. A investigação revelou que o target original (`risk_gap`) era fortemente correlacionado e, em grande parte, derivado de uma variável (`IAN`) que não estava disponível para entrada no aplicativo Streamlit. Essa dependência oculta gerava um desalinhamento fundamental entre o modelo e o ambiente de aplicação, comprometendo a confiabilidade das previsões.

## Abordagem da Solução: Reconstrução do Target

Para resolver o problema de inconsistência e desalinhamento, implementamos uma abordagem inovadora centrada na reconstrução do target de risco:

1.  **Estimação do Índice de Desenvolvimento Educacional (INDE):** O `INDE`, um índice composto que reflete o desempenho global do aluno, foi estimado a partir dos seis indicadores educacionais disponíveis no aplicativo (IDA, IEG, IAA, IPS, IPP, IPV) utilizando um modelo de regressão linear. Esta estimativa alcançou um R² de 0.91, garantindo uma representação fiel do `INDE` sem depender de variáveis externas ao app.
2.  **Definição do Novo Target (`risk_gap_v2`):** Com base no `INDE` estimado, um novo target binário (`risk_gap_v2`) foi criado. Um aluno é classificado com **alto risco (1)** se seu `INDE_estimado` for **inferior a 7.2** (a mediana do `INDE` no dataset), e **baixo risco (0)** caso contrário. Esta definição garante que o target seja semanticamente coerente com o conceito de risco de lacunas de aprendizado e diretamente derivável dos inputs do app.

Esta reconstrução do target foi o **diferencial crítico** que permitiu a criação de um modelo preditivo robusto e alinhado com as necessidades da aplicação.

## Descrição dos Datasets

O projeto utilizou dados educacionais fornecidos pela ONG 'Passos Mágicos'. Os principais arquivos incluem:

*   `data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`: Dados brutos do Datathon.
*   `data/raw/PEDE_PASSOS_DATASET_FIAP.csv`: Dados adicionais para o projeto.
*   `data/processed/processed_data.csv`: Dataset pré-processado utilizado para treinamento do modelo.

## Metodologia

### Análise Exploratória de Dados (EDA)

*   Análise de dados multi-ano para identificar tendências e padrões.
*   Identificação da forte correlação entre a variável `IAN` e o target `risk_gap` original, revelando a causa raiz da inconsistência do modelo anterior.

### Engenharia de Features

Além dos seis indicadores brutos (IDA, IEG, IAA, IPS, IPP, IPV), foram criadas três features derivadas, replicando a lógica do aplicativo:

*   `perception_gap`: Diferença entre Autoavaliação (IAA) e Desempenho Acadêmico (IDA).
*   `behavioral_score`: Média de Engajamento (IEG) e Suporte Psicossocial (IPS).
*   `relative_performance`: Desvio do Desempenho Acadêmico (IDA) em relação a um benchmark de 7.

### Modelagem Preditiva

#### Detalhes do Modelo

*   **Algoritmo:** Gradient Boosting Classifier.
*   **Features:** As 6 features brutas (`IDA`, `IEG`, `IAA`, `IPS`, `IPP`, `IPV`) mais as 3 features derivadas (`perception_gap`, `behavioral_score`, `relative_performance`), totalizando 9 features. O `INDE_estimado` foi usado para criar o target, mas não como feature de entrada do modelo, garantindo maior robustez.
*   **Estratégia de Avaliação:** O modelo foi avaliado utilizando métricas como ROC-AUC e F1-Score em um conjunto de teste estratificado. Um threshold de classificação de **0.35** foi escolhido para o target de risco, priorizando o *recall* (capacidade de identificar corretamente os alunos em risco) para minimizar falsos negativos, o que é crucial em contextos de intervenção educacional.

#### Validação de Casos Extremos

Um dos requisitos fundamentais foi garantir que o modelo se comportasse logicamente em cenários extremos:

*   **Todos os inputs = 0:** O modelo previu **100% de probabilidade de ALTO RISCO**.
*   **Todos os inputs = 10:** O modelo previu **0.01% de probabilidade de BAIXO RISCO**.

Esses resultados confirmam a consistência e a coerência do modelo reconstruído.

## Aplicação Streamlit

O projeto inclui uma aplicação interativa desenvolvida em Streamlit que permite aos usuários simular o risco de lacunas de aprendizado de um aluno ajustando os seis indicadores educacionais. A aplicação exibe a probabilidade de risco, a classificação (Alto/Baixo Risco) e uma análise detalhada dos fatores, fornecendo recomendações pedagógicas.

### Como Usar o Aplicativo

1.  Certifique-se de ter o Python e as dependências instaladas (veja `requirements.txt`).
2.  Navegue até o diretório raiz do projeto.
3.  Execute o comando: `streamlit run streamlit_app/app.py`
4.  A aplicação será aberta em seu navegador padrão.

## Estrutura do Repositório

```
. 
├── data/
│   ├── processed/
│   │   └── processed_data.csv
│   └── raw/
│       ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│       └── PEDE_PASSOS_DATASET_FIAP.csv
├── models/
│   └── model_risk_gap.pkl
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── streamlit_app/
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Como Rodar o Projeto

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/MatheusDiniz23/passos-magicos-datathon.git
    cd passos-magicos-datathon
    ```
2.  **Criar Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: .\venv\Scripts\activate
    ```
3.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Treinar o Modelo (Opcional, o modelo já está salvo):**
    ```bash
    python train_model.py
    ```
5.  **Executar o Aplicativo Streamlit:**
    ```bash
    streamlit run streamlit_app/app.py
    ```

## Tecnologias Utilizadas

*   **Python**
*   **Pandas:** Manipulação e análise de dados.
*   **NumPy:** Computação numérica.
*   **Scikit-learn:** Modelagem de Machine Learning (Gradient Boosting, Regressão Linear).
*   **Streamlit:** Desenvolvimento da aplicação web interativa.
*   **Joblib:** Serialização e desserialização do modelo.

## Melhorias Futuras

*   **Coleta de Dados Contínua:** Implementar um pipeline para ingestão e atualização automática de dados.
*   **Interpretabilidade do Modelo (XAI):** Adicionar explicações locais para as predições do modelo (e.g., SHAP values) na aplicação Streamlit.
*   **Feedback Loop:** Desenvolver um mecanismo para coletar feedback sobre as predições e intervenções, permitindo o retreinamento e a melhoria contínua do modelo.
*   **Otimização do Threshold:** Explorar métodos para otimizar o threshold de classificação com base em custos de falsos positivos e falsos negativos específicos do domínio educacional.
