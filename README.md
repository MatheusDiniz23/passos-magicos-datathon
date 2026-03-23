# Datathon Passos Mágicos: Predição de Risco de Lacunas de Aprendizado

## Visão Geral

Este projeto tem como objetivo identificar alunos em risco de lacunas de aprendizado a partir de indicadores educacionais. A solução utiliza técnicas de análise de dados e machine learning para apoiar decisões pedagógicas de forma proativa.

---

## Problema

O modelo inicial apresentava inconsistências, especialmente em cenários extremos.  
A causa foi a dependência da variável **IAN**, que não está disponível na aplicação final, gerando desalinhamento entre modelo e uso prático.

---

## Solução (Diferencial do Projeto)

Para resolver o problema, foi realizada a **reconstrução do target**:

- Estimativa do **INDE** a partir dos indicadores disponíveis  
- Criação de um novo target (`risk_gap_v2`) baseado nessa estimativa  
- Alinhamento completo entre modelo e aplicação  

Essa abordagem garantiu predições mais consistentes e coerentes.

---

## Dados

Foram utilizados dados educacionais da ONG Passos Mágicos:

- Indicadores: IDA, IEG, IAA, IPS, IPP, IPV  
- Base multi-ano (2021–2024)  
- Dataset processado para modelagem  

---

## Metodologia

- Análise Exploratória de Dados (EDA)  
- Engenharia de Features  
  - perception_gap  
  - behavioral_score  
  - relative_performance  
- Prevenção de Data Leakage  
- Modelagem com Machine Learning  

---

## Modelo

- **Algoritmo:** Gradient Boosting  
- **Features:** 6 indicadores + 3 derivadas  
- **Foco:** Recall (redução de falsos negativos)  
- Modelo validado com cross-validation  

---

## Aplicação

O projeto inclui uma aplicação interativa em **Streamlit**, que permite:

- Inserir indicadores do aluno  
- Simular cenários  
- Obter predição de risco em tempo real  

---

## Estrutura do Projeto
