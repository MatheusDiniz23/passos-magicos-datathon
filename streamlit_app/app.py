import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Predição de Risco - Passos Mágicos",
    page_icon="📚",
    layout="centered"
)

# Título e descrição
st.title("🎯 Predição de Risco de Lacunas de Aprendizado")
st.markdown("**Datathon Passos Mágicos** - Sistema de Identificação de Alunos em Risco")

# Carregamento do modelo
@st.cache_resource
def load_model():
    with open('models/model_risk_gap.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

try:
    model_data = load_model()
    model = model_data['model']
    threshold = model_data['threshold']
    features = model_data['features']
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Seção de entrada
st.markdown("---")
st.subheader("📊 Indicadores do Aluno")
st.markdown("Utilize os sliders abaixo para inserir os indicadores de desenvolvimento do aluno (escala 0-10):")

# Criação dos sliders em duas colunas
col1, col2 = st.columns(2)

with col1:
    IDA = st.slider("🎓 Desempenho Acadêmico (IDA)", 0.0, 10.0, 5.0, step=0.1)
    IEG = st.slider("🤝 Engajamento (IEG)", 0.0, 10.0, 5.0, step=0.1)
    IAA = st.slider("💭 Autoavaliação (IAA)", 0.0, 10.0, 5.0, step=0.1)

with col2:
    IPS = st.slider("❤️ Suporte Psicossocial (IPS)", 0.0, 10.0, 5.0, step=0.1)
    IPP = st.slider("📈 Progresso Pessoal (IPP)", 0.0, 10.0, 5.0, step=0.1)
    IPV = st.slider("🎯 Ponto de Virada (IPV)", 0.0, 10.0, 5.0, step=0.1)

# Cálculo de features derivadas
INDE = (IDA + IEG + IAA + IPS + IPP + IPV) / 6  # Média dos indicadores
perception_gap = IAA - IDA
behavioral_score = (IEG + IPS) / 2
relative_performance = IDA - 5  # Normalizado em relação à média (5)

# Botão de predição
st.markdown("---")
if st.button("🔮 Realizar Predição", use_container_width=True):
    # Preparar dados para o modelo
    input_data = np.array([[
        INDE,
        IDA,
        IEG,
        IAA,
        IPS,
        IPP,
        IPV,
        perception_gap,
        behavioral_score,
        relative_performance
    ]])
    
    # Fazer predição
    probability = model.predict_proba(input_data)[0][1]
    
    # Determinar risco
    is_high_risk = probability >= threshold
    
    # Exibir resultado
    st.markdown("---")
    st.subheader("📋 Resultado da Predição")
    
    # Barra de progresso
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Cor baseada no risco
        if is_high_risk:
            st.metric("Status", "🔴 ALTO RISCO", f"{probability*100:.1f}%")
        else:
            st.metric("Status", "🟢 BAIXO RISCO", f"{probability*100:.1f}%")
    
    with col2:
        st.metric("Confiança", f"{max(probability, 1-probability)*100:.1f}%")
    
    # Visualização da probabilidade
    st.progress(probability, text=f"Probabilidade de Risco: {probability*100:.2f}%")
    
    # Explicação
    st.markdown("---")
    st.subheader("💡 Interpretação")
    
    if is_high_risk:
        st.warning(
            "⚠️ **Este aluno foi identificado como tendo ALTO RISCO de lacunas de aprendizado.**\n\n"
            "**Recomendações:**\n"
            "- Realizar acompanhamento pedagógico intensivo\n"
            "- Avaliar necessidade de suporte psicossocial adicional\n"
            "- Considerar ajustes no plano de desenvolvimento individual\n"
            "- Aumentar frequência de avaliações de progresso"
        )
    else:
        st.success(
            "✅ **Este aluno foi identificado como tendo BAIXO RISCO.**\n\n"
            "**Próximos passos:**\n"
            "- Continuar monitoramento regular\n"
            "- Manter estratégias pedagógicas atuais\n"
            "- Explorar oportunidades de aprofundamento"
        )
    
    # Fatores de risco
    st.markdown("---")
    st.subheader("🔍 Análise de Fatores")
    
    factor_data = {
        "Fator": [
            "Desempenho Acadêmico",
            "Engajamento",
            "Gap de Percepção",
            "Suporte Psicossocial",
            "Progresso Pessoal"
        ],
        "Valor": [
            f"{IDA:.1f}",
            f"{IEG:.1f}",
            f"{perception_gap:.1f}",
            f"{IPS:.1f}",
            f"{IPP:.1f}"
        ],
        "Status": [
            "🟢 Bom" if IDA >= 6 else "🟡 Médio" if IDA >= 4 else "🔴 Baixo",
            "🟢 Bom" if IEG >= 6 else "🟡 Médio" if IEG >= 4 else "🔴 Baixo",
            "🟢 Alinhado" if abs(perception_gap) <= 2 else "🟡 Moderado" if abs(perception_gap) <= 4 else "🔴 Desalinhado",
            "🟢 Bom" if IPS >= 6 else "🟡 Médio" if IPS >= 4 else "🔴 Baixo",
            "🟢 Bom" if IPP >= 6 else "🟡 Médio" if IPP >= 4 else "🔴 Baixo"
        ]
    }
    
    df_factors = pd.DataFrame(factor_data)
    st.dataframe(df_factors, use_container_width=True, hide_index=True)

# Seção de informação
st.markdown("---")
st.markdown(
    "### 📚 Sobre Este Modelo\n\n"
    "Este modelo foi treinado com dados históricos de 2020 a 2022 da ONG Passos Mágicos. "
    "Ele utiliza indicadores de desempenho acadêmico, engajamento e desenvolvimento psicossocial "
    "para identificar alunos que podem estar em risco de desenvolver lacunas de aprendizado.\n\n"
    "**Fatores Principais de Risco:**\n"
    "- Baixo desempenho acadêmico (IDA)\n"
    "- Baixo engajamento (IEG)\n"
    "- Alto gap de percepção (diferença entre autoavaliação e desempenho real)\n"
    "- Falta de suporte psicossocial (IPS)\n\n"
    "**Nota:** Este sistema é uma ferramenta de suporte à decisão e não substitui a avaliação profissional da equipe pedagógica."
)
