import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

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
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "model_risk_gap.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

try:
    model_data = load_model()
    model = model_data["model"]
    threshold = 0.35
    features = model_data["features"]
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Seção de entrada
st.markdown("---")
st.subheader("📊 Indicadores do Aluno")
st.markdown("Utilize os sliders abaixo para inserir os indicadores de desenvolvimento do aluno (escala 0-10):")

# Sliders
col1, col2 = st.columns(2)

with col1:
    IDA = st.slider("🎓 Desempenho Acadêmico (IDA)", 0.0, 10.0, 5.0, step=0.1)
    IEG = st.slider("🤝 Engajamento (IEG)", 0.0, 10.0, 5.0, step=0.1)
    IAA = st.slider("💭 Autoavaliação (IAA)", 0.0, 10.0, 5.0, step=0.1)

with col2:
    IPS = st.slider("❤️ Suporte Psicossocial (IPS)", 0.0, 10.0, 5.0, step=0.1)
    IPP = st.slider("📈 Progresso Pessoal (IPP)", 0.0, 10.0, 5.0, step=0.1)
    IPV = st.slider("🎯 Ponto de Virada (IPV)", 0.0, 10.0, 5.0, step=0.1)

# Feature engineering
INDE = IDA
perception_gap = IAA - IDA
behavioral_score = (IEG + IPS) / 2
relative_performance = IDA - 7

feature_values = {
    "INDE": INDE,
    "IDA": IDA,
    "IEG": IEG,
    "IAA": IAA,
    "IPS": IPS,
    "IPP": IPP,
    "IPV": IPV,
    "perception_gap": perception_gap,
    "behavioral_score": behavioral_score,
    "relative_performance": relative_performance,
}

# Predição
st.markdown("---")
if st.button("🔮 Realizar Predição", use_container_width=True):
    try:
        input_df = pd.DataFrame(
            [[feature_values[f] for f in features]],
            columns=features
        )

        probability = model.predict_proba(input_df)[0][1]

        # =========================
        # 🔥 REGRA DE SANIDADE (NOVO)
        # =========================
        high_inputs = [IDA, IEG, IAA, IPS, IPP, IPV]

        if min(high_inputs) >= 9:
            probability = min(probability, 0.10)

        elif IDA >= 8 and IEG >= 8 and IPS >= 8 and IPP >= 8 and IPV >= 8:
            probability = min(probability, 0.20)

        # Classificação
        is_high_risk = probability >= threshold

        # Resultado
        st.markdown("---")
        st.subheader("📋 Resultado da Predição")

        col1, col2 = st.columns([1, 1])

        with col1:
            if is_high_risk:
                st.metric("Status", "🔴 ALTO RISCO", f"{probability*100:.1f}%")
            else:
                st.metric("Status", "🟢 BAIXO RISCO", f"{probability*100:.1f}%")

        with col2:
            st.metric("Confiança", f"{max(probability, 1 - probability) * 100:.1f}%")

        st.progress(float(probability), text=f"Probabilidade de Risco: {probability*100:.2f}%")

        # Debug
        st.caption(f"Threshold do modelo: {threshold:.3f}")
        st.caption(f"Probabilidade prevista: {probability:.3f}")

        # Interpretação
        st.markdown("---")
        st.subheader("💡 Interpretação")

        if is_high_risk:
            st.warning(
                "⚠️ **Este aluno foi identificado como tendo ALTO RISCO de lacunas de aprendizado.**\n\n"
                "**Recomendações:**\n"
                "- Realizar acompanhamento pedagógico intensivo\n"
                "- Avaliar suporte psicossocial\n"
                "- Ajustar plano individual\n"
                "- Aumentar monitoramento"
            )
        else:
            st.success(
                "✅ **Este aluno foi identificado como tendo BAIXO RISCO.**\n\n"
                "- Manter acompanhamento regular\n"
                "- Continuar estratégias atuais"
            )

        # Fatores
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

    except Exception as e:
        st.error("Erro na predição:")
        st.exception(e)
