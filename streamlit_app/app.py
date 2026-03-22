"""
app.py — Datathon Passos Mágicos  (v2.0)
=========================================
Sistema de Predição de Risco de Lacunas de Aprendizado.

Alinhamento com o modelo v2.0
-------------------------------
Features usadas (9):
  Brutas   : IDA, IEG, IAA, IPS, IPP, IPV
  Derivadas: perception_gap, behavioral_score, relative_performance

O modelo NÃO usa IAN, INDE real, nem year — apenas o que os sliders fornecem.
"""

import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Risco - Passos Mágicos",
    page_icon="📚",
    layout="centered"
)

st.title("🎯 Predição de Risco de Lacunas de Aprendizado")
st.markdown("**Datathon Passos Mágicos** — Sistema de Identificação de Alunos em Risco")

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "model_risk_gap.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model_data = load_model()
    model     = model_data["model"]
    features  = model_data["features"]
    threshold = model_data["threshold"]   # 0.35 — orientado a recall
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Input sliders
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Indicadores do Aluno")
st.markdown(
    "Utilize os sliders abaixo para inserir os indicadores de desenvolvimento "
    "do aluno (escala 0–10):"
)

col1, col2 = st.columns(2)

with col1:
    IDA = st.slider("🎓 Desempenho Acadêmico (IDA)", 0.0, 10.0, 5.0, step=0.1)
    IEG = st.slider("🤝 Engajamento (IEG)",           0.0, 10.0, 5.0, step=0.1)
    IAA = st.slider("💭 Autoavaliação (IAA)",          0.0, 10.0, 5.0, step=0.1)

with col2:
    IPS = st.slider("❤️ Suporte Psicossocial (IPS)",  0.0, 10.0, 5.0, step=0.1)
    IPP = st.slider("📈 Progresso Pessoal (IPP)",      0.0, 10.0, 5.0, step=0.1)
    IPV = st.slider("🎯 Ponto de Virada (IPV)",        0.0, 10.0, 5.0, step=0.1)

# ─────────────────────────────────────────────
# Feature engineering — idêntica ao treinamento
# ─────────────────────────────────────────────
perception_gap       = IAA - IDA              # gap entre autopercepção e desempenho
behavioral_score     = (IEG + IPS) / 2        # média de engajamento e suporte
relative_performance = IDA - 7                # desvio em relação ao benchmark do programa

feature_values = {
    "IDA":                  IDA,
    "IEG":                  IEG,
    "IAA":                  IAA,
    "IPS":                  IPS,
    "IPP":                  IPP,
    "IPV":                  IPV,
    "perception_gap":       perception_gap,
    "behavioral_score":     behavioral_score,
    "relative_performance": relative_performance,
}

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.markdown("---")
if st.button("🔮 Realizar Predição", use_container_width=True):
    try:
        # Montar DataFrame na ordem exata em que o modelo foi treinado
        input_df = pd.DataFrame(
            [[feature_values[f] for f in features]],
            columns=features
        )

        # Probabilidade bruta do modelo — nunca modificada após este ponto
        probability = model.predict_proba(input_df)[0][1]

        # Classificação baseada no threshold
        is_high_risk = probability >= threshold

        # ── Resultado ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Resultado da Predição")

        col1, col2 = st.columns([1, 1])

        with col1:
            if is_high_risk:
                st.metric("Status", "🔴 ALTO RISCO",  f"{probability * 100:.1f}%")
            else:
                st.metric("Status", "🟢 BAIXO RISCO", f"{probability * 100:.1f}%")

        with col2:
            distance = abs(probability - threshold)
            if distance < 0.07:
                confidence_label = "⚠️ Borderline"
            elif distance < 0.20:
                confidence_label = "🟡 Moderada"
            else:
                confidence_label = "🟢 Alta" if not is_high_risk else "🔴 Alta"
            st.metric("Confiança da Predição", confidence_label)

        st.progress(
            float(probability),
            text=f"Probabilidade de risco (saída do modelo): {probability * 100:.2f}%"
        )

        st.caption(f"Threshold de classificação: {threshold:.2f}")
        st.caption(f"Probabilidade bruta do modelo: {probability:.4f}")

        # ── Interpretação ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Interpretação")

        if is_high_risk:
            st.warning(
                "⚠️ **Este aluno foi identificado como tendo ALTO RISCO de lacunas "
                "de aprendizado.**\n\n"
                "**Recomendações:**\n"
                "- Realizar acompanhamento pedagógico intensivo\n"
                "- Avaliar suporte psicossocial\n"
                "- Ajustar plano individual de desenvolvimento\n"
                "- Aumentar frequência de monitoramento"
            )
        else:
            st.success(
                "✅ **Este aluno foi identificado como tendo BAIXO RISCO.**\n\n"
                "- Manter acompanhamento regular\n"
                "- Continuar as estratégias pedagógicas atuais\n"
                "- Monitorar evolução nos próximos ciclos"
            )

        # ── Tabela de fatores ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Análise de Fatores")

        factor_data = {
            "Fator": [
                "Desempenho Acadêmico (IDA)",
                "Engajamento (IEG)",
                "Autoavaliação (IAA)",
                "Suporte Psicossocial (IPS)",
                "Progresso Pessoal (IPP)",
                "Ponto de Virada (IPV)",
                "Gap de Percepção (IAA − IDA)",
                "Score Comportamental",
                "Desempenho Relativo (IDA − 7)",
            ],
            "Valor": [
                f"{IDA:.1f}",
                f"{IEG:.1f}",
                f"{IAA:.1f}",
                f"{IPS:.1f}",
                f"{IPP:.1f}",
                f"{IPV:.1f}",
                f"{perception_gap:+.1f}",
                f"{behavioral_score:.1f}",
                f"{relative_performance:+.1f}",
            ],
            "Status": [
                "🟢 Bom"  if IDA >= 6 else "🟡 Médio" if IDA >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IEG >= 6 else "🟡 Médio" if IEG >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IAA >= 6 else "🟡 Médio" if IAA >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IPS >= 6 else "🟡 Médio" if IPS >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IPP >= 6 else "🟡 Médio" if IPP >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IPV >= 6 else "🟡 Médio" if IPV >= 4 else "🔴 Baixo",
                ("🟢 Alinhado" if abs(perception_gap) <= 2
                 else "🟡 Moderado" if abs(perception_gap) <= 4
                 else "🔴 Desalinhado"),
                "🟢 Bom"  if behavioral_score >= 6 else "🟡 Médio" if behavioral_score >= 4 else "🔴 Baixo",
                "🟢 Acima" if relative_performance >= 0 else "🟡 Próximo" if relative_performance >= -1 else "🔴 Abaixo",
            ],
        }

        st.dataframe(
            pd.DataFrame(factor_data),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error("Erro na predição:")
        st.exception(e)
