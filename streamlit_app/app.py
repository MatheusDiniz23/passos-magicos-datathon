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
st.markdown("**Datathon Passos Mágicos** - Sistema de Identificação de Alunos em Risco")

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "model_risk_gap.pkl"
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

try:
    model_data = load_model()
    model    = model_data["model"]
    features = model_data["features"]
    # Load threshold from artifact when available; fall back to tuned default.
    # 0.35 was chosen to favour recall, reducing missed at-risk students.
    threshold = model_data.get("threshold", 0.35)
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
# Feature engineering
# NOTE: INDE is used here as a proxy equal to IDA for simplification.
# In the original dataset INDE is a composite index; a full implementation
# would recompute it from its weighted components.
# ─────────────────────────────────────────────
INDE               = IDA                    # proxy — see note above
perception_gap     = IAA - IDA              # self-perception vs. academic performance
behavioral_score   = (IEG + IPS) / 2        # average of engagement and psychosocial support
relative_performance = IDA - 7              # deviation from the programme's reference benchmark

feature_values = {
    "INDE":                 INDE,
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
        # Build input DataFrame in the exact column order the model was trained on
        input_df = pd.DataFrame(
            [[feature_values[f] for f in features]],
            columns=features
        )

        # Raw model probability — this value is NEVER modified after this point
        probability = model.predict_proba(input_df)[0][1]

        # ── Business interpretation warning ──────────────────────────────────
        # When every raw indicator is exceptionally high the prediction may fall
        # outside the distribution the model was trained on.  We surface this as
        # an informational notice rather than altering the model output.
        raw_indicators = [IDA, IEG, IAA, IPS, IPP, IPV]
        if min(raw_indicators) >= 9:
            st.warning(
                "⚠️ **Todos os indicadores estão em níveis muito elevados.** "
                "Interprete a predição com cautela — este perfil pode estar fora "
                "da distribuição de treino do modelo."
            )
        elif IDA >= 8 and IEG >= 8 and IPS >= 8 and IPP >= 8 and IPV >= 8:
            st.info(
                "ℹ️ **A maioria dos indicadores está em níveis elevados.** "
                "A predição abaixo é o resultado direto do modelo."
            )
        # ─────────────────────────────────────────────────────────────────────

        # Threshold-based classification (recall-oriented: 0.35 < 0.50)
        is_high_risk = probability >= threshold

        # ── Result display ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Resultado da Predição")

        col1, col2 = st.columns([1, 1])

        with col1:
            if is_high_risk:
                st.metric("Status", "🔴 ALTO RISCO",  f"{probability * 100:.1f}%")
            else:
                st.metric("Status", "🟢 BAIXO RISCO", f"{probability * 100:.1f}%")

        with col2:
            # Distance from the decision boundary expressed as a plain label
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

        # Transparency captions — values shown are always the unaltered model output
        st.caption(f"Threshold de classificação: {threshold:.3f}")
        st.caption(f"Probabilidade bruta do modelo: {probability:.4f}")

        # ── Interpretation ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Interpretação")

        if is_high_risk:
            st.warning(
                "⚠️ **Este aluno foi identificado como tendo ALTO RISCO de lacunas "
                "de aprendizado.**\n\n"
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

        # ── Factor analysis table ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Análise de Fatores")

        factor_data = {
            "Fator": [
                "Desempenho Acadêmico (IDA)",
                "Engajamento (IEG)",
                "Gap de Percepção (IAA − IDA)",
                "Suporte Psicossocial (IPS)",
                "Progresso Pessoal (IPP)",
                "Ponto de Virada (IPV)",
            ],
            "Valor": [
                f"{IDA:.1f}",
                f"{IEG:.1f}",
                f"{perception_gap:+.1f}",
                f"{IPS:.1f}",
                f"{IPP:.1f}",
                f"{IPV:.1f}",
            ],
            "Status": [
                "🟢 Bom"  if IDA >= 6 else "🟡 Médio" if IDA >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IEG >= 6 else "🟡 Médio" if IEG >= 4 else "🔴 Baixo",
                ("🟢 Alinhado" if abs(perception_gap) <= 2
                 else "🟡 Moderado" if abs(perception_gap) <= 4
                 else "🔴 Desalinhado"),
                "🟢 Bom"  if IPS >= 6 else "🟡 Médio" if IPS >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IPP >= 6 else "🟡 Médio" if IPP >= 4 else "🔴 Baixo",
                "🟢 Bom"  if IPV >= 6 else "🟡 Médio" if IPV >= 4 else "🔴 Baixo",
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
