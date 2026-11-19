import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ia
from core.modelo import IDIOMAS, NOMES_IDIOMAS, ModeloTokens

st.set_page_config(page_title="TokenScope · Custo de IA", page_icon="◆", layout="wide")

ROXO, VERDE, CINZA, ROSA = "#6D28D9", "#10B981", "#94A3B8", "#F43F5E"
LIMITE_CHAMADAS = 20  # por sessão, quando a chave é a do servidor

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
@keyframes sobe { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes brilho { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.hero { animation: sobe .7s ease both; margin-bottom: 1.4rem; }
.hero .selo { display:inline-block; font-size:.75rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:#6D28D9; background:#F3EEFF; border:1px solid #E4D9FF; padding:.3rem .7rem; border-radius:999px; }
.hero h1 { font-size: 2.7rem; font-weight: 800; line-height:1.1; margin:.8rem 0 .5rem; letter-spacing:-.02em;
  background: linear-gradient(90deg,#1E1B4B,#6D28D9,#10B981,#6D28D9); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: brilho 12s ease infinite; }
.hero p { color:#475569; font-size:1.05rem; max-width: 760px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: .4rem 0 1.2rem; }
.card { background:#fff; border:1px solid #EEF0F4; border-radius:16px; padding:18px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); transition: transform .25s ease, box-shadow .25s ease, border-color .25s;
  animation: sobe .6s ease both; }
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px -12px rgba(109,40,217,.25); border-color:#E4D9FF; }
.card .rot { font-size:.78rem; color:#64748B; font-weight:500; text-transform:uppercase; letter-spacing:.05em; }
.card .val { font-size:1.75rem; font-weight:700; color:#0F172A; margin-top:.25rem; letter-spacing:-.01em; }
.card .dica { font-size:.82rem; color:#94A3B8; margin-top:.2rem; }
.card.destaque { background: linear-gradient(135deg,#6D28D9,#4C1D95); border:none; }
.card.destaque .rot, .card.destaque .dica { color:#DDD6FE; } .card.destaque .val { color:#fff; }
.nota { background:#F8FAFC; border-left:3px solid #6D28D9; padding:.8rem 1rem; border-radius:8px; color:#334155;
  font-size:.92rem; animation: sobe .6s ease both; }
.tag { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.tag.ok { background:#ECFDF5; color:#047857; } .tag.fora { background:#FFF1F2; color:#BE123C; }
section[data-testid="stSidebar"] { background:#FBFAFF; border-right:1px solid #EEF0F4; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 16px; transition: background .2s; }
.stTabs [data-baseweb="tab"]:hover { background:#F3EEFF; }
div[data-testid="stButton"] button[kind="primary"] { border-radius:12px; padding:.6rem 1.2rem; font-weight:600;
  background: linear-gradient(135deg,#7C3AED,#6D28D9); border:none; transition: transform .2s, box-shadow .2s; }
div[data-testid="stButton"] button[kind="primary"]:hover { transform: translateY(-1px); box-shadow:0 8px 20px -8px rgba(109,40,217,.6); }
textarea { border-radius:12px !important; }
footer { visibility:hidden; }
</style>
""")


def cards(itens):
    html = "".join(
        f'<div class="card {"destaque" if d else ""}" style="animation-delay:{i * 70}ms">'
        f'<div class="rot">{r}</div><div class="val">{v}</div><div class="dica">{h}</div></div>'
        for i, (r, v, h, d) in enumerate(itens)
    )
    st.html(f'<div class="cards">{html}</div>')


def grafico(fig, altura=360):
    fig.update_layout(
        height=altura, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#334155"),
        legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(font_family="Inter"),
    )
    fig.update_xaxes(gridcolor="#EEF0F4", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0F4", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def usd(v):
    return f"US$ {v:,.4f}" if v < 1 else f"US$ {v:,.2f}"


@st.cache_resource(show_spinner="Treinando o modelo com 18.924 conversas reais…")
def carregar_modelo():
    return ModeloTokens()


@st.cache_data(show_spinner="Calibrando com o contador de tokens do Gemini…")
def calibrar(_chave, modelo):
    return ia.fator_calibracao(_chave, modelo)


def segredo(nome):
    try:
        return st.secrets.get(nome) or os.environ.get(nome)
    except Exception:
        return os.environ.get(nome)


modelo = carregar_modelo()
st.session_state.setdefault("historico", [])
st.session_state.setdefault("chamadas", 0)

# ---------- Barra lateral ----------
with st.sidebar:
    st.markdown("### ◆ TokenScope")
    st.caption("Configuração da simulação")
    chave_servidor = segredo("GEMINI_API_KEY")
    chave_propria = st.text_input("Sua API key do Gemini (opcional)", type="password",
                                  help="Se vazio, usa a chave configurada no servidor, com limite de chamadas por sessão.")
    chave = chave_propria or chave_servidor
    nome_modelo = st.text_input("Modelo do Gemini", value=segredo("GEMINI_MODEL") or "gemini-2.5-flash")
    idioma = st.selectbox("Idioma dos usuários", IDIOMAS, index=IDIOMAS.index("pt-BR"),
                          format_func=lambda l: NOMES_IDIOMAS[l])
    req_mes = st.slider("Requisições por mês", 1_000, 200_000, 10_000, step=1_000)
    st.markdown("**Preço do modelo** (US$ por 1M tokens)")
    c1, c2 = st.columns(2)
    preco_in = c1.number_input("Entrada", 0.0, 100.0, 0.30, step=0.05, format="%.2f")
    preco_out = c2.number_input("Saída", 0.0, 200.0, 2.50, step=0.10, format="%.2f")
    st.caption("Valores de exemplo: confira a tabela de preços do modelo escolhido.")

fator, calibrado = 1.0, False
if chave:
    try:
        fator, calibrado = calibrar(chave, nome_modelo), True
    except Exception as e:
        st.sidebar.warning(f"Não consegui calibrar com o Gemini ({type(e).__name__}). Usando a unidade o200k.")

# ---------- Hero ----------
st.html("""
<div class="hero">
  <span class="selo">IA aplicada · IA experimental</span>
  <h1>Quanto custa a IA<br>da sua aplicação?</h1>
  <p>Cole o prompt que a sua aplicação vai enviar. Um modelo treinado com <b>18.924 conversas reais</b>
  prevê quantos tokens a IA vai gerar e quanto isso custa por mês. Depois, executamos no <b>Gemini</b>
  e comparamos a previsão com o consumo real.</p>
</div>
""")

aba_sim, aba_vivo, aba_dados, aba_valid, aba_sobre = st.tabs(
    ["Simulador", "Ao vivo com Gemini", "Os dados", "O modelo é válido?", "Sobre o projeto"])

# ---------- Simulador ----------
with aba_sim:
    esq, dir_ = st.columns([1.35, 1], gap="large")
    with esq:
        prompt = st.text_area(
            "Prompt da sua aplicação",
            value="Explique de forma simples o que é inflação e dê dois exemplos do dia a dia.",
            height=170,
        )
        continuacao = st.toggle("É continuação de uma conversa", value=False)
    prev = modelo.prever(prompt, idioma, continuacao)
    ent_tok = prev.prompt_tokens * fator
    sai_tok = prev.media * fator
    custo_req = (ent_tok * preco_in + sai_tok * preco_out) / 1e6
    with dir_:
        unidade = "tokens do Gemini" if calibrado else "tokens (o200k)"
        aviso = "" if calibrado else "<br><br>Sem chave do Gemini, os valores estão na unidade do tokenizador o200k."
        st.html(f"""
        <div class="nota" style="margin-top:1.8rem">A resposta deve ter entre <b>{prev.p10 * fator:,.0f}</b> e
        <b>{prev.p90 * fator:,.0f}</b> {unidade} (faixa de 80%), com mediana de <b>{prev.mediana * fator:,.0f}</b>.{aviso}</div>""")

    cards([
        ("Tokens de entrada", f"{ent_tok:,.0f}", "o que a aplicação envia", False),
        ("Saída prevista", f"{sai_tok:,.0f}", f"faixa {prev.p10 * fator:,.0f} – {prev.p90 * fator:,.0f}", False),
        ("Custo por requisição", usd(custo_req), "entrada + saída", False),
        ("Custo por mês", usd(custo_req * req_mes), f"{req_mes:,} requisições", True),
    ])

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("##### Para onde vai o dinheiro")
        fig = go.Figure(go.Pie(
            labels=["Entrada (prompt)", "Saída (resposta)"],
            values=[ent_tok * preco_in, sai_tok * preco_out], hole=.68,
            marker=dict(colors=[CINZA, ROXO]), textinfo="percent", sort=False))
        grafico(fig, 320)
    with g2:
        st.markdown("##### Quanto economizo limitando a resposta?")
        limites = [64, 128, 192, 256, 384, 512, 768, 1024]
        eco = np.array([modelo.economia_com_limite(l) for l in limites])
        fig = go.Figure()
        fig.add_bar(x=[str(l) for l in limites], y=eco[:, 0] * 100, name="economia na saída",
                    marker_color=VERDE, hovertemplate="%{y:.0f}%<extra></extra>")
        fig.add_scatter(x=[str(l) for l in limites], y=eco[:, 1] * 100, name="respostas cortadas",
                        mode="lines+markers", line=dict(color=ROSA, width=2), hovertemplate="%{y:.0f}%<extra></extra>")
        fig.update_layout(xaxis_title="limite de tokens da resposta (max_output_tokens)", yaxis_title="%")
        grafico(fig, 320)
    st.caption("A curva usa o tamanho real das respostas da base: limitar economiza, mas corta respostas. "
               "O ponto de equilíbrio é onde a economia ainda é alta e poucas respostas são cortadas.")

# ---------- Ao vivo ----------
with aba_vivo:
    st.markdown("#### Previsão × consumo real")
    st.caption("Executa o prompt do Simulador no Gemini. A API informa os tokens realmente gastos, "
               "e cada chamada entra no histórico abaixo para medir o acerto do modelo.")
    usando_servidor = bool(chave) and not chave_propria
    restante = LIMITE_CHAMADAS - st.session_state.chamadas
    if not chave:
        st.info("Informe uma API key do Gemini na barra lateral para executar ao vivo.")
    else:
        if usando_servidor:
            st.caption(f"Chamadas restantes nesta sessão: {restante}")
        if st.button("Executar no Gemini", type="primary", disabled=usando_servidor and restante <= 0):
            try:
                with st.spinner("Chamando o Gemini…"):
                    r = ia.gerar(prompt, chave, nome_modelo)
                st.session_state.chamadas += 1
                st.session_state.historico.append({
                    "prompt": prompt[:80] + ("…" if len(prompt) > 80 else ""),
                    "previsto": prev.mediana * fator, "p10": prev.p10 * fator, "p90": prev.p90 * fator,
                    "real": r.resposta_tokens, "raciocinio": r.raciocinio_tokens,
                    "entrada": r.prompt_tokens, "segundos": r.segundos,
                    "custo": (r.prompt_tokens * preco_in + (r.resposta_tokens + r.raciocinio_tokens) * preco_out) / 1e6,
                })
                st.session_state.ultima = r.texto
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    st.warning("Limite de uso do Gemini atingido. Aguarde um minuto e tente de novo.")
                else:
                    st.error(f"Erro ao chamar o Gemini: {type(e).__name__}: {msg[:200]}")

    hist = pd.DataFrame(st.session_state.historico)
    if hist.empty:
        st.html('<div class="nota">Nenhuma execução ainda. Os resultados aparecem aqui.</div>')
    else:
        u = hist.iloc[-1]
        dentro = u.p10 <= u.real <= u.p90
        st.html(f'<span class="tag {"ok" if dentro else "fora"}">'
                f'{"Dentro da faixa prevista" if dentro else "Fora da faixa prevista"}</span>')
        cards([
            ("Previsto (última)", f"{u.previsto:,.0f}", f"faixa {u.p10:,.0f} – {u.p90:,.0f}", False),
            ("Real (última)", f"{u.real:,.0f}",
             f"+ {u.raciocinio:,.0f} de raciocínio" if u.raciocinio else "tokens de resposta", False),
            ("Acerto da faixa", f"{100 * ((hist.real >= hist.p10) & (hist.real <= hist.p90)).mean():.0f}%",
             f"em {len(hist)} execuções (meta ≈ 80%)", False),
            ("Tokens de IA gastos", f"{int(hist.entrada.sum() + hist.real.sum() + hist.raciocinio.sum()):,}",
             f"custo {usd(hist.custo.sum())}", True),
        ])
        with st.expander("Resposta do Gemini"):
            st.markdown(st.session_state.get("ultima", ""))
        x = list(range(1, len(hist) + 1))
        fig = go.Figure()
        fig.add_scatter(x=x + x[::-1], y=list(hist.p90) + list(hist.p10[::-1]), fill="toself",
                        fillcolor="rgba(109,40,217,.10)", line=dict(width=0), name="faixa prevista (80%)",
                        hoverinfo="skip")
        fig.add_scatter(x=x, y=hist.previsto, mode="lines+markers", name="previsto", line=dict(color=ROXO, dash="dot"))
        fig.add_scatter(x=x, y=hist.real, mode="lines+markers", name="real (Gemini)", line=dict(color=VERDE, width=3))
        fig.update_layout(xaxis_title="execução", yaxis_title="tokens da resposta")
        grafico(fig)
        st.dataframe(hist.drop(columns=["p10", "p90"]).round(2), hide_index=True, width="stretch")

# ---------- Dados ----------
with aba_dados:
    p = modelo.pares
    cards([
        ("Conversas analisadas", f"{len(p):,}", "pares prompt → resposta", False),
        ("Mediana do prompt", f"{p.prompt_tokens.median():,.0f}", "tokens", False),
        ("Mediana da resposta", f"{p.resp_tokens.median():,.0f}", "tokens", False),
        ("Resposta / prompt", f"{p.resp_tokens.median() / p.prompt_tokens.median():.1f}×", "a saída domina", True),
    ])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("##### Distribuição de tokens")
        bins = np.logspace(0, 3.8, 45)
        fig = go.Figure()
        for col, nome, cor in [("prompt_tokens", "prompt", CINZA), ("resp_tokens", "resposta", ROXO)]:
            h, e = np.histogram(p[col], bins=bins)
            fig.add_bar(x=np.sqrt(e[:-1] * e[1:]), y=h, name=nome, marker_color=cor, opacity=.8)
        fig.update_layout(barmode="overlay", xaxis_type="log", xaxis_title="tokens (escala log)", yaxis_title="conversas")
        grafico(fig)
    with g2:
        st.markdown("##### O idioma muda o custo")
        d = modelo.idiomas.sort_values("tokens_por_1000_caracteres")
        fig = go.Figure(go.Bar(
            x=d.tokens_por_1000_caracteres, y=[NOMES_IDIOMAS[l] for l in d.lang], orientation="h",
            marker_color=[ROXO if l == idioma else "#C4B5FD" for l in d.lang],
            hovertemplate="%{x:.0f} tokens<extra></extra>"))
        fig.update_layout(xaxis_title="tokens a cada 1.000 caracteres")
        grafico(fig)
    st.markdown("##### Prompt × resposta")
    a = p.sample(4000, random_state=0)
    corr = np.corrcoef(np.log1p(p.prompt_tokens), np.log1p(p.resp_tokens))[0, 1]
    fig = go.Figure(go.Scattergl(x=a.prompt_tokens, y=a.resp_tokens, mode="markers",
                                 marker=dict(size=4, color=ROXO, opacity=.25)))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="tokens do prompt", yaxis_title="tokens da resposta",
                      title=dict(text=f"correlação (log) = {corr:.2f}: existe, mas é fraca", font=dict(size=13)))
    grafico(fig, 380)
    st.caption("Fonte: OpenAssistant Conversations (oasst1), Apache 2.0. Tokens contados com o tokenizador público o200k_base.")

# ---------- Validação ----------
with aba_valid:
    m = modelo.metricas.set_index("modelo")
    gb = m.loc["Gradient Boosting"]
    real, previsto = modelo.custo_mensal(req_mes, preco_in, preco_out)
    erro_mes = 100 * np.median(np.abs(previsto - real) / real)
    cards([
        ("R² do modelo", f"{gb.r2:.2f}", "por resposta (escala log)", False),
        ("Erro médio", f"{gb.mae:.0f} tokens", f"baseline: {m.loc['Baseline (mediana)'].mae:.0f}", False),
        ("Acerto da faixa 80%", f"{100 * modelo.cobertura_faixa:.0f}%", "no conjunto de teste", False),
        ("Erro no custo do mês", f"{erro_mes:.1f}%", f"{req_mes:,} requisições", True),
    ])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("##### Erro por resposta (menor é melhor)")
        fig = go.Figure(go.Bar(x=m.index, y=m.mae, marker_color=[CINZA, "#A78BFA", ROXO],
                               text=[f"{v:.0f}" for v in m.mae], textposition="outside"))
        fig.update_layout(yaxis_title="erro médio absoluto (tokens)")
        grafico(fig)
    with g2:
        st.markdown("##### Custo mensal: previsto × real")
        fig = go.Figure()
        fig.add_histogram(x=real, name="real", marker_color=CINZA, opacity=.75, nbinsx=50)
        fig.add_histogram(x=previsto, name="previsto", marker_color=ROXO, opacity=.75, nbinsx=50)
        fig.update_layout(barmode="overlay", xaxis_title="US$ por mês", yaxis_title="simulações (Monte Carlo)")
        grafico(fig)
    st.html(f"""
    <div class="nota"><b>Leitura honesta.</b> Prever o tamanho de <i>uma</i> resposta é difícil: o modelo explica
    ~{100 * gb.r2:.0f}% da variação e bate o baseline por pouco. Mas a aplicação paga milhares de respostas, e na soma
    os erros se compensam: o custo do mês fica a ~{erro_mes:.0f}% do real. A faixa de 80% acerta
    {100 * modelo.cobertura_faixa:.0f}% das vezes no teste. Na aba <b>Ao vivo</b>, a mesma métrica é medida contra o Gemini.</div>""")

# ---------- Sobre ----------
with aba_sobre:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
#### O problema
Toda aplicação com IA generativa paga por **token**. A aplicação conhece o prompt que envia,
mas **não sabe o tamanho da resposta**, que é a parte mais cara. Sem isso, não dá para planejar o orçamento.

#### A técnica
**Inferir** os tokens da resposta a partir do prompt, **comparar** com uma referência ingênua e
**avaliar o impacto** no custo mensal. Depois, **validar ao vivo** com os tokens reais que o Gemini informa.

#### A família de algoritmos
- **Aprendizado supervisionado, regressão**: Gradient Boosting (previsão central) e regressão quantílica
  (faixa de 80%), comparados com regressão linear e baseline.
- **Simulação de Monte Carlo** (bootstrap) para o custo mensal.
""")
    with c2:
        st.markdown("""
#### Os dados
[OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1): ~88 mil mensagens reais,
18.924 pares prompt → melhor resposta, em vários idiomas. Licença Apache 2.0.

#### O token de IA
A aplicação usa a **API do Gemini**: cada execução consome tokens reais, que são medidos e comparados com a previsão.
Para comparar na mesma unidade, o app mede uma amostra da base com o contador de tokens do Gemini e calibra o modelo.

#### Limitações
- As respostas da base são de voluntários, não de um modelo comercial.
- Modelos com raciocínio ("thinking") gastam tokens extras que não aparecem no texto. O app mostra esses tokens à parte.
- Os preços são editáveis e mudam com o tempo.
""")
    st.caption("Atividade de IA aplicada e IA experimental, Prof. Wagner, 3º ciclo.")
