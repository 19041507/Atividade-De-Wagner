import os
import threading
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ia
from core.modelo import NOMES_IDIOMAS, ModeloTokens

st.set_page_config(page_title="TokenScope", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

ROXO, VERDE, CINZA, ROSA = "#6D28D9", "#10B981", "#94A3B8", "#F43F5E"
INSTRUCAO = ("Você é um assistente prestativo e direto. Responda em português do Brasil, "
             "com clareza, usando listas ou exemplos quando ajudarem.")
SUGESTOES = [
    "Explique juros compostos com um exemplo",
    "Monte um roteiro de estudos de Python em 4 semanas",
    "Quais as diferenças entre IA, machine learning e deep learning?",
]


def escolher_sugestao():
    st.session_state.pendente = st.session_state.sugestao
    st.session_state.sugestao = None


def config(nome, padrao):
    try:
        v = st.secrets.get(nome)
    except Exception:
        v = None
    return v if v is not None else os.environ.get(nome, padrao)


CHAVE = config("GEMINI_API_KEY", None)
MODELO_IA = config("GEMINI_MODEL", "gemini-3.8-flash")
MODELO_RESERVA = config("GEMINI_MODEL_RESERVA", None)  # usado se o principal estiver sobrecarregado
PRECO_IN = float(config("PRECO_ENTRADA", 0.30))   # US$ por 1M tokens
PRECO_OUT = float(config("PRECO_SAIDA", 2.50))    # US$ por 1M tokens (inclui raciocínio)
LIMITE_SESSAO = int(config("LIMITE_SESSAO", 15))
LIMITE_DIA = int(config("LIMITE_DIA", 300))

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1080px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none; }
@keyframes sobe { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes brilho { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes pulso { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.topo { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
  margin-bottom: 2.2rem; animation: sobe .5s ease both; }
.marca { font-weight:800; font-size:1.15rem; letter-spacing:-.01em; color:#0F172A; }
.marca span { color:#6D28D9; }
.status { font-size:.8rem; color:#475569; background:#F8FAFC; border:1px solid #EEF0F4; padding:.35rem .75rem; border-radius:999px; }
.status i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.45rem; vertical-align:middle; animation: pulso 2s infinite; }
.hero { animation: sobe .7s ease both; margin-bottom: 1.8rem; }
.hero h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); font-weight: 800; line-height:1.1; margin:0 0 .6rem; letter-spacing:-.03em;
  padding-bottom:.1em;
  background: linear-gradient(90deg,#1E1B4B,#6D28D9,#10B981,#6D28D9); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: brilho 12s ease infinite; }
.hero p { color:#64748B; font-size:1rem; line-height:1.6; max-width: 680px; margin:0; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: .4rem 0 1.2rem; }
.card { background:#fff; border:1px solid #EEF0F4; border-radius:16px; padding:18px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); transition: transform .25s ease, box-shadow .25s ease, border-color .25s;
  animation: sobe .6s ease both; }
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px -12px rgba(109,40,217,.25); border-color:#E4D9FF; }
.card .rot { font-size:.76rem; color:#64748B; font-weight:500; text-transform:uppercase; letter-spacing:.05em; }
.card .val { font-size:1.7rem; font-weight:700; color:#0F172A; margin-top:.25rem; letter-spacing:-.01em; }
.card .dica { font-size:.82rem; color:#94A3B8; margin-top:.2rem; }
.card.destaque { background: linear-gradient(135deg,#6D28D9,#4C1D95); border:none; }
.card.destaque .rot, .card.destaque .dica { color:#DDD6FE; } .card.destaque .val { color:#fff; }
.nota { background:#F8FAFC; border-left:3px solid #6D28D9; padding:.8rem 1rem; border-radius:8px; color:#334155;
  font-size:.92rem; animation: sobe .6s ease both; }
.recibo { display:flex; flex-wrap:wrap; gap:6px; margin-top:.6rem; animation: sobe .5s ease both; }
.chip { font-size:.74rem; font-weight:500; color:#475569; background:#F8FAFC; border:1px solid #EEF0F4;
  padding:.2rem .6rem; border-radius:999px; }
.chip b { color:#0F172A; } .chip.ok { background:#ECFDF5; border-color:#D1FAE5; color:#047857; }
.chip.fora { background:#FFF1F2; border-color:#FFE4E6; color:#BE123C; }
.vazio { text-align:center; color:#64748B; padding:1.8rem 1rem 1.2rem; animation: sobe .6s ease both; }
.vazio .icone { width:44px; height:44px; margin:0 auto .8rem; border-radius:14px; display:flex; align-items:center;
  justify-content:center; font-size:1.3rem; color:#fff; background:linear-gradient(135deg,#7C3AED,#10B981);
  box-shadow:0 10px 24px -10px rgba(109,40,217,.6); }
.vazio b { color:#0F172A; display:block; font-size:1.1rem; margin-bottom:.25rem; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background:#F5F3FA; padding:4px; border-radius:14px; width:fit-content; max-width:100%; overflow-x:auto; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 16px; height:auto; color:#475569; transition: background .2s, color .2s; }
.stTabs [data-baseweb="tab"]:hover { color:#6D28D9; }
.stTabs [aria-selected="true"] { background:#fff; color:#6D28D9; box-shadow:0 1px 3px rgba(15,23,42,.08); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }
div[data-testid="stChatMessage"] { animation: sobe .4s ease both; border-radius:16px; padding:14px 16px; margin-bottom:.6rem;
  background:#FFFFFF; border:1px solid #EEF0F4; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#F8FAFC; border-color:#F1F5F9; }
div[data-testid="stChatInput"] { border-radius:16px; box-shadow:0 8px 30px -12px rgba(15,23,42,.18); }
div[data-testid="stPills"] { display:flex; justify-content:center; }
div[data-testid="stPills"] button { border-radius:999px; transition: transform .2s, border-color .2s; }
div[data-testid="stPills"] button:hover { transform: translateY(-1px); border-color:#C4B5FD; color:#6D28D9; }
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


def br(v, casas=0):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def usd(v):
    return f"US$ {br(v, 5 if v < 0.01 else 4 if v < 1 else 2)}"


@st.cache_resource(show_spinner="Carregando o modelo treinado com 18.924 conversas reais…")
def carregar_modelo():
    return ModeloTokens()


@st.cache_resource
def registro():
    """Uso de todos os visitantes, guardado enquanto o servidor estiver no ar."""
    return {"trava": threading.Lock(), "eventos": [], "dia": date.today(), "chamadas_dia": 0}


@st.cache_data(show_spinner=False)
def calibrar(_chave, modelo):
    return ia.fator_calibracao(_chave, modelo)


modelo = carregar_modelo()
reg = registro()
st.session_state.setdefault("conversa", [])
st.session_state.setdefault("enviadas", 0)

fator, erro_calibracao = 1.0, None
if CHAVE:
    try:
        fator = calibrar(CHAVE, MODELO_IA)
    except Exception as e:
        erro_calibracao = str(e)[:200]

online = bool(CHAVE) and erro_calibracao is None
st.html(f"""
<div class="topo">
  <div class="marca">◆ Token<span>Scope</span></div>
  <div class="status"><i style="background:{VERDE if online else ROSA}"></i>
  {f"{MODELO_IA} conectado" if online else "IA indisponível"}</div>
</div>
<div class="hero">
  <h1>Use IA. Veja quanto ela custa.</h1>
  <p>Converse com o Gemini. Antes de cada resposta, um modelo treinado com <b>18.924 conversas reais</b>
  prevê quantos tokens a IA vai gerar. Depois, mostramos o consumo real e o custo, e o painel acumula o uso de todos.</p>
</div>
""")

aba_chat, aba_painel, aba_algo, aba_sobre = st.tabs([
    ":material/chat_bubble: Assistente", ":material/monitoring: Painel de consumo",
    ":material/model_training: O algoritmo", ":material/info: Sobre"])


def recibo(m):
    dentro = m["p10"] <= m["real"] <= m["p90"]
    rac = f'<span class="chip">raciocínio <b>{br(m["raciocinio"])}</b></span>' if m["raciocinio"] else ""
    st.html(f"""<div class="recibo">
      <span class="chip">previsto <b>~{br(m["previsto"])}</b> ({br(m["p10"])}–{br(m["p90"])})</span>
      <span class="chip {"ok" if dentro else "fora"}">real <b>{br(m["real"])}</b> tokens</span>{rac}
      <span class="chip">entrada <b>{br(m["entrada"])}</b></span>
      <span class="chip">custo <b>{usd(m["custo"])}</b></span>
      <span class="chip">{br(m["segundos"], 1)} s</span></div>""")


# ---------- Assistente ----------
with aba_chat:
    if erro_calibracao:
        st.warning(f"O Gemini recusou a conexão: {erro_calibracao}")
    elif not CHAVE:
        st.warning("A chave do Gemini não está configurada no servidor (Secrets → GEMINI_API_KEY).")

    caixa = st.container()
    with caixa:
        if not st.session_state.conversa:
            st.html('<div class="vazio"><div class="icone">✦</div><b>Pergunte qualquer coisa</b>'
                    'Cada resposta vem com a previsão de tokens, o consumo real e o custo.</div>')
            st.pills("Sugestões", SUGESTOES, key="sugestao", on_change=escolher_sugestao,
                     label_visibility="collapsed", disabled=not online)
        for m in st.session_state.conversa:
            with st.chat_message("user" if m["papel"] == "user" else "assistant",
                                 avatar=":material/person:" if m["papel"] == "user" else ":material/auto_awesome:"):
                st.markdown(m["texto"])
                if m["papel"] == "model":
                    recibo(m)

    restante = LIMITE_SESSAO - st.session_state.enviadas
    bloqueado = not online or restante <= 0 or reg["chamadas_dia"] >= LIMITE_DIA
    pergunta = st.chat_input(
        "Escreva sua mensagem…" if restante > 0 else "Limite de mensagens desta sessão atingido",
        disabled=bloqueado) or st.session_state.pop("pendente", None)
    st.caption(f"{restante} mensagens restantes nesta sessão · preço de tabela: "
               f"US$ {br(PRECO_IN, 2)} (entrada) e US$ {br(PRECO_OUT, 2)} (saída) por 1M tokens")

    if pergunta and not bloqueado:
        conversa = st.session_state.conversa
        prev = modelo.prever(pergunta, "pt-BR", continuacao=bool(conversa))
        conversa.append({"papel": "user", "texto": pergunta})
        with caixa:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(pergunta)
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.html(f'<div class="recibo"><span class="chip">previsão: <b>~{br(prev.mediana * fator)}</b> tokens '
                        f'({br(prev.p10 * fator)}–{br(prev.p90 * fator)})</span></div>')
                try:
                    with st.spinner("Gerando resposta…"):
                        with reg["trava"]:
                            if reg["dia"] != date.today():
                                reg["dia"], reg["chamadas_dia"] = date.today(), 0
                            reg["chamadas_dia"] += 1
                        r = ia.conversar(conversa, CHAVE, MODELO_IA, INSTRUCAO, reserva=MODELO_RESERVA)
                except Exception as e:
                    conversa.pop()
                    codigo = getattr(e, "code", None)
                    if codigo in (500, 503, 504):
                        st.warning("O Gemini está sobrecarregado agora (instabilidade do Google). "
                                   "Tentamos algumas vezes sem sucesso. Envie de novo em alguns segundos.")
                    elif codigo == 429:
                        st.warning("O limite de uso do Gemini foi atingido. Tente de novo em um minuto.")
                    else:
                        st.error(f"Erro ao chamar o Gemini: {str(e)[:200]}")
                else:
                    st.session_state.enviadas += 1
                    resp = {
                        "papel": "model", "texto": r.texto,
                        "previsto": prev.mediana * fator, "p10": prev.p10 * fator, "p90": prev.p90 * fator,
                        "real": r.resposta_tokens, "raciocinio": r.raciocinio_tokens, "entrada": r.prompt_tokens,
                        "segundos": r.segundos,
                        "custo": (r.prompt_tokens * PRECO_IN + (r.resposta_tokens + r.raciocinio_tokens) * PRECO_OUT) / 1e6,
                    }
                    conversa.append(resp)
                    with reg["trava"]:
                        reg["eventos"].append({**{k: v for k, v in resp.items() if k not in ("papel", "texto")},
                                               "quando": datetime.now()})
                    st.rerun()

# ---------- Painel ----------
with aba_painel:
    ev = pd.DataFrame(reg["eventos"])
    if ev.empty:
        st.html('<div class="vazio"><div class="icone">◔</div><b>Ainda não há uso registrado</b>'
                'Converse com o assistente: cada resposta entra aqui com os tokens e o custo reais.</div>')
    else:
        saida = ev.real + ev.raciocinio
        acerto = ((ev.real >= ev.p10) & (ev.real <= ev.p90)).mean()
        cards([
            ("Respostas geradas", br(len(ev)), "por todos os visitantes", False),
            ("Tokens consumidos", br(ev.entrada.sum() + saida.sum()),
             f"{br(ev.entrada.sum())} entrada · {br(saida.sum())} saída", False),
            ("Acerto da previsão", f"{br(100 * acerto)}%", "respostas dentro da faixa (meta ≈ 80%)", False),
            ("Custo total", usd(ev.custo.sum()), f"média {usd(ev.custo.mean())} por resposta", True),
        ])
        g1, g2 = st.columns([1.6, 1], gap="large")
        with g1:
            st.markdown("##### Previsto × real, resposta a resposta")
            x = list(range(1, len(ev) + 1))
            fig = go.Figure()
            fig.add_scatter(x=x + x[::-1], y=list(ev.p90) + list(ev.p10[::-1]), fill="toself",
                            fillcolor="rgba(109,40,217,.10)", line=dict(width=0), name="faixa prevista (80%)",
                            hoverinfo="skip")
            fig.add_scatter(x=x, y=ev.previsto, mode="lines+markers", name="previsto",
                            line=dict(color=ROXO, dash="dot"))
            fig.add_scatter(x=x, y=ev.real, mode="lines+markers", name="real", line=dict(color=VERDE, width=3))
            fig.update_layout(xaxis_title="resposta", yaxis_title="tokens da resposta")
            grafico(fig)
        with g2:
            st.markdown("##### Para onde vai o dinheiro")
            fig = go.Figure(go.Pie(
                labels=["Entrada", "Resposta", "Raciocínio"],
                values=[ev.entrada.sum() * PRECO_IN, ev.real.sum() * PRECO_OUT, ev.raciocinio.sum() * PRECO_OUT],
                hole=.68, marker=dict(colors=[CINZA, ROXO, "#A78BFA"]), textinfo="percent", sort=False))
            grafico(fig)
        st.html(f'<div class="nota">No ritmo atual, <b>10.000 respostas</b> custariam cerca de '
                f'<b>{usd(ev.custo.mean() * 10_000)}</b>. O dado é real: vem dos tokens que o próprio Gemini informou '
                f'em cada chamada.</div>')

# ---------- O algoritmo ----------
with aba_algo:
    p = modelo.pares
    m = modelo.metricas.set_index("modelo")
    gb = m.loc["Gradient Boosting"]
    real, previsto = modelo.custo_mensal(10_000, PRECO_IN, PRECO_OUT)
    erro_mes = 100 * np.median(np.abs(previsto - real) / real)
    st.markdown("#### Como a previsão é feita")
    st.caption("Base pública OpenAssistant (oasst1): 18.924 pares de pergunta → resposta reais. "
               "Família: aprendizado supervisionado (regressão). Validação em 25% dos dados que o modelo não viu.")
    cards([
        ("Conversas na base", br(len(p)), "pares pergunta → resposta", False),
        ("Erro médio", f"{br(gb.mae)} tokens", f"chute ingênuo: {br(m.loc['Baseline (mediana)'].mae)}", False),
        ("Acerto da faixa 80%", f"{br(100 * modelo.cobertura_faixa)}%", "no conjunto de teste", False),
        ("Erro no custo do mês", f"{br(erro_mes, 1)}%", "10.000 respostas (Monte Carlo)", True),
    ])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("##### Distribuição de tokens na base")
        bins = np.logspace(0, 3.8, 45)
        fig = go.Figure()
        for col, nome, cor in [("prompt_tokens", "pergunta", CINZA), ("resp_tokens", "resposta", ROXO)]:
            h, e = np.histogram(p[col], bins=bins)
            fig.add_bar(x=np.sqrt(e[:-1] * e[1:]), y=h, name=nome, marker_color=cor, opacity=.8)
        fig.update_layout(barmode="overlay", xaxis_type="log", xaxis_title="tokens (escala log)", yaxis_title="conversas")
        grafico(fig)
    with g2:
        st.markdown("##### Erro por resposta (menor é melhor)")
        fig = go.Figure(go.Bar(x=m.index, y=m.mae, marker_color=[CINZA, "#A78BFA", ROXO],
                               text=[br(v) for v in m.mae], textposition="outside"))
        fig.update_layout(yaxis_title="erro médio absoluto (tokens)")
        grafico(fig)
    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown("##### O idioma muda o custo")
        d = modelo.idiomas.sort_values("tokens_por_1000_caracteres")
        fig = go.Figure(go.Bar(
            x=d.tokens_por_1000_caracteres, y=[NOMES_IDIOMAS[l] for l in d.lang], orientation="h",
            marker_color=[ROXO if l == "pt-BR" else "#C4B5FD" for l in d.lang],
            hovertemplate="%{x:.0f} tokens<extra></extra>"))
        fig.update_layout(xaxis_title="tokens a cada 1.000 caracteres")
        grafico(fig)
    with g4:
        st.markdown("##### Custo de 10.000 respostas: previsto × real")
        fig = go.Figure()
        fig.add_histogram(x=real, name="real", marker_color=CINZA, opacity=.75, nbinsx=50)
        fig.add_histogram(x=previsto, name="previsto", marker_color=ROXO, opacity=.75, nbinsx=50)
        fig.update_layout(barmode="overlay", xaxis_title="US$", yaxis_title="simulações")
        grafico(fig)
    st.html(f"""
    <div class="nota"><b>O que os dados mostram.</b> Prever o tamanho de <i>uma</i> resposta é difícil: o modelo explica
    ~{br(100 * gb.r2)}% da variação e bate o chute ingênuo por pouco. Mas uma aplicação paga milhares de respostas, e na
    soma os erros se compensam: o custo de 10.000 respostas fica a ~{br(erro_mes)}% do real. A faixa de 80% acerta
    {br(100 * modelo.cobertura_faixa)}% das vezes no teste, e o <b>Painel de consumo</b> mede a mesma coisa contra o Gemini, ao vivo.</div>""")

# ---------- Sobre ----------
with aba_sobre:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
#### O problema
Toda aplicação com IA generativa paga por **token**. Ela conhece a pergunta que envia,
mas **não sabe o tamanho da resposta**, que é a parte mais cara. Sem isso, não dá para planejar o orçamento.

#### A solução
O TokenScope usa a IA de verdade (Gemini) e, a cada resposta, **prevê** os tokens antes e **mede** o consumo real
depois. O painel acumula o uso e mostra se a previsão acerta.

#### A família de algoritmos
- **Aprendizado supervisionado, regressão**: Gradient Boosting (previsão central) e regressão quantílica
  (faixa de 80%), comparados com regressão linear e com um chute ingênuo.
- **Simulação de Monte Carlo** para o custo de milhares de respostas.
""")
    with c2:
        st.markdown(f"""
#### Os dados
[OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1): ~88 mil mensagens reais,
18.924 pares pergunta → melhor resposta, em vários idiomas. Licença Apache 2.0.

#### Tokens de IA
Cada mensagem gasta tokens reais do **{MODELO_IA}**, informados pela própria API. Para comparar previsão e
consumo na mesma unidade, o app mede uma amostra da base com o contador de tokens do Gemini e calibra o modelo.

#### Limitações
- As respostas da base são de voluntários, não de um modelo comercial.
- Modelos com raciocínio gastam tokens extras que não aparecem no texto. Eles entram no custo, mas não na previsão.
- O painel guarda o uso enquanto o servidor estiver no ar.
""")
    st.caption("Atividade de IA aplicada e IA experimental, Prof. Wagner, 3º ciclo.")
