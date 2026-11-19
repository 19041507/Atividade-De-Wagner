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

TINTA, MUDO, LINHA, PAPEL = "#161616", "#6B6A65", "#E3E1DA", "#F7F6F2"
DESTAQUE, BOM, RUIM, CINZA = "#C2410C", "#3F6B4E", "#A33A2B", "#B8B5AC"
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
RESERVA_FIXA = config("GEMINI_MODEL_RESERVA", None)  # se vazio, o app escolhe um Flash leve disponível na chave
PRECO_IN = float(config("PRECO_ENTRADA", 0.30))   # US$ por 1M tokens
PRECO_OUT = float(config("PRECO_SAIDA", 2.50))    # US$ por 1M tokens (inclui raciocínio)
LIMITE_SESSAO = int(config("LIMITE_SESSAO", 15))
LIMITE_DIA = int(config("LIMITE_DIA", 300))

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root { --tinta:#161616; --mudo:#6B6A65; --linha:#E3E1DA; --papel:#F7F6F2; --destaque:#C2410C; --bom:#3F6B4E; --ruim:#A33A2B;
  --serif:'Instrument Serif', Georgia, serif; --sans:'IBM Plex Sans', system-ui, sans-serif; --mono:'IBM Plex Mono', ui-monospace, monospace; }
html, body, .stApp, .stMarkdown, button, input, textarea, [class*="css"] { font-family: var(--sans) !important; }
.stApp { background: var(--papel);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.035'/%3E%3C/svg%3E"); }
::selection { background: #F3D9C9; }
.block-container, div[data-testid="stMainBlockContainer"] { padding: 3rem 3rem 2.5rem; max-width: 100%; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none; }
footer { visibility: hidden; }
@keyframes surge { from { opacity: 0; } to { opacity: 1; } }

/* cabeçalho em estilo de publicação */
.topo { display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap;
  padding-bottom:.7rem; border-bottom:1px solid var(--tinta); box-shadow: 0 3px 0 -2px var(--linha); animation: surge .6s ease both; }
.marca { font-family: var(--serif); font-size: 1.7rem; color: var(--tinta); letter-spacing:-.01em; }
.marca small { font-family: var(--sans); font-size:.78rem; color: var(--mudo); margin-left:.8rem; letter-spacing:.02em; }
.status { font-family: var(--mono); font-size:.76rem; color: var(--mudo); }
.status i { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.4rem; vertical-align:1px; }
.abre { display:grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr); gap: 48px; align-items:end; margin: 2.2rem 0 1.8rem; animation: surge .8s ease both; }
@media (max-width: 960px) { .abre { grid-template-columns: 1fr; gap: 24px; } }
.abre h1 { font-family: var(--serif) !important; font-weight: 400; font-size: clamp(2.6rem, 5.2vw, 4.2rem); line-height: .98;
  letter-spacing: -.02em; color: var(--tinta); margin: 0 0 1rem; padding: 0; }
.abre h1 em { color: var(--destaque); }
.abre p { color: var(--mudo); font-size: 1.02rem; line-height: 1.6; max-width: 600px; margin: 0; }
.numeros { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); border-top:1px solid var(--linha); }
.numeros div { padding: .9rem 0 0 1rem; border-left:1px solid var(--linha); }
.numeros div:first-child { border-left:none; padding-left:0; }
.rotulo { display:block; font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color: var(--mudo); font-weight:500; }
.numeros b { display:block; font-family: var(--mono); font-weight:500; font-size:1.35rem; color: var(--tinta);
  margin-top:.3rem; font-variant-numeric: tabular-nums; }

/* navegação como sumário: texto e sublinhado */
div[data-testid="stButtonGroup"]:has(button[kind^="segmented_control"]) { border-bottom:1px solid var(--linha); margin-bottom: 1.6rem; }
button[kind="segmented_control"], button[kind="segmented_controlActive"] { background: transparent !important; border: none !important;
  border-radius: 0 !important; box-shadow: none !important; padding: .55rem 0 !important; margin-right: 1.8rem;
  color: var(--mudo) !important; position: relative; }
button[kind="segmented_control"]::after, button[kind="segmented_controlActive"]::after { content:""; position:absolute; left:0; right:0;
  bottom:-1px; height:2px; background: var(--tinta); transform: scaleX(0); transform-origin: left; transition: transform .25s ease; }
button[kind="segmented_control"]:hover { color: var(--tinta) !important; }
button[kind="segmented_control"]:hover::after { transform: scaleX(.35); }
button[kind="segmented_controlActive"] { color: var(--tinta) !important; }
button[kind="segmented_controlActive"]::after { transform: scaleX(1); }
button[kind="segmented_control"] p, button[kind="segmented_controlActive"] p { font-size: .95rem; font-weight: 500; }

/* chat */
div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { display:none; }
div[data-testid="stChatMessage"] { background: transparent; padding: .2rem 0 1.2rem; gap: 0; animation: surge .35s ease both; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { padding-bottom: .6rem; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background:#EDEBE4; border-radius: 4px 14px 14px 14px; padding: .65rem .95rem; width: fit-content; max-width: 85%; }
div[data-testid="stChatMessageContent"] p, div[data-testid="stChatMessageContent"] li { font-size: 1rem; line-height: 1.65; color: var(--tinta); }
.quem { font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color: var(--mudo); font-weight:500; margin-bottom:.35rem; }
.conta { font-family: var(--mono); font-size:.76rem; color: var(--mudo); margin-top:.7rem; padding-top:.55rem;
  border-top:1px dashed var(--linha); display:flex; flex-wrap:wrap; gap:.25rem 1.1rem; font-variant-numeric: tabular-nums; }
.conta b { color: var(--tinta); font-weight:500; }
.conta .ok { color: var(--bom); } .conta .fora { color: var(--ruim); }
div[data-testid="stChatInput"] { background:#fff; border:1px solid var(--linha); border-radius: 10px; box-shadow: none; transition: border-color .2s; }
div[data-testid="stChatInput"]:focus-within { border-color: var(--tinta); }

/* começo da conversa */
.comeco h3 { font-family: var(--serif) !important; font-weight:400; font-size: 2rem; color: var(--tinta); margin: .4rem 0 .3rem; padding:0; }
.comeco p { color: var(--mudo); margin: 0 0 1rem; }
button[kind="pills"] { background: transparent !important; border: 1px solid var(--linha) !important; border-radius: 6px !important;
  color: var(--tinta) !important; transition: border-color .2s, background .2s; }
button[kind="pills"]:hover { border-color: var(--tinta) !important; background: #fff !important; }
button[kind="pills"] p::after { content: "  →"; color: var(--mudo); }

/* cupom da conversa */
.cupom { background:#FFFEFB; border:1px solid var(--linha); border-bottom:none; padding: 1.2rem 1.2rem 1.4rem; font-family: var(--mono);
  font-size:.8rem; color: var(--tinta); position: relative; animation: surge .6s ease both;
  box-shadow: 0 1px 0 rgba(0,0,0,.02), 0 10px 20px -16px rgba(0,0,0,.25); }
.cupom::after { content:""; position:absolute; left:-1px; right:-1px; bottom:-8px; height:8px;
  background: linear-gradient(-45deg, transparent 6px, #FFFEFB 0) 0 0/12px 8px repeat-x,
              linear-gradient(45deg, transparent 6px, #FFFEFB 0) 0 0/12px 8px repeat-x; }
.cupom .tit { text-align:center; letter-spacing:.2em; font-size:.72rem; color: var(--mudo); }
.cupom hr { border:none; border-top:1px dashed #CFCBC0; margin:.7rem 0; }
.cupom .l { display:flex; align-items:baseline; gap:.4rem; margin:.28rem 0; }
.cupom .l span:nth-child(2) { flex:1; border-bottom:1px dotted #CFCBC0; transform: translateY(-3px); }
.cupom .total { font-weight:500; font-size:.92rem; }
.cupom .rodape { color: var(--mudo); font-size:.72rem; line-height:1.6; margin-top:.8rem; }

/* painéis de números e notas */
.faixa { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); border-top:1px solid var(--tinta);
  border-bottom:1px solid var(--linha); margin: .2rem 0 2rem; }
.faixa div { padding: 1rem 1.2rem 1.1rem; border-left:1px solid var(--linha); }
.faixa div:first-child { border-left:none; padding-left:0; }
.faixa b { display:block; font-family: var(--mono); font-weight:500; font-size:1.7rem; color: var(--tinta); margin:.35rem 0 .2rem; font-variant-numeric: tabular-nums; }
.faixa b.acento { color: var(--destaque); }
.faixa small { color: var(--mudo); font-size:.8rem; }
.secao h2 { font-family: var(--serif) !important; font-weight:400; font-size: 2.1rem; color: var(--tinta); margin: 0 0 .3rem; padding:0; }
.secao p { color: var(--mudo); margin: 0 0 1.4rem; max-width: 760px; }
.graf { font-size:.95rem; font-weight:600; color: var(--tinta); margin: 0 0 .1rem; }
.graf + .sub { font-size:.82rem; color: var(--mudo); margin-bottom:.2rem; }
.nota { border-left:2px solid var(--destaque); padding:.2rem 0 .2rem 1rem; color:#3A3935; font-size:.98rem; line-height:1.65;
  max-width: 880px; margin: 1rem 0; }
.vazio { padding: 2.5rem 0; color: var(--mudo); }
.vazio h3 { font-family: var(--serif) !important; font-weight:400; font-size: 1.8rem; color: var(--tinta); margin:0 0 .3rem; padding:0; }
.stMarkdown h4 { font-family: var(--serif) !important; font-weight:400; font-size:1.6rem; }
</style>
""")


def faixa(itens):
    html = "".join(
        f'<div><span class="rotulo">{r}</span><b class="{"acento" if d else ""}">{v}</b><small>{h}</small></div>'
        for r, v, h, d in itens
    )
    st.html(f'<div class="faixa">{html}</div>')


def titulo_grafico(t, sub=""):
    st.html(f'<div class="graf">{t}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""))


def grafico(fig, altura=340):
    fig.update_layout(
        height=altura, margin=dict(l=4, r=4, t=30, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans", color=MUDO, size=12),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12)), hoverlabel=dict(font_family="IBM Plex Mono"),
        bargap=.08,
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", linecolor=LINHA, zeroline=False, ticks="outside", tickcolor=LINHA)
    fig.update_yaxes(gridcolor=LINHA, griddash="dot", zeroline=False, linecolor="rgba(0,0,0,0)")
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


@st.cache_data(show_spinner=False, ttl=3600)
def reserva_automatica(_chave, principal):
    return ia.escolher_reserva(_chave, principal)


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
MODELO_RESERVA = RESERVA_FIXA
if online and not MODELO_RESERVA:
    try:
        MODELO_RESERVA = reserva_automatica(CHAVE, MODELO_IA)
    except Exception:
        MODELO_RESERVA = None

eventos = reg["eventos"]
st.html(f"""
<div class="topo">
  <div class="marca">TokenScope<small>um estudo sobre o custo de usar IA</small></div>
  <div class="status"><i style="background:{BOM if online else RUIM}"></i>{MODELO_IA} · {"online" if online else "indisponível"}</div>
</div>
<div class="abre">
  <div>
    <h1>Use IA. Veja quanto <em>ela custa.</em></h1>
    <p>Converse com o Gemini. Antes de cada resposta, um modelo treinado com 18.924 conversas reais
    prevê quantos tokens a IA vai gerar. Depois, mostramos o consumo real e o custo.</p>
  </div>
  <div class="numeros">
    <div><span class="rotulo">Respostas geradas</span><b>{br(len(eventos))}</b></div>
    <div><span class="rotulo">Tokens consumidos</span><b>{br(sum(e["entrada"] + e["real"] + e["raciocinio"] for e in eventos))}</b></div>
    <div><span class="rotulo">Custo total</span><b>{usd(sum(e["custo"] for e in eventos))}</b></div>
  </div>
</div>
""")

SECOES = {"Assistente": "Assistente", "Painel": "Painel de consumo", "Algoritmo": "O algoritmo", "Sobre": "Sobre"}
secao = st.segmented_control("Seção", list(SECOES), format_func=SECOES.get, default="Assistente",
                             key="secao", label_visibility="collapsed") or "Assistente"


def conta(m):
    dentro = m["p10"] <= m["real"] <= m["p90"]
    partes = [
        f'previsto <b>~{br(m["previsto"])}</b> ({br(m["p10"])}–{br(m["p90"])})',
        f'real <b>{br(m["real"])}</b> <span class="{"ok" if dentro else "fora"}">{"✓ na faixa" if dentro else "✕ fora da faixa"}</span>',
    ]
    if m["raciocinio"]:
        partes.append(f'raciocínio <b>{br(m["raciocinio"])}</b>')
    partes += [f'entrada <b>{br(m["entrada"])}</b>', f'<b>{usd(m["custo"])}</b>', f'{br(m["segundos"], 1)} s']
    if m.get("modelo") and m["modelo"] != MODELO_IA:
        partes.append(f'via {m["modelo"]}')
    st.html(f'<div class="conta">{"".join(f"<span>{p}</span>" for p in partes)}</div>')


def quem(nome):
    st.html(f'<div class="quem">{nome}</div>')


# ---------- Assistente ----------
if secao == "Assistente":
    if erro_calibracao:
        st.warning(f"O Gemini recusou a conexão: {erro_calibracao}")
    elif not CHAVE:
        st.warning("A chave do Gemini não está configurada no servidor (Secrets → GEMINI_API_KEY).")

    col_chat, col_lado = st.columns([2.3, 1], gap="large")
    restante = LIMITE_SESSAO - st.session_state.enviadas
    bloqueado = not online or restante <= 0 or reg["chamadas_dia"] >= LIMITE_DIA

    with col_lado:
        respostas = [m for m in st.session_state.conversa if m["papel"] == "model"]
        acertos = sum(m["p10"] <= m["real"] <= m["p90"] for m in respostas)
        ent = sum(m["entrada"] for m in respostas)
        sai = sum(m["real"] + m["raciocinio"] for m in respostas)

        def l(rot, val):
            return f'<div class="l"><span>{rot}</span><span></span><span>{val}</span></div>'
        st.html(f"""<div class="cupom">
          <div class="tit">CUPOM DA CONVERSA</div>
          <div class="tit">{datetime.now().strftime("%d/%m/%Y %H:%M")}</div><hr>
          {l("respostas", br(len(respostas)))}
          {l("tokens de entrada", br(ent))}
          {l("tokens de saída", br(sai))}
          {l("previsões na faixa", f"{acertos}/{len(respostas)}" if respostas else "–")}<hr>
          <div class="l total"><span>TOTAL</span><span></span><span>{usd(sum(m["custo"] for m in respostas))}</span></div><hr>
          <div class="rodape">preço de tabela por 1M tokens: US$ {br(PRECO_IN, 2)} entrada · US$ {br(PRECO_OUT, 2)} saída<br>
          raciocínio do modelo é cobrado como saída<br>{restante} mensagens restantes nesta sessão</div>
        </div>""")

    with col_chat:
        caixa = st.container()
        with caixa:
            boas_vindas = st.empty()
            if not st.session_state.conversa:
                with boas_vindas.container():
                    st.html('<div class="comeco"><h3>Comece com uma pergunta.</h3>'
                            '<p>Cada resposta vem com a previsão de tokens, o consumo real e o custo.</p></div>')
                    st.pills("Sugestões", SUGESTOES, key="sugestao", on_change=escolher_sugestao,
                             label_visibility="collapsed", disabled=not online)
            for m in st.session_state.conversa:
                with st.chat_message("user" if m["papel"] == "user" else "assistant"):
                    if m["papel"] == "model":
                        quem("Gemini")
                    st.markdown(m["texto"])
                    if m["papel"] == "model":
                        conta(m)
        pergunta = st.chat_input(
            "Escreva sua pergunta…" if restante > 0 else "Limite de mensagens desta sessão atingido",
            disabled=bloqueado) or st.session_state.pop("pendente", None)

    if pergunta and not bloqueado:
        boas_vindas.empty()
        conversa = st.session_state.conversa
        prev = modelo.prever(pergunta, "pt-BR", continuacao=bool(conversa))
        conversa.append({"papel": "user", "texto": pergunta})
        with caixa:
            with st.chat_message("user"):
                st.markdown(pergunta)
            with st.chat_message("assistant"):
                quem("Gemini")
                saida = st.empty()
                saida.markdown(f"_Pensando… previsão de ~{br(prev.mediana * fator)} tokens "
                               f"({br(prev.p10 * fator)}–{br(prev.p90 * fator)})_")
                try:
                    with reg["trava"]:
                        if reg["dia"] != date.today():
                            reg["dia"], reg["chamadas_dia"] = date.today(), 0
                        reg["chamadas_dia"] += 1
                    r = ia.conversar(conversa, CHAVE, MODELO_IA, INSTRUCAO, reserva=MODELO_RESERVA,
                                     ao_receber=lambda t: saida.markdown(t + " ▍"))
                except ia.GeminiIndisponivel as e:
                    conversa.pop()
                    saida.empty()
                    if "429" in str(e):
                        st.warning("O limite de uso do Gemini foi atingido. Tente de novo em um minuto.")
                    else:
                        st.warning("O Gemini não respondeu a tempo (instabilidade do Google). "
                                   "Tentamos algumas vezes sem sucesso. Envie de novo em alguns segundos.")
                    st.caption(f"Detalhe técnico: {str(e)[:160]}")
                except Exception as e:
                    conversa.pop()
                    saida.empty()
                    st.error(f"Erro ao chamar o Gemini: {str(e)[:200]}")
                else:
                    st.session_state.enviadas += 1
                    resp = {
                        "papel": "model", "texto": r.texto,
                        "previsto": prev.mediana * fator, "p10": prev.p10 * fator, "p90": prev.p90 * fator,
                        "real": r.resposta_tokens, "raciocinio": r.raciocinio_tokens, "entrada": r.prompt_tokens,
                        "segundos": r.segundos, "modelo": r.modelo,
                        "custo": (r.prompt_tokens * PRECO_IN + (r.resposta_tokens + r.raciocinio_tokens) * PRECO_OUT) / 1e6,
                    }
                    conversa.append(resp)
                    with reg["trava"]:
                        reg["eventos"].append({**{k: v for k, v in resp.items() if k not in ("papel", "texto")},
                                               "quando": datetime.now()})
                    st.rerun()

# ---------- Painel ----------
if secao == "Painel":
    st.html('<div class="secao"><h2>Painel de consumo</h2><p>Tudo o que foi gasto pelos visitantes, com os tokens '
            'que o próprio Gemini informou em cada chamada.</p></div>')
    ev = pd.DataFrame(eventos)
    if ev.empty:
        st.html('<div class="vazio"><h3>Nada por aqui ainda.</h3>'
                'Converse com o assistente: cada resposta entra aqui com os tokens e o custo reais.</div>')
    else:
        saida = ev.real + ev.raciocinio
        acerto = ((ev.real >= ev.p10) & (ev.real <= ev.p90)).mean()
        faixa([
            ("Respostas geradas", br(len(ev)), "por todos os visitantes", False),
            ("Tokens consumidos", br(ev.entrada.sum() + saida.sum()),
             f"{br(ev.entrada.sum())} entrada · {br(saida.sum())} saída", False),
            ("Acerto da previsão", f"{br(100 * acerto)}%", "dentro da faixa de 80%", False),
            ("Custo total", usd(ev.custo.sum()), f"média de {usd(ev.custo.mean())} por resposta", True),
        ])
        g1, g2 = st.columns([1.7, 1], gap="large")
        with g1:
            titulo_grafico("Previsto × real, resposta a resposta", "a área clara é a faixa em que o modelo esperava acertar 80% das vezes")
            x = list(range(1, len(ev) + 1))
            fig = go.Figure()
            fig.add_scatter(x=x + x[::-1], y=list(ev.p90) + list(ev.p10[::-1]), fill="toself",
                            fillcolor="rgba(22,22,22,.06)", line=dict(width=0), name="faixa prevista", hoverinfo="skip")
            fig.add_scatter(x=x, y=ev.previsto, mode="lines", name="previsto", line=dict(color=TINTA, dash="dot", width=1.5))
            fig.add_scatter(x=x, y=ev.real, mode="lines+markers", name="real", line=dict(color=DESTAQUE, width=2),
                            marker=dict(size=6))
            fig.update_layout(xaxis_title="resposta", yaxis_title="tokens")
            grafico(fig)
        with g2:
            titulo_grafico("Para onde vai o dinheiro", "participação de cada tipo de token no custo")
            partes = pd.Series({"Entrada": ev.entrada.sum() * PRECO_IN, "Resposta": ev.real.sum() * PRECO_OUT,
                                "Raciocínio": ev.raciocinio.sum() * PRECO_OUT})
            partes = 100 * partes / partes.sum()
            fig = go.Figure(go.Bar(x=partes.values, y=partes.index, orientation="h",
                                   marker_color=[CINZA, DESTAQUE, TINTA], text=[f"{br(v)}%" for v in partes.values],
                                   textposition="outside", hovertemplate="%{x:.1f}%<extra></extra>"))
            fig.update_layout(xaxis=dict(visible=False, range=[0, 115]), yaxis=dict(autorange="reversed"))
            grafico(fig, 260)
        st.html(f'<div class="nota">No ritmo atual, <b>10.000 respostas</b> custariam cerca de '
                f'<b>{usd(ev.custo.mean() * 10_000)}</b>.</div>')

# ---------- O algoritmo ----------
if secao == "Algoritmo":
    p = modelo.pares
    m = modelo.metricas.set_index("modelo")
    gb = m.loc["Gradient Boosting"]
    real, previsto = modelo.custo_mensal(10_000, PRECO_IN, PRECO_OUT)
    erro_mes = 100 * np.median(np.abs(previsto - real) / real)
    st.html('<div class="secao"><h2>Como a previsão é feita</h2><p>Base pública OpenAssistant (oasst1), com 18.924 '
            'pares de pergunta e resposta reais. Aprendizado supervisionado (regressão), validado em 25% dos dados '
            'que o modelo não viu durante o treino.</p></div>')
    faixa([
        ("Conversas na base", br(len(p)), "pares pergunta → resposta", False),
        ("Erro médio", f"{br(gb.mae)} tokens", f"chute ingênuo: {br(m.loc['Baseline (mediana)'].mae)}", False),
        ("Acerto da faixa 80%", f"{br(100 * modelo.cobertura_faixa)}%", "no conjunto de teste", False),
        ("Erro no custo do mês", f"{br(erro_mes, 1)}%", "10.000 respostas (Monte Carlo)", True),
    ])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        titulo_grafico("Quantos tokens têm perguntas e respostas", "a resposta costuma ser ~10 vezes maior")
        bins = np.logspace(0, 3.8, 45)
        fig = go.Figure()
        for col, nome, cor in [("prompt_tokens", "pergunta", CINZA), ("resp_tokens", "resposta", DESTAQUE)]:
            h, e = np.histogram(p[col], bins=bins)
            fig.add_bar(x=np.sqrt(e[:-1] * e[1:]), y=h, name=nome, marker_color=cor, opacity=.85)
        fig.update_layout(barmode="overlay", xaxis_type="log", xaxis_title="tokens (escala log)", yaxis_title="conversas")
        grafico(fig)
    with g2:
        titulo_grafico("Erro por resposta", "quanto menor, melhor")
        fig = go.Figure(go.Bar(x=m.index, y=m.mae, marker_color=[CINZA, CINZA, DESTAQUE],
                               text=[br(v) for v in m.mae], textposition="outside"))
        fig.update_layout(yaxis_title="erro médio absoluto (tokens)", yaxis_range=[0, m.mae.max() * 1.18])
        grafico(fig)
    g3, g4 = st.columns(2, gap="large")
    with g3:
        titulo_grafico("O idioma muda o custo", "tokens gastos a cada 1.000 caracteres")
        d = modelo.idiomas.sort_values("tokens_por_1000_caracteres")
        fig = go.Figure(go.Bar(
            x=d.tokens_por_1000_caracteres, y=[NOMES_IDIOMAS[l] for l in d.lang], orientation="h",
            marker_color=[DESTAQUE if l == "pt-BR" else CINZA for l in d.lang],
            hovertemplate="%{x:.0f} tokens<extra></extra>"))
        grafico(fig)
    with g4:
        titulo_grafico("Custo de 10.000 respostas", "2.000 simulações: o previsto fica colado no real")
        fig = go.Figure()
        fig.add_histogram(x=real, name="real", marker_color=CINZA, opacity=.8, nbinsx=50)
        fig.add_histogram(x=previsto, name="previsto", marker_color=DESTAQUE, opacity=.75, nbinsx=50)
        fig.update_layout(barmode="overlay", xaxis_title="US$", yaxis_title="simulações")
        grafico(fig)
    st.html(f"""
    <div class="nota"><b>O que os dados mostram.</b> Prever o tamanho de <i>uma</i> resposta é difícil: o modelo explica
    ~{br(100 * gb.r2)}% da variação e bate o chute ingênuo por pouco. Mas uma aplicação paga milhares de respostas, e na
    soma os erros se compensam: o custo de 10.000 respostas fica a ~{br(erro_mes)}% do real. A faixa de 80% acerta
    {br(100 * modelo.cobertura_faixa)}% das vezes no teste, e o painel de consumo mede a mesma coisa contra o Gemini, ao vivo.</div>""")

# ---------- Sobre ----------
if secao == "Sobre":
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
