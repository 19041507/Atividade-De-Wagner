import os
import threading
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ia, piloto, producao
from core.banco import conectar
from core.modelo import DADOS, NOMES_IDIOMAS, ModeloTokens

st.set_page_config(page_title="TokenScope", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

TINTA, MUDO, LINHA, PAPEL = "#161616", "#6B6A65", "#E3E1DA", "#F7F6F2"
DESTAQUE, BOM, RUIM, CINZA = "#C2410C", "#3F6B4E", "#A33A2B", "#B8B5AC"
EXECUCOES = DADOS / "execucoes"
META_ECONOMIA, META_QUALIDADE, META_CONCORDANCIA = 0.30, 0.90, 0.80
CORES_ESTRATEGIA = {"Sempre Flash": CINZA, "Sempre Lite": CINZA, "Regra fixa": MUDO, "Piloto automático": DESTAQUE,
                    "Escolha perfeita": TINTA}

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

/* metas */
.metas { border-top:1px solid var(--tinta); margin: 0 0 2rem; }
.meta { display:grid; grid-template-columns: 1.6rem minmax(0,1fr) auto; gap: 1rem; padding:.7rem 0; border-bottom:1px solid var(--linha);
  align-items:baseline; font-size:.95rem; color:#3A3935; }
.meta b { font-family: var(--mono); font-weight:500; color: var(--tinta); font-variant-numeric: tabular-nums; }
.meta .s { font-family: var(--mono); }
.meta .ok { color: var(--bom); } .meta .fora { color: var(--ruim); } .meta .pend { color: var(--mudo); }

/* decisão e respostas lado a lado */
.pedido { font-size:1.05rem; line-height:1.6; color: var(--tinta); border-top:1px solid var(--tinta); padding:1rem 0 .6rem; }
.etiquetas { font-family: var(--mono); font-size:.78rem; color: var(--mudo); display:flex; flex-wrap:wrap; gap:.3rem 1.2rem; margin-bottom:1.2rem; }
.etiquetas b { color: var(--tinta); font-weight:500; }
.etiquetas .lite { color: var(--bom); } .etiquetas .flash { color: var(--destaque); }

/* como funciona */
.passos { counter-reset: passo; border-top:1px solid var(--tinta); margin-bottom: 2.4rem; }
.passo { display:grid; grid-template-columns: 4.2rem minmax(0,1.2fr) minmax(0,1fr); gap: 1.4rem; padding: 1.2rem 0;
  border-bottom:1px solid var(--linha); align-items:start; }
@media (max-width: 900px) { .passo { grid-template-columns: 3rem 1fr; } .passo .formula { grid-column: 2; } }
.passo::before { counter-increment: passo; content: counter(passo, decimal-leading-zero); font-family: var(--serif);
  font-size: 2.2rem; line-height: 1; color: var(--destaque); }
.passo h4 { font-family: var(--sans) !important; font-size: 1rem !important; font-weight:600; color: var(--tinta); margin:0 0 .3rem; padding:0; }
.passo p { color:#3A3935; font-size:.93rem; line-height:1.6; margin:0; }
.passo .quemfaz { display:inline-block; font-size:.66rem; letter-spacing:.09em; text-transform:uppercase; font-weight:500;
  color: var(--mudo); border:1px solid var(--linha); padding:.05rem .45rem; border-radius:3px; margin-bottom:.45rem; }
.passo .quemfaz.ml { color: var(--destaque); border-color:#EBC9B6; }
.passo .quemfaz.gem { color: var(--bom); border-color:#C9D9CE; }
.formula { font-family: var(--mono); font-size:.8rem; color: var(--tinta); background:#FFFEFB; border:1px solid var(--linha);
  padding:.75rem .9rem; line-height:1.75; white-space: pre-wrap; font-variant-numeric: tabular-nums; }
.formula i { color: var(--mudo); font-style: normal; }

/* assistente */
div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { display:none; }
div[data-testid="stChatMessage"] { background: transparent; padding: .2rem 0 1.2rem; gap: 0; animation: surge .35s ease both; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { padding-bottom: .6rem; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background:#EDEBE4; border-radius: 4px 14px 14px 14px; padding: .65rem .95rem; width: fit-content; max-width: 85%; }
div[data-testid="stChatMessageContent"] p, div[data-testid="stChatMessageContent"] li { font-size: 1rem; line-height: 1.65; color: var(--tinta); }
div[data-testid="stChatInput"] { background:#fff; border:1px solid var(--linha); border-radius: 10px; box-shadow: none; }
div[data-testid="stChatInput"]:focus-within { border-color: var(--tinta); }
.etiquetas.conta { margin: .7rem 0 .3rem; padding-top:.55rem; border-top:1px dashed var(--linha); }
.comeco h3 { font-family: var(--serif) !important; font-weight:400; font-size: 2rem; color: var(--tinta); margin: .4rem 0 .3rem; padding:0; }
.comeco p { color: var(--mudo); margin: 0 0 1rem; }
.lado { background:#FFFEFB; border:1px solid var(--linha); padding: 1.1rem 1.2rem; font-size:.85rem; color:#3A3935; }
.lado .rotulo { margin-bottom:.4rem; }
.lado .l { display:flex; justify-content:space-between; gap:1rem; padding:.3rem 0; border-bottom:1px dotted var(--linha); }
.lado .l b { font-family: var(--mono); font-weight:500; color: var(--tinta); font-variant-numeric: tabular-nums; }
.lado p { color: var(--mudo); font-size:.78rem; line-height:1.55; margin:1rem 0 0; }
.lado .rotulo + .l { border-top:1px solid var(--tinta); }
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


def pct(v):
    return f"{br(100 * v)}%"


def secao_titulo(titulo, texto):
    st.html(f'<div class="secao"><h2>{titulo}</h2><p>{texto}</p></div>')


def config(nome, padrao=None):
    try:
        v = st.secrets.get(nome)
    except Exception:
        v = None
    return v if v not in (None, "") else os.environ.get(nome, padrao)


CHAVE = config("GEMINI_API_KEY")
LIMITE_SESSAO = int(config("LIMITE_SESSAO", 20))  # perguntas por pessoa (código no link) por dia
LIMITE_DIA = int(config("LIMITE_DIA", 300))
ORCAMENTO_DIA = float(config("ORCAMENTO_DIA", 1.00))  # US$: passou disso, só o Lite até o dia seguinte
MAX_TOKENS_RESPOSTA = int(config("MAX_TOKENS_RESPOSTA", 2048))  # teto de saída por resposta (inclui raciocínio)
PRECOS = producao.precos(config)
banco = conectar(config("SUPABASE_URL"), config("SUPABASE_KEY"))


@st.cache_resource(show_spinner="Carregando o previsor treinado com 18.924 conversas reais…")
def carregar_modelo():
    return ModeloTokens()


@st.cache_data(ttl=600, show_spinner=False)
def carregar_execucoes():
    return piloto.carregar_historico(EXECUCOES), piloto.Roteador.carregar(EXECUCOES / "roteador.json")


@st.cache_data(ttl=30, show_spinner=False)
def carregar_reais():
    return banco.listar() if banco else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_modelos(_chave):
    return producao.modelos(config, _chave)


@st.cache_resource(show_spinner="Calculando a regra inicial…")
def roteador_inicial(_modelo):
    return piloto.regra_inicial(_modelo, pd.read_csv(DADOS / "fila_pedidos.csv"))


def ultimo_diario():
    arquivos = sorted(EXECUCOES.glob("????-??-??*.md"))
    return (arquivos[-1].stem, arquivos[-1].read_text()) if arquivos else None


modelo = carregar_modelo()
hist, roteador = carregar_execucoes()
rot = roteador or roteador_inicial(modelo)
tem_controle = not hist.empty
erro_banco = None
try:
    reais = carregar_reais()
except Exception as e:
    reais, erro_banco = pd.DataFrame(), str(e)[:200]
real = producao.resumo_real(reais)
MODELOS, erro_ia = None, None
if CHAVE:
    try:
        MODELOS = carregar_modelos(CHAVE)
    except Exception as e:
        erro_ia = str(e)[:200]
online = MODELOS is not None

if tem_controle:
    aval = piloto.avaliar(hist, rot.limiar_p90)
    pil = aval.loc["Piloto automático"]

# A manchete usa o uso real quando existe; antes disso, o teste controlado.
if real and real["pedidos"] >= 10:
    manchete = {"economia": real["economia"], "qualidade": real["qualidade"], "pedidos": real["pedidos"], "fonte": "uso real"}
elif tem_controle:
    manchete = {"economia": pil.economia, "qualidade": pil.qualidade, "pedidos": len(hist), "fonte": "teste controlado"}
else:
    manchete = None

status = (f"roteador {'regra' if rot.tipo == 'regra' else 'v' + str(rot.versao)} · "
          f"{MODELOS['lite'] + ' / ' + MODELOS['flash'] if online else 'Gemini indisponível'}")
st.html(f"""
<div class="topo">
  <div class="marca">TokenScope<small>piloto automático de custo de IA</small></div>
  <div class="status"><i style="background:{BOM if online else RUIM}"></i>{status}</div>
</div>
<div class="abre">
  <div>
    <h1>A IA escolhe o modelo. <em>Você paga menos.</em></h1>
    <p>Um assistente em que cada pergunta é roteada sozinha para o Gemini Flash-Lite (barato) ou para o Flash
    (mais capaz). Uma segunda IA audita uma amostra das respostas, e o piloto se retreina todo dia com o que aprendeu.</p>
  </div>
  <div class="numeros">
    <div><span class="rotulo">Economia</span><b>{pct(manchete["economia"]) if manchete else "–"}</b></div>
    <div><span class="rotulo">Qualidade</span><b>{pct(manchete["qualidade"]) if manchete and manchete["qualidade"] is not None else "–"}</b></div>
    <div><span class="rotulo">Pedidos</span><b>{br(manchete["pedidos"]) if manchete else "–"}</b></div>
  </div>
</div>
""")

SECOES = {"Assistente": "Assistente", "Resultados": "Resultados", "Bastidores": "Bastidores"}
secao = st.segmented_control("Seção", list(SECOES), format_func=SECOES.get, default="Assistente",
                             key="secao", label_visibility="collapsed") or "Assistente"


def salvar_feedback(id_):
    valor = st.session_state.get(f"fb_{id_}")
    if banco and valor is not None:
        try:
            banco.atualizar(id_, {"feedback": 1 if valor == 1 else -1})
            carregar_reais.clear()
        except Exception:
            st.toast("Não consegui salvar a avaliação.")


def rodape(m):
    economia = 1 - m["custo"] / m["custo_flash"] if m["custo_flash"] else 0
    partes = [f'<b class="{m["rota"]}">{m["rota"].capitalize()}</b>', m["motivo"], f'<b>{usd(m["custo"])}</b>']
    if m["rota"] == "lite":
        partes.append(f'o Flash custaria ~{usd(m["custo_flash"])} (−{pct(economia)})')
    if m.get("segundos"):
        partes.append(f'{br(m["segundos"], 1)} s')
    if m.get("cortada"):
        partes.append(f'cortada no limite de {br(MAX_TOKENS_RESPOSTA)} tokens')
    st.html(f'<div class="etiquetas conta">{"".join(f"<span>{p}</span>" for p in partes)}</div>')
    if m.get("id"):
        padrao = {1: 1, -1: 0}.get(m.get("feedback"))
        st.feedback("thumbs", key=f"fb_{m['id']}", default=padrao, on_change=salvar_feedback, args=(m["id"],))


def novo_codigo():
    return uuid.uuid4().hex[:12]


def conversa_do_banco(sessao: str) -> list[dict]:
    """Reabre uma conversa a partir do banco: cada linha é uma pergunta e a resposta entregue."""
    if reais.empty:
        return []
    msgs = []
    for r in reais[reais.sessao == sessao].sort_values("criado_em").itertuples():
        msgs.append({"papel": "user", "texto": r.pedido})
        msgs.append({"papel": "model", "texto": r.resposta, "id": r.id, "rota": r.rota, "motivo": r.motivo,
                     "custo": r.custo, "custo_flash": r.custo_flash_estimado if r.rota == "lite" else r.custo,
                     "feedback": None if pd.isna(r.feedback) else int(r.feedback)})
    return msgs


def abrir_conversa(codigo: str):
    st.query_params["c"] = codigo
    st.session_state.pop("conversa_aberta", None)


# ---------- Assistente ----------
if secao == "Assistente":
    # Sem login: o código do usuário (u) e o da conversa (c) ficam no endereço da página.
    # Guardar o link nos favoritos guarda as conversas; quem tiver o link também as vê.
    if "u" not in st.query_params:
        st.query_params["u"] = novo_codigo()
    if "c" not in st.query_params:
        st.query_params["c"] = novo_codigo()
    usuario, codigo = st.query_params["u"], st.query_params["c"]
    sessao = f"{usuario}/{codigo}"
    if st.session_state.get("conversa_aberta") != sessao:
        st.session_state.conversa = conversa_do_banco(sessao)
        st.session_state.conversa_aberta = sessao

    gasto_hoje, pedidos_hoje = producao.gasto_e_contagem_hoje(reais)
    meus = reais[reais.sessao.fillna("").str.startswith(f"{usuario}/")] if not reais.empty else reais
    _, minhas_hoje = producao.gasto_e_contagem_hoje(meus)
    minhas_hoje = max(minhas_hoje, sum(m["papel"] == "user" for m in st.session_state.conversa if not m.get("id")))
    restante = LIMITE_SESSAO - minhas_hoje
    if erro_ia:
        st.warning(f"O Gemini recusou a conexão: {erro_ia}")
    elif not CHAVE:
        st.warning("A chave do Gemini não está configurada no servidor (Secrets → GEMINI_API_KEY).")
    if not banco:
        st.info("O banco (Supabase) não está configurado: o assistente responde, mas nada é medido nem guardado.")
    elif erro_banco:
        st.warning(f"Não consegui ler o banco: {erro_banco}")
    bloqueado = not online or restante <= 0 or pedidos_hoje >= LIMITE_DIA

    col_lista, col_chat, col_lado = st.columns([0.9, 2.4, 1], gap="large")
    with col_lista:
        st.button("+ Nova conversa", on_click=abrir_conversa, args=(novo_codigo(),), width="stretch")
        if not meus.empty:
            conversas = (meus.sort_values("criado_em").groupby("sessao")
                         .agg(titulo=("pedido", "first"), ultima=("criado_em", "max"))
                         .sort_values("ultima", ascending=False))
            st.html('<span class="rotulo" style="margin-top:1rem">Suas conversas</span>')
            for sess, c in conversas.head(30).iterrows():
                atual = sess == sessao
                st.button(c.titulo[:42] + ("…" if len(c.titulo) > 42 else ""), key=f"conv_{sess}",
                          type="secondary" if atual else "tertiary", disabled=atual, width="stretch",
                          on_click=abrir_conversa, args=(sess.split("/", 1)[1],))
            st.caption("Salve o link desta página para voltar às suas conversas.")

    with col_lado:
        da_conversa = [m for m in st.session_state.conversa if m["papel"] == "model"]
        pago = sum(m["custo"] for m in da_conversa)
        flash = sum(m["custo_flash"] for m in da_conversa)
        st.html(f"""<div class="lado">
          <span class="rotulo">Nesta conversa</span>
          <div class="l"><span>respostas</span><b>{len(da_conversa)}</b></div>
          <div class="l"><span>no Lite</span><b>{sum(m["rota"] == "lite" for m in da_conversa)}</b></div>
          <div class="l"><span>pago</span><b>{usd(pago)}</b></div>
          <div class="l"><span>se fosse tudo Flash</span><b>{usd(flash)}</b></div>
          <span class="rotulo" style="margin-top:1.2rem">Hoje, todos os usuários</span>
          <div class="l"><span>pedidos</span><b>{pedidos_hoje} de {LIMITE_DIA}</b></div>
          <div class="l"><span>gasto</span><b>{usd(gasto_hoje)} de {usd(ORCAMENTO_DIA)}</b></div>
          <p>Você ainda pode fazer {max(restante, 0)} perguntas hoje. As perguntas ficam guardadas para medir e
          treinar o piloto: não envie dados pessoais.</p>
        </div>""")

    with col_chat:
        caixa = st.container()
        with caixa:
            if not st.session_state.conversa:
                st.html('<div class="comeco"><h3>Pergunte qualquer coisa.</h3>'
                        '<p>O piloto escolhe o modelo, e você vê qual respondeu, por quê e quanto custou.</p></div>')
            for m in st.session_state.conversa:
                with st.chat_message("user" if m["papel"] == "user" else "assistant"):
                    st.markdown(m["texto"])
                    if m["papel"] == "model":
                        rodape(m)
        pergunta = st.chat_input(
            "Escreva sua pergunta…" if restante > 0 else "Você atingiu o limite de perguntas de hoje",
            disabled=bloqueado)

    if pergunta and not bloqueado:
        conversa = st.session_state.conversa
        d = producao.decidir(pergunta, rot, modelo, gasto_hoje, ORCAMENTO_DIA)
        conversa.append({"papel": "user", "texto": pergunta})
        mensagens = [{"papel": m["papel"], "texto": m["texto"]} for m in conversa]
        with caixa:
            with st.chat_message("user"):
                st.markdown(pergunta)
            with st.chat_message("assistant"):
                saida = st.empty()
                saida.markdown(f"_Piloto: **{d.rota.capitalize()}** · {d.motivo}_")
                outro = "flash" if d.rota == "lite" else "lite"
                try:
                    r = ia.conversar(mensagens, CHAVE, MODELOS[d.rota], producao.INSTRUCAO, reserva=MODELOS[outro],
                                     ao_receber=lambda t: saida.markdown(t + " ▍"),
                                     max_tokens=MAX_TOKENS_RESPOSTA)
                except Exception as e:
                    conversa.pop()
                    saida.empty()
                    if "429" in str(e):
                        st.warning("A cota da chave do Gemini acabou por agora. Tente de novo mais tarde.")
                    elif isinstance(e, ia.GeminiIndisponivel):
                        st.warning("O Gemini não respondeu a tempo (instabilidade do Google). Envie de novo em alguns segundos.")
                    else:
                        st.error(f"Erro ao chamar o Gemini: {str(e)[:200]}")
                else:
                    reg = producao.registro(sessao, pergunta, d, r, MODELOS, PRECOS,
                                            piloto.razao_flash_lite(hist), rot.versao, MAX_TOKENS_RESPOSTA)
                    id_ = None
                    if banco:
                        try:
                            id_ = banco.inserir(reg)
                            carregar_reais.clear()
                        except Exception as e:
                            st.caption(f"Resposta não registrada no banco: {str(e)[:120]}")
                    if id_ and not reg["forcado"] and producao.amostra_auditoria():
                        threading.Thread(target=producao.auditar, daemon=True,
                                         args=(banco, id_, mensagens, reg, MODELOS, PRECOS, CHAVE, MAX_TOKENS_RESPOSTA)).start()
                    conversa.append({"papel": "model", "texto": r.texto, "id": id_, "rota": reg["rota"],
                                     "motivo": reg["motivo"], "custo": reg["custo"],
                                     "custo_flash": reg["custo_flash_estimado"], "segundos": r.segundos,
                                     "cortada": r.cortada})
                    st.rerun()

# ---------- Resultados ----------
if secao == "Resultados":
    if real:
        eco_ic, qual_ic = producao.bootstrap_real(reais)
        secao_titulo("Em uso real", f"{br(real['pedidos'])} perguntas de usuários, cada uma roteada pelo piloto. "
                     f"O piloto pagou {usd(real['custo'])}; usando sempre o Flash, seriam {usd(real['custo_flash'])}.")
        faixa([
            ("Economia", pct(real["economia"]), f"IC 95%: {pct(eco_ic[0])} a {pct(eco_ic[1])}", True),
            ("Qualidade", pct(real["qualidade"]) if real["qualidade"] is not None else "–",
             (f"IC 95%: {pct(qual_ic[0])} a {pct(qual_ic[1])} · " if qual_ic else "")
             + f"{real['auditados']} respostas auditadas", False),
            ("Aprovação", pct(real["aprovacao"]) if real["aprovacao"] is not None else "–",
             f"👍 em {real['feedback']} avaliações de usuários", False),
            ("No Lite", pct(real["lite"]), "o resto foi para o Flash", False),
        ])
    elif tem_controle:
        secao_titulo("Resultados", "Ainda não há uso real suficiente; os números abaixo vêm do teste controlado diário.")
    else:
        secao_titulo("Resultados", "")
        st.html('<div class="vazio"><h3>Ainda não há resultados.</h3>Use o assistente ou aguarde a primeira rodada '
                'do teste controlado, que roda todo dia pelo GitHub Actions.</div>')

    # Projeção: custo por pedido do uso real quando há volume; senão, do teste controlado.
    if (real and real["pedidos"] >= 10) or tem_controle:
        if real and real["pedidos"] >= 10:
            c_pil, c_fla, fonte = reais.custo.values, piloto.custo_flash_real(reais), "do uso real"
        else:
            c_pil = np.where(hist.rota == "lite", hist.custo_lite, hist.custo_flash)
            c_fla, fonte = hist.custo_flash.values, "do teste controlado"
        volumes = {10_000: "10 mil", 100_000: "100 mil", 1_000_000: "1 milhão"}
        g1, g2 = st.columns([1, 1.3], gap="large")
        with g1:
            titulo_grafico("Quanto uma empresa economizaria", f"Monte Carlo com o custo de cada pedido {fonte}")
            volume = st.segmented_control("Pedidos por mês", list(volumes), format_func=lambda v: f"{volumes[v]}/mês",
                                          default=1_000_000, key="volume") or 1_000_000
            proj = piloto.projetar(np.asarray(c_pil, float), np.asarray(c_fla, float), volume)
            p5, p50, p95 = np.percentile(proj, [5, 50, 95])
            st.html(f'<div class="faixa"><div><span class="rotulo">Economia por mês</span><b class="acento">'
                    f'{usd(p50)}</b><small>90% das simulações entre {usd(p5)} e {usd(p95)}</small></div>'
                    f'<div><span class="rotulo">Por ano</span><b>{usd(12 * p50)}</b>'
                    f'<small>{volumes[volume]} pedidos por mês</small></div></div>')
        with g2:
            diario = ultimo_diario()
            if diario:
                titulo_grafico("Diário do piloto", f"escrito pela própria IA ao fim da rodada {diario[0]}")
                st.html(f'<div class="nota">{diario[1]}</div>')

    if tem_controle:
        st.html('<div class="secao" style="margin-top:2rem"><h2>Teste controlado</h2><p>Todo dia, pedidos reais do '
                'OpenAssistant recebem as duas respostas e passam pela juíza. Assim dá para comparar todas as '
                'estratégias nos mesmos pedidos.</p></div>')
        g1, g2 = st.columns([1.3, 1], gap="large")
        with g1:
            titulo_grafico("Custo × qualidade", "o melhor lugar é em cima e à esquerda")
            fig = go.Figure()
            for nome, r in aval.iterrows():
                fig.add_scatter(x=[r.custo], y=[100 * r.qualidade], mode="markers+text", name=nome, text=[nome],
                                textposition="top center",
                                marker=dict(size=16 if nome == "Piloto automático" else 11, color=CORES_ESTRATEGIA[nome]),
                                hovertemplate=f"{nome}<br>US$ %{{x:.4f}} · %{{y:.0f}}%<extra></extra>")
            fig.add_hline(y=100 * META_QUALIDADE, line=dict(color=MUDO, dash="dot", width=1),
                          annotation_text="meta de qualidade", annotation_position="bottom right")
            fig.update_layout(showlegend=False, xaxis_title="custo de gerar as respostas (US$)",
                              yaxis_title="qualidade (%)", yaxis_range=[min(50, 100 * aval.qualidade.min() - 5), 105])
            grafico(fig)
        with g2:
            eco_c = piloto.intervalo(hist, lambda s: piloto.avaliar(s).loc["Piloto automático", "economia"], n=300)
            regra = aval.loc["Regra fixa"]
            aud = reais[reais.auditado & reais.lite_ok.notna() & reais.feedback.notna() & (reais.rota == "lite")] \
                if not reais.empty else pd.DataFrame()
            concordancia = float(((aud.feedback > 0) == aud.lite_ok.astype(bool)).mean()) if len(aud) >= 10 else None

            def meta(texto, valor, ok):
                simbolo, classe = ("✓", "ok") if ok else ("✕", "fora") if ok is not None else ("…", "pend")
                return f'<div class="meta"><span class="s {classe}">{simbolo}</span><span>{texto}</span><b>{valor}</b></div>'
            titulo_grafico("Metas definidas antes da primeira rodada")
            st.html('<div class="metas">'
                    + meta(f"Economia ≥ {pct(META_ECONOMIA)} sobre sempre Flash",
                           f"{pct(pil.economia)} ({pct(eco_c[0])}–{pct(eco_c[1])})", pil.economia >= META_ECONOMIA)
                    + meta(f"Qualidade ≥ {pct(META_QUALIDADE)}", pct(pil.qualidade), pil.qualidade >= META_QUALIDADE)
                    + meta("Qualidade acima de usar sempre o Lite",
                           f"{pct(pil.qualidade)} × {pct(aval.loc['Sempre Lite'].qualidade)}",
                           pil.qualidade > aval.loc["Sempre Lite"].qualidade)
                    + meta("Aprender rende mais que a regra fixa",
                           f"{pct(pil.economia)} × {pct(regra.economia)}",
                           (pil.economia > regra.economia and pil.qualidade >= regra.qualidade) if rot.tipo == "aprendido" else None)
                    + meta(f"Juíza concorda com os usuários em ≥ {pct(META_CONCORDANCIA)}",
                           f"{pct(concordancia)} de {len(aud)}" if concordancia is not None else "aguardando avaliações",
                           concordancia >= META_CONCORDANCIA if concordancia is not None else None)
                    + '</div>')

# ---------- Bastidores ----------
if secao == "Bastidores":
    secao_titulo("Bastidores", "Como o piloto funciona e todos os dados por trás dos resultados.")

    with st.expander("Como o piloto funciona", expanded=True):
        st.html(f"""<div class="passos">
          <div class="passo"><div><span class="quemfaz ml">aprendizado de máquina</span><h4>Previsão do tamanho da resposta</h4>
            <p>Um Gradient Boosting treinado com 18.924 conversas reais estima quantos tokens a resposta vai ter.
            Respostas longas tendem a ser pedidos mais difíceis.</p></div>
            <div class="formula">entrada: tokens do pedido · tem código?
         nº de "?" · idioma
<i>saída:</i>  p10 · mediana · p90 de tokens</div></div>

          <div class="passo"><div><span class="quemfaz ml">aprendizado de máquina</span><h4>Decisão: Lite ou Flash</h4>
            <p>Antes de qualquer chamada à IA. No começo, uma regra; depois, uma regressão logística que aprende
            com os vereditos a chance de o Lite dar conta do pedido.</p></div>
            <div class="formula">Lite se P(Lite dá conta) ≥ corte
<i>corte:</i> o menor que mantém a qualidade ≥ {pct(META_QUALIDADE)}
<i>agora:</i> {"regra, p90 < " + br(rot.limiar_p90) + " tokens" if rot.tipo == "regra" else f"v{rot.versao}, corte {br(rot.corte, 2)}"}</div></div>

          <div class="passo"><div><span class="quemfaz gem">Gemini</span><h4>Só o escolhido responde</h4>
            <p>No assistente, só o modelo escolhido gera a resposta: a economia é real. Se o orçamento do dia acabar,
            tudo vai para o Lite; se um modelo estiver sobrecarregado, o outro responde.</p></div>
            <div class="formula">custo = ( entrada × preço de entrada
        + (resposta + raciocínio) × preço de saída ) ÷ 1.000.000</div></div>

          <div class="passo"><div><span class="quemfaz gem">Gemini · juíza</span><h4>Auditoria em amostra</h4>
            <p>Em 1 de cada 5 perguntas, depois que o usuário já recebeu a resposta, o outro modelo também responde e
            uma IA juíza compara as duas às cegas. A juíza é o <b>{MODELOS["juiza"] if online else "Gemini Pro"}</b>:
            maior que os dois modelos avaliados e diferente deles, para não julgar a si mesma. O usuário também
            avalia com 👍/👎.</p></div>
            <div class="formula">Lite deu conta = juíza disse "empate"
                  ou preferiu o Lite</div></div>

          <div class="passo"><div><span class="quemfaz">automação</span><h4>Teste controlado diário</h4>
            <p>Às 8h, o GitHub Actions roda pedidos do OpenAssistant nos dois modelos e na juíza: é a comparação
            justa entre estratégias.</p></div>
            <div class="formula">fila: {br(len(pd.read_csv(DADOS / "fila_pedidos.csv")))} pedidos que o previsor nunca viu</div></div>

          <div class="passo"><div><span class="quemfaz ml">aprendizado contínuo</span><h4>Retreino, diário e alerta</h4>
            <p>O roteador se retreina com os vereditos do teste e do uso real. A IA escreve o diário do dia; se a
            qualidade cair abaixo da meta, o workflow abre uma issue sozinho.</p></div>
            <div class="formula">roteador.json + diário → commit automático
<i>→</i> app atualizado</div></div>
        </div>""")
        titulo_grafico("Teste o roteador", "decide na hora, sem chamar o Gemini")
        texto = st.text_input("Pedido", "Explique juros compostos com um exemplo", label_visibility="collapsed")
        if texto.strip():
            x, prev = piloto.caracteristicas(texto, "pt-BR", modelo)
            rota, motivo = rot.decidir(x)
            st.html(f'<div class="etiquetas"><span>previsão <b>~{br(prev.mediana)}</b> tokens '
                    f'({br(prev.p10)}–{br(prev.p90)})</span><span>escolha <b class="{rota}">{rota.capitalize()}</b></span>'
                    f'<span>{motivo}</span></div>')

    with st.expander("Uso real: últimas perguntas"):
        if reais.empty:
            st.caption("Nenhuma pergunta registrada ainda.")
        else:
            u = reais.sort_values("criado_em", ascending=False).head(200)
            st.dataframe(pd.DataFrame({
                "Quando": u.criado_em.dt.tz_convert("America/Sao_Paulo").dt.strftime("%d/%m %H:%M"),
                "Pergunta": u.pedido.str.slice(0, 90),
                "Escolha": u.rota.str.capitalize(),
                "Motivo": u.motivo,
                "Custo": u.custo.round(5),
                "Auditada": np.where(u.auditado, u.vencedor.map({"lite": "Lite melhor", "flash": "Flash melhor",
                                                                 "empate": "empate"}), ""),
                "Usuário": u.feedback.map({1: "👍", -1: "👎"}).fillna(""),
            }), hide_index=True, width="stretch")

    with st.expander("Teste controlado: rodada a rodada"):
        if not tem_controle:
            st.caption("A primeira rodada ainda não aconteceu.")
        else:
            rodadas = hist.groupby("rodada").apply(
                lambda d: pd.concat([piloto.avaliar(d).loc["Piloto automático"],
                                     pd.Series({"pedidos": len(d), "versao": int(d.versao_roteador.iloc[0])})]),
                include_groups=False).reset_index()
            titulo_grafico("Economia e qualidade do piloto", "a linha pontilhada é a meta de qualidade")
            fig = go.Figure()
            fig.add_scatter(x=rodadas.rodada, y=100 * rodadas.qualidade, mode="lines+markers", name="qualidade",
                            line=dict(color=TINTA, width=2))
            fig.add_scatter(x=rodadas.rodada, y=100 * rodadas.economia, mode="lines+markers", name="economia",
                            line=dict(color=DESTAQUE, width=2))
            fig.add_hline(y=100 * META_QUALIDADE, line=dict(color=MUDO, dash="dot", width=1))
            fig.update_layout(yaxis_title="%", yaxis_range=[0, 105], xaxis_type="category")
            grafico(fig)
            st.dataframe(pd.DataFrame({
                "Rodada": rodadas.rodada, "Pedidos": rodadas.pedidos.astype(int),
                "Roteador": ["regra" if v == 0 else f"v{int(v)}" for v in rodadas.versao],
                "No Lite": (100 * rodadas.lite).round(0).astype(int).astype(str) + "%",
                "Economia": (100 * rodadas.economia).round(0).astype(int).astype(str) + "%",
                "Qualidade": (100 * rodadas.qualidade).round(0).astype(int).astype(str) + "%",
            }), hide_index=True, width="stretch")
            if roteador and roteador.versoes:
                titulo_grafico("Treinos do roteador", "validação cruzada: medida em pedidos que o modelo não viu no treino")
                v = pd.DataFrame(roteador.versoes)
                st.dataframe(pd.DataFrame({
                    "Versão": [f"v{i}" for i in range(1, len(v) + 1)], "Treinado em": v.data, "Rótulos": v.pedidos,
                    "AUC": v.auc.round(2), "Corte": v.corte.round(2),
                    "No Lite": (100 * v.lite_validacao).round(0).astype(int).astype(str) + "%",
                    "Qualidade": (100 * v.qualidade_validacao).round(0).astype(int).astype(str) + "%",
                }), hide_index=True, width="stretch")
            custo_teste = hist.custo_flash.sum() + hist.custo_lite.sum() + hist.custo_juiza.sum()
            st.html(f'<div class="nota">O teste controlado gera as duas respostas e chama a juíza em todos os pedidos: '
                    f'custou {usd(custo_teste)} até agora. É o preço de medir com rigor.</div>')

    with st.expander("Teste controlado: cada decisão"):
        if not tem_controle:
            st.caption("A primeira rodada ainda não aconteceu.")
        else:
            rodada = st.selectbox("Rodada", sorted(hist.rodada.unique(), reverse=True))
            d = hist[hist.rodada == rodada].reset_index(drop=True)
            acertou = (d.rota == "flash") | d.lite_ok.astype(bool)
            ev = st.dataframe(pd.DataFrame({
                "Pedido": d.pedido.str.slice(0, 90), "Escolha": d.rota.str.capitalize(),
                "Juíza": d.vencedor.map({"lite": "Lite melhor", "flash": "Flash melhor", "empate": "empate"}),
                "Resultado": np.where(acertou, "✓", "✕ Lite não deu conta"),
                "Custo": np.where(d.rota == "lite", d.custo_lite, d.custo_flash).round(5),
            }), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=300)
            p = d.iloc[ev.selection.rows[0] if ev.selection.rows else 0]
            st.html(f'<div class="pedido">{p.pedido}</div>'
                    f'<div class="etiquetas"><span>escolha <b class="{p.rota}">{p.rota.capitalize()}</b></span>'
                    f'<span>{p.motivo}</span><span>juíza: <b>{p.motivo_juiza}</b></span></div>')
            c1, c2 = st.columns(2, gap="large")
            for col, braco, nome in [(c1, "flash", p.modelo_flash), (c2, "lite", p.modelo_lite)]:
                with col:
                    titulo_grafico(f"{braco.capitalize()} · nota {int(p['nota_' + braco])}",
                                   f"{nome} · {br(p[braco + '_saida'] + p[braco + '_raciocinio'])} tokens de saída · "
                                   f"{usd(p['custo_' + braco])}")
                    with st.container(height=380, border=True):
                        st.markdown(p["resposta_" + braco])

    with st.expander("O previsor de tokens"):
        p = modelo.pares
        m = modelo.metricas.set_index("modelo")
        gb = m.loc["Gradient Boosting"]
        faixa([
            ("Conversas na base", br(len(p)), "pares pergunta → resposta", False),
            ("Erro médio", f"{br(gb.mae)} tokens", f"chute ingênuo: {br(m.loc['Baseline (mediana)'].mae)}", False),
            ("Acerto da faixa 80%", f"{br(100 * modelo.cobertura_faixa)}%", "no conjunto de teste", False),
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
            titulo_grafico("O idioma muda o custo", "tokens gastos a cada 1.000 caracteres")
            d = modelo.idiomas.sort_values("tokens_por_1000_caracteres")
            fig = go.Figure(go.Bar(
                x=d.tokens_por_1000_caracteres, y=[NOMES_IDIOMAS[l] for l in d.lang], orientation="h",
                marker_color=[DESTAQUE if l == "pt-BR" else CINZA for l in d.lang],
                hovertemplate="%{x:.0f} tokens<extra></extra>"))
            grafico(fig)
        st.html(f'<div class="nota">Sozinho, o previsor explica ~{br(100 * gb.r2)}% da variação do tamanho de uma resposta: '
                f'pouco para prever custo, mas útil como uma das pistas para decidir o modelo.</div>')

    with st.expander("Sobre, validade e limitações"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""
#### A família de algoritmos
- **Regressão** (Gradient Boosting e quantílica): prevê o tamanho da resposta.
- **Classificação** (regressão logística): prevê se o modelo barato dá conta do pedido.
- **LLM como juiz**: o Gemini avalia as respostas às cegas, conferido com o 👍/👎 dos usuários.
- **Aprendizado contínuo**: o roteador se retreina todo dia, com o teste e com o uso real.
- **Bootstrap e Monte Carlo**: intervalos de confiança e projeção do custo mensal.

#### Os dados
[OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache 2.0): treino do previsor
e pedidos do teste controlado. O uso real vem das perguntas feitas no assistente.
""")
        with c2:
            st.markdown("""
#### É válido?
- No teste controlado, as estratégias são comparadas nos mesmos pedidos.
- A juíza não sabe qual modelo gerou cada resposta, e a ordem é sorteada.
- O corte do roteador é escolhido com validação cruzada.
- No uso real, a qualidade é estimada por uma amostra aleatória auditada.

#### Limitações
- A juíza é do mesmo fornecedor dos modelos avaliados (mas é um modelo maior e diferente deles).
- Quando o Lite responde e não é auditado, o custo do Flash é estimado.
- Os preços por token são de tabela e mudam com o tempo.
""")
        st.caption("Atividade de IA aplicada e IA experimental, Prof. Wagner, 3º ciclo.")
