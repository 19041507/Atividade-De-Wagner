import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import piloto
from core.modelo import DADOS, NOMES_IDIOMAS, ModeloTokens

st.set_page_config(page_title="TokenScope", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

TINTA, MUDO, LINHA, PAPEL = "#161616", "#6B6A65", "#E3E1DA", "#F7F6F2"
DESTAQUE, BOM, RUIM, CINZA = "#C2410C", "#3F6B4E", "#A33A2B", "#B8B5AC"
EXECUCOES = DADOS / "execucoes"
META_ECONOMIA, META_QUALIDADE, META_CONCORDANCIA = 0.30, 0.90, 0.80
PRECO_IN, PRECO_OUT = piloto.PRECOS_PADRAO["flash"]
CORES_ESTRATEGIA = {"Sempre Flash": CINZA, "Sempre Lite": CINZA, "Piloto automático": DESTAQUE, "Escolha perfeita": TINTA}

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


@st.cache_resource(show_spinner="Carregando o previsor treinado com 18.924 conversas reais…")
def carregar_modelo():
    return ModeloTokens()


@st.cache_data(ttl=600, show_spinner=False)
def carregar_execucoes():
    return piloto.carregar_historico(EXECUCOES), piloto.Roteador.carregar(EXECUCOES / "roteador.json")


@st.cache_data(show_spinner=False)
def carregar_validacao():
    caminho = DADOS / "validacao_humana.csv"
    if not caminho.exists():
        return None
    v = pd.read_csv(caminho, dtype=str).fillna("")
    v = v[v.sua_escolha.str.strip().str.upper().isin(["A", "B", "EMPATE"])]
    return v if len(v) else None


@st.cache_resource(show_spinner="Calculando a regra inicial…")
def roteador_inicial(_modelo):
    return piloto.regra_inicial(_modelo, pd.read_csv(DADOS / "fila_pedidos.csv"))


modelo = carregar_modelo()
hist, roteador = carregar_execucoes()
tem_dados = not hist.empty
if tem_dados:
    aval = piloto.avaliar(hist)
    pil = aval.loc["Piloto automático"]
    rodadas = hist.groupby("data").apply(
        lambda d: pd.concat([piloto.avaliar(d).loc["Piloto automático"],
                             pd.Series({"pedidos": len(d), "versao": int(d.versao_roteador.iloc[0])})]),
        include_groups=False).reset_index()

concordancia = None
validacao = carregar_validacao()
if validacao is not None and tem_dados:
    juiza = hist.set_index("id").vencedor
    escolha = validacao.sua_escolha.str.strip().str.upper()
    humano = np.where(escolha == "EMPATE", "empate",
                      np.where((escolha == "A") == (validacao.gabarito_a == "lite"), "lite", "flash"))
    # o que importa para o roteador: o Lite deu conta (empate ou melhor) ou não
    acordo = (humano != "flash") == (juiza.reindex(validacao.id).values != "flash")
    concordancia = (float(np.mean(acordo)), len(validacao))

ultima = f'última rodada {pd.to_datetime(hist.data.max()):%d/%m} · roteador v{roteador.versao}' if tem_dados else "aguardando a primeira rodada"
st.html(f"""
<div class="topo">
  <div class="marca">TokenScope<small>piloto automático de custo de IA</small></div>
  <div class="status"><i style="background:{BOM if tem_dados else CINZA}"></i>{ultima}</div>
</div>
<div class="abre">
  <div>
    <h1>A IA escolhe o modelo. <em>Você paga menos.</em></h1>
    <p>Todo dia, sem ninguém clicar, o piloto recebe pedidos reais e decide, para cada um, entre o Gemini Flash-Lite
    (barato) e o Flash (mais capaz). Uma segunda IA confere se a qualidade se manteve, e o piloto aprende com as notas.</p>
  </div>
  <div class="numeros">
    <div><span class="rotulo">Pedidos atendidos</span><b>{br(len(hist)) if tem_dados else "–"}</b></div>
    <div><span class="rotulo">Economia</span><b>{pct(pil.economia) if tem_dados else "–"}</b></div>
    <div><span class="rotulo">Qualidade</span><b>{pct(pil.qualidade) if tem_dados else "–"}</b></div>
  </div>
</div>
""")

SECOES = {"Resultado": "Resultado", "Rodadas": "Rodadas", "Decisoes": "Decisões", "Como": "Como funciona",
          "Previsor": "O previsor de tokens", "Sobre": "Sobre"}
secao = st.segmented_control("Seção", list(SECOES), format_func=SECOES.get, default="Resultado",
                             key="secao", label_visibility="collapsed") or "Resultado"


def sem_dados():
    st.html('<div class="vazio"><h3>A primeira rodada ainda não aconteceu.</h3>'
            'O piloto roda uma vez por dia pelo GitHub Actions. Assim que terminar, os resultados aparecem aqui.</div>')


# ---------- Resultado ----------
if secao == "Resultado":
    secao_titulo("O resultado", "Os mesmos pedidos, quatro jeitos de escolher o modelo. Para medir cada um, o sistema gera "
                 "as duas respostas em todos os pedidos e a IA juíza compara: assim sabemos o que teria acontecido "
                 "com qualquer escolha.")
    if not tem_dados:
        sem_dados()
    else:
        flash = aval.loc["Sempre Flash"]
        faixa([
            ("Economia", pct(pil.economia), "sobre usar sempre o Flash", True),
            ("Qualidade", pct(pil.qualidade), "pedidos tão bem atendidos quanto pelo Flash", False),
            ("Pedidos no Lite", pct(pil.lite), "o resto foi para o Flash", False),
            ("Custo", usd(pil.custo), f"sempre Flash: {usd(flash.custo)}", False),
        ])
        g1, g2 = st.columns([1.3, 1], gap="large")
        with g1:
            titulo_grafico("Custo × qualidade", "o melhor lugar é em cima e à esquerda")
            fig = go.Figure()
            for nome, r in aval.iterrows():
                fig.add_scatter(x=[r.custo], y=[100 * r.qualidade], mode="markers+text", name=nome, text=[nome],
                                textposition="top center", marker=dict(size=16 if nome == "Piloto automático" else 11,
                                                                       color=CORES_ESTRATEGIA[nome]),
                                hovertemplate=f"{nome}<br>US$ %{{x:.4f}} · %{{y:.0f}}%<extra></extra>")
            fig.add_hline(y=100 * META_QUALIDADE, line=dict(color=MUDO, dash="dot", width=1),
                          annotation_text="meta de qualidade", annotation_position="bottom right")
            fig.update_layout(showlegend=False, xaxis_title="custo de gerar as respostas (US$)",
                              yaxis_title="qualidade (%)", yaxis_range=[min(50, 100 * aval.qualidade.min() - 5), 105])
            grafico(fig)
        with g2:
            titulo_grafico("Economia sobre usar sempre o Flash", "quanto maior, melhor")
            fig = go.Figure(go.Bar(x=100 * aval.economia, y=aval.index, orientation="h",
                                   marker_color=[CORES_ESTRATEGIA[n] for n in aval.index],
                                   text=[pct(v) for v in aval.economia], textposition="outside",
                                   hovertemplate="%{x:.1f}%<extra></extra>"))
            fig.update_layout(xaxis=dict(visible=False, range=[0, 115]), yaxis=dict(autorange="reversed"))
            grafico(fig, 300)

        def meta(texto, valor, ok):
            simbolo, classe = ("✓", "ok") if ok else ("✕", "fora") if ok is not None else ("…", "pend")
            return f'<div class="meta"><span class="s {classe}">{simbolo}</span><span>{texto}</span><b>{valor}</b></div>'
        titulo_grafico("Metas definidas antes da primeira rodada")
        st.html('<div class="metas">'
                + meta(f"Economia de pelo menos {pct(META_ECONOMIA)} sobre sempre Flash", pct(pil.economia),
                       pil.economia >= META_ECONOMIA)
                + meta(f"Qualidade em pelo menos {pct(META_QUALIDADE)} dos pedidos", pct(pil.qualidade),
                       pil.qualidade >= META_QUALIDADE)
                + meta("Qualidade melhor que usar sempre o Lite",
                       f"{pct(pil.qualidade)} × {pct(aval.loc['Sempre Lite'].qualidade)}",
                       pil.qualidade > aval.loc["Sempre Lite"].qualidade)
                + meta(f"Juíza concorda com uma pessoa em pelo menos {pct(META_CONCORDANCIA)} dos casos",
                       f"{pct(concordancia[0])} de {concordancia[1]}" if concordancia else "aguardando avaliação",
                       concordancia[0] >= META_CONCORDANCIA if concordancia else None)
                + '</div>')

        perfeita = aval.loc["Escolha perfeita"]
        st.html(f'<div class="nota"><b>Como ler.</b> Usar sempre o Lite é o mais barato, mas falha em '
                f'{pct(1 - aval.loc["Sempre Lite"].qualidade)} dos pedidos. Usar sempre o Flash é o mais seguro e o mais caro. '
                f'O piloto fica no meio: economiza {pct(pil.economia)} e mantém {pct(pil.qualidade)} de qualidade. '
                f'A escolha perfeita, que só daria para fazer conhecendo as respostas, economizaria '
                f'{pct(perfeita.economia)}: é o espaço que o roteador ainda pode ganhar aprendendo.</div>')

# ---------- Rodadas ----------
if secao == "Rodadas":
    secao_titulo("Rodada a rodada", "Cada rodada usa o roteador treinado com todas as notas anteriores. "
                 "A primeira usa uma regra simples; a partir de 40 vereditos, ele passa a aprender.")
    if not tem_dados:
        sem_dados()
    else:
        datas = pd.to_datetime(rodadas.data).dt.strftime("%d/%m")
        titulo_grafico("Economia e qualidade do piloto", "a linha pontilhada é a meta de qualidade")
        fig = go.Figure()
        fig.add_scatter(x=datas, y=100 * rodadas.qualidade, mode="lines+markers", name="qualidade",
                        line=dict(color=TINTA, width=2))
        fig.add_scatter(x=datas, y=100 * rodadas.economia, mode="lines+markers", name="economia",
                        line=dict(color=DESTAQUE, width=2))
        fig.add_hline(y=100 * META_QUALIDADE, line=dict(color=MUDO, dash="dot", width=1))
        fig.update_layout(yaxis_title="%", yaxis_range=[0, 105], xaxis_type="category")
        grafico(fig)

        tabela = pd.DataFrame({
            "Rodada": datas, "Pedidos": rodadas.pedidos.astype(int),
            "Roteador": ["regra" if v == 0 else f"v{int(v)}" for v in rodadas.versao],
            "No Lite": (100 * rodadas.lite).round(0).astype(int).astype(str) + "%",
            "Economia": (100 * rodadas.economia).round(0).astype(int).astype(str) + "%",
            "Qualidade": (100 * rodadas.qualidade).round(0).astype(int).astype(str) + "%",
        })
        st.dataframe(tabela, hide_index=True, width="stretch")

        if roteador and roteador.versoes:
            titulo_grafico("Treinos do roteador", "validação cruzada: medida em pedidos que o modelo não viu no treino")
            v = pd.DataFrame(roteador.versoes)
            st.dataframe(pd.DataFrame({
                "Versão": [f"v{i}" for i in range(1, len(v) + 1)],
                "Treinado em": pd.to_datetime(v.data).dt.strftime("%d/%m"),
                "Vereditos": v.pedidos,
                "AUC": v.auc.round(2),
                "Corte": v.corte.round(2),
                "No Lite (validação)": (100 * v.lite_validacao).round(0).astype(int).astype(str) + "%",
                "Qualidade (validação)": (100 * v.qualidade_validacao).round(0).astype(int).astype(str) + "%",
            }), hide_index=True, width="stretch")

        avaliacao = hist.custo_flash.sum() + hist.custo_lite.sum() + hist.custo_juiza.sum()
        st.html(f'<div class="nota"><b>Quanto custa provar.</b> Para medir as quatro estratégias, cada pedido gera as duas '
                f'respostas e passa pela juíza: até agora {usd(avaliacao)}. Em produção, o piloto chamaria só o modelo '
                f'escolhido e gastaria {usd(pil.custo)}.</div>')

# ---------- Decisões ----------
if secao == "Decisoes":
    secao_titulo("Cada decisão", "O que o piloto decidiu, por quê, e o que a juíza achou. Clique em um pedido para ver "
                 "as duas respostas lado a lado.")
    if not tem_dados:
        sem_dados()
    else:
        datas = sorted(hist.data.unique(), reverse=True)
        dia = st.selectbox("Rodada", datas, format_func=lambda d: f"{pd.to_datetime(d):%d/%m/%Y}")
        d = hist[hist.data == dia].reset_index(drop=True)
        acertou = (d.rota == "flash") | d.lite_ok.astype(bool)
        lista = pd.DataFrame({
            "Pedido": d.pedido.str.slice(0, 90),
            "Escolha": d.rota.str.capitalize(),
            "Juíza": d.vencedor.map({"lite": "Lite melhor", "flash": "Flash melhor", "empate": "empate"}),
            "Resultado": np.where(acertou, "✓", "✕ Lite não deu conta"),
            "Custo": np.where(d.rota == "lite", d.custo_lite, d.custo_flash).round(5),
        })
        ev = st.dataframe(lista, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          height=320)
        linha = ev.selection.rows[0] if ev.selection.rows else 0
        p = d.iloc[linha]
        st.html(f'<div class="pedido">{p.pedido}</div>'
                f'<div class="etiquetas"><span>escolha <b class="{p.rota}">{p.rota.capitalize()}</b></span>'
                f'<span>{p.motivo}</span><span>juíza: <b>{p.motivo_juiza}</b></span></div>')
        c1, c2 = st.columns(2, gap="large")
        for col, braco, nome in [(c1, "flash", p.modelo_flash), (c2, "lite", p.modelo_lite)]:
            with col:
                titulo_grafico(f"{braco.capitalize()} · nota {int(p['nota_' + braco])}",
                               f"{nome} · {br(p[braco + '_saida'] + p[braco + '_raciocinio'])} tokens de saída · "
                               f"{usd(p['custo_' + braco])}")
                with st.container(height=420, border=True):
                    st.markdown(p["resposta_" + braco])

# ---------- Como funciona ----------
if secao == "Como":
    secao_titulo("Como o piloto funciona", "Três IAs trabalham juntas: o previsor estima o tamanho da resposta, o "
                 "roteador decide o modelo e a juíza confere a qualidade. As notas da juíza ensinam o roteador.")
    rot = roteador or roteador_inicial(modelo)
    st.html(f"""<div class="passos">
      <div class="passo"><div><span class="quemfaz">automação</span><h4>Todo dia, sozinho</h4>
        <p>O GitHub Actions dispara a rodada às 8h. O piloto sorteia pedidos reais do OpenAssistant que ainda não
        atendeu, todos do conjunto de teste: o previsor nunca viu nenhum deles.</p></div>
        <div class="formula">fila: {br(len(pd.read_csv(DADOS / "fila_pedidos.csv")))} pedidos reais
<i>por rodada:</i> 50 pedidos</div></div>

      <div class="passo"><div><span class="quemfaz ml">aprendizado de máquina</span><h4>Previsão do tamanho da resposta</h4>
        <p>O Gradient Boosting treinado com 18.924 conversas estima quantos tokens a resposta vai ter (mediana e
        faixa de 80%). Respostas longas tendem a ser pedidos mais difíceis.</p></div>
        <div class="formula">entrada: tokens do pedido · tem código?
         nº de "?" · idioma
<i>saída:</i>  p10 · mediana · p90 de tokens</div></div>

      <div class="passo"><div><span class="quemfaz ml">aprendizado de máquina</span><h4>Decisão: Lite ou Flash</h4>
        <p>Na primeira rodada, uma regra: pedido sem código e com resposta prevista curta vai para o Lite. Depois, uma
        regressão logística aprende com as notas da juíza a chance de o Lite dar conta de cada pedido.</p></div>
        <div class="formula">Lite se P(Lite dá conta) ≥ corte
<i>corte:</i> o menor que mantém a qualidade ≥ {pct(META_QUALIDADE)}
<i>agora:</i> {"regra, p90 < " + br(rot.limiar_p90) + " tokens" if rot.tipo == "regra" else f"v{rot.versao}, corte {br(rot.corte, 2)}"}</div></div>

      <div class="passo"><div><span class="quemfaz gem">Gemini</span><h4>As duas respostas</h4>
        <p>Para provar o resultado, o Flash e o Lite respondem a todos os pedidos. Em produção, só o escolhido
        responderia. A própria API informa os tokens de cada chamada.</p></div>
        <div class="formula">custo = ( entrada × preço de entrada
        + (resposta + raciocínio) × preço de saída ) ÷ 1.000.000</div></div>

      <div class="passo"><div><span class="quemfaz gem">Gemini · juíza</span><h4>A juíza compara</h4>
        <p>Outra chamada ao Gemini recebe o pedido e as duas respostas como A e B, em ordem sorteada, sem saber qual
        modelo gerou cada uma, e devolve notas de 1 a 5 e o veredito em JSON.</p></div>
        <div class="formula">Lite deu conta = juíza disse "empate"
                  ou preferiu o Lite</div></div>

      <div class="passo"><div><span class="quemfaz ml">aprendizado contínuo</span><h4>O roteador se retreina</h4>
        <p>Os vereditos de todas as rodadas viram dados de treino. O corte é escolhido com validação cruzada, em
        pedidos que o modelo não viu. O resultado vai para o repositório e o app se atualiza sozinho.</p></div>
        <div class="formula">roteador.json + execucoes/AAAA-MM-DD.csv
<i>→</i> commit automático → app atualizado</div></div>
    </div>""")

    titulo_grafico("Teste um pedido", "o roteador atual decide na hora, sem chamar o Gemini")
    texto = st.text_input("Pedido", "Explique juros compostos com um exemplo", label_visibility="collapsed")
    if texto.strip():
        x, prev = piloto.caracteristicas(texto, "pt-BR", modelo)
        rota, motivo = rot.decidir(x)
        st.html(f'<div class="etiquetas"><span>previsão <b>~{br(prev.mediana)}</b> tokens '
                f'({br(prev.p10)}–{br(prev.p90)})</span><span>escolha <b class="{rota}">{rota.capitalize()}</b></span>'
                f'<span>{motivo}</span></div>')

# ---------- O previsor ----------
if secao == "Previsor":
    p = modelo.pares
    m = modelo.metricas.set_index("modelo")
    gb = m.loc["Gradient Boosting"]
    real, previsto = modelo.custo_mensal(10_000, PRECO_IN, PRECO_OUT)
    erro_mes = 100 * np.median(np.abs(previsto - real) / real)
    secao_titulo("O previsor de tokens", "A primeira peça do piloto. Base pública OpenAssistant (oasst1), com 18.924 "
                 "pares de pergunta e resposta reais. Aprendizado supervisionado (regressão), validado em 25% dos "
                 "dados que o modelo não viu durante o treino.")
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
    <div class="nota"><b>Por que o previsor sozinho não bastava.</b> Prever o tamanho de <i>uma</i> resposta é difícil:
    o modelo explica ~{br(100 * gb.r2)}% da variação e bate o chute ingênuo por pouco. No total de milhares de respostas
    os erros se compensam (~{br(erro_mes)}% no custo do mês), mas para isso uma média histórica já serve. O ganho real
    veio de usar a previsão como uma das pistas para <b>decidir</b> o modelo de cada pedido.</div>""")

# ---------- Sobre ----------
if secao == "Sobre":
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
#### O problema
Toda aplicação com IA generativa paga por **token**, e o modelo mais capaz custa várias vezes mais que o mais
simples. Mandar tudo para o modelo caro desperdiça dinheiro; mandar tudo para o barato derruba a qualidade.

#### A solução
Um **piloto automático** que decide, pedido a pedido, qual modelo usar, mede o resultado com uma IA juíza e
**aprende sozinho** com isso. Roda todo dia, sem ninguém clicar.

#### A família de algoritmos
- **Regressão** (Gradient Boosting e regressão quantílica): prevê o tamanho da resposta.
- **Classificação** (regressão logística): prevê se o modelo barato dá conta do pedido.
- **LLM como juiz**: o Gemini avalia as respostas, validado contra avaliação humana.
- **Aprendizado contínuo**: o roteador é retreinado a cada rodada.
- **Monte Carlo**: projeção do custo de milhares de respostas.
""")
    with c2:
        st.markdown("""
#### Os dados
[OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1): ~88 mil mensagens reais,
18.924 pares pergunta → melhor resposta. Licença Apache 2.0. Os pedidos do piloto saem do conjunto de teste.

#### É válido?
- As duas respostas são geradas para todos os pedidos: a comparação entre estratégias é nos mesmos pedidos.
- A juíza não sabe qual modelo gerou cada resposta, e a ordem é sorteada.
- O corte do roteador é escolhido com validação cruzada.
- Uma amostra é avaliada por uma pessoa para medir se a juíza é confiável.

#### Limitações
- A juíza é do mesmo fornecedor dos modelos avaliados.
- Os preços por token são de tabela e mudam com o tempo.
- Os pedidos do OpenAssistant são genéricos; uma aplicação real teria outro perfil.
""")
    st.caption("Atividade de IA aplicada e IA experimental, Prof. Wagner, 3º ciclo.")
