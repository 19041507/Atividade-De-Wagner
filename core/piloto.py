"""Piloto automático: para cada pedido, decide sozinho entre o modelo barato (Lite) e o capaz (Flash).

A decisão sai antes de qualquer chamada à IA, só com o que a aplicação já sabe do pedido. Depois de cada
rodada, as notas da IA juíza viram rótulos e o roteador é retreinado: ele aprende em quais pedidos o Lite dá conta.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from core.modelo import IDIOMAS, ModeloTokens, features_do_texto

COLUNAS = (["log_prompt_tokens", "tem_codigo", "perguntas", "log_p10", "log_mediana", "log_p90"]
           + ["idioma_" + l for l in IDIOMAS])
PRECOS_PADRAO = {"flash": (0.30, 2.50), "lite": (0.10, 0.40)}  # US$ por 1M tokens (entrada, saída)


def caracteristicas(texto: str, lang: str, modelo: ModeloTokens):
    """O que o roteador vê de um pedido: as features do previsor e a previsão de tokens da resposta."""
    x = features_do_texto(texto, lang, continuacao=False).iloc[0].to_dict()
    prev = modelo.prever(texto, lang)
    x.update(log_p10=np.log1p(prev.p10), log_mediana=np.log1p(prev.mediana), log_p90=np.log1p(prev.p90))
    return {c: float(x[c]) for c in COLUNAS}, prev


def custo(precos: tuple[float, float], entrada: float, saida: float) -> float:
    return (entrada * precos[0] + saida * precos[1]) / 1e6


@dataclass
class Roteador:
    tipo: str = "regra"          # "regra" na partida a frio; "aprendido" depois que há rótulos da juíza
    limiar_p90: float = 0.0      # regra: Lite quando a previsão p90 fica abaixo disso e o pedido não tem código
    media: list = field(default_factory=list)
    desvio: list = field(default_factory=list)
    pesos: list = field(default_factory=list)
    intercepto: float = 0.0
    corte: float = 0.5           # aprendido: Lite quando P(Lite dá conta) >= corte
    versoes: list = field(default_factory=list)  # um registro por treino, para a curva de aprendizado

    @property
    def versao(self) -> int:
        return len(self.versoes)

    def prob_lite_ok(self, x: dict) -> float | None:
        if self.tipo != "aprendido":
            return None
        z = (np.array([x[c] for c in COLUNAS]) - self.media) / self.desvio
        return float(1 / (1 + np.exp(-(z @ np.array(self.pesos) + self.intercepto))))

    def decidir(self, x: dict) -> tuple[str, str]:
        if self.tipo == "aprendido":
            p = self.prob_lite_ok(x)
            if p >= self.corte:
                return "lite", f"chance de o Lite dar conta: {p:.0%} (corte {self.corte:.0%})"
            return "flash", f"chance de o Lite dar conta: {p:.0%}, abaixo do corte de {self.corte:.0%}"
        p90 = np.expm1(x["log_p90"])
        if x["tem_codigo"]:
            return "flash", "pedido com código"
        if p90 < self.limiar_p90:
            return "lite", f"resposta curta prevista (p90 {p90:.0f} < {self.limiar_p90:.0f} tokens)"
        return "flash", f"resposta longa prevista (p90 {p90:.0f} ≥ {self.limiar_p90:.0f} tokens)"

    def salvar(self, caminho: Path):
        caminho.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))

    @classmethod
    def carregar(cls, caminho: Path) -> "Roteador | None":
        return cls(**json.loads(caminho.read_text())) if caminho.exists() else None


def regra_inicial(modelo: ModeloTokens, fila: pd.DataFrame) -> Roteador:
    """Partida a frio: o limiar é a mediana da previsão p90 na fila, então metade dos pedidos começa no Lite."""
    p90 = [modelo.prever(t, l).p90 for t, l in zip(fila.texto, fila.lang)]
    return Roteador(limiar_p90=float(np.median(p90)))


def qualidade(rota_lite: np.ndarray, lite_ok: np.ndarray) -> float:
    """Fração de pedidos bem atendidos: o Flash é a referência; o Lite falha quando a juíza preferiu o Flash."""
    return float(1 - np.mean(rota_lite & ~lite_ok))


def treinar(historico: pd.DataFrame, meta_qualidade: float, versao_anterior: Roteador) -> Roteador | None:
    """Regressão logística para P(Lite dá conta), com o corte escolhido para bater a meta de qualidade."""
    y = historico.lite_ok.astype(bool).values
    if len(y) < 40 or min(y.sum(), (~y).sum()) < 5:
        return None
    X = historico[["x_" + c for c in COLUNAS]].values
    media, desvio = X.mean(0), X.std(0)
    desvio[desvio == 0] = 1
    Z = (X - media) / desvio
    reg = LogisticRegression(max_iter=1000)
    # Probabilidades de validação cruzada: o corte é escolhido em pedidos que cada modelo não viu no treino.
    prob = cross_val_predict(reg, Z, y, cv=5, method="predict_proba")[:, 1]
    corte = 1.01  # se nenhum corte bate a meta, tudo vai para o Flash
    for c in np.sort(np.unique(prob)):  # do mais permissivo (mais Lite, mais barato) para o mais rígido
        if qualidade(prob >= c, y) >= meta_qualidade:
            corte = float(c)
            break
    reg.fit(Z, y)
    return Roteador(
        tipo="aprendido", limiar_p90=versao_anterior.limiar_p90,
        media=media.tolist(), desvio=desvio.tolist(), pesos=reg.coef_[0].tolist(),
        intercepto=float(reg.intercept_[0]), corte=corte,
        versoes=versao_anterior.versoes + [{
            "data": str(historico.data.max()), "pedidos": int(len(y)), "auc": float(roc_auc_score(y, prob)),
            "corte": corte, "lite_validacao": float(np.mean(prob >= corte)),
            "qualidade_validacao": qualidade(prob >= corte, y),
        }],
    )


def carregar_historico(pasta: Path) -> pd.DataFrame:
    """Rodadas do teste controlado: um CSV por rodada (AAAA-MM-DD.csv, ou AAAA-MM-DD-HHMM.csv se houve mais de uma)."""
    arquivos = sorted(pasta.glob("????-??-??*.csv"))
    if not arquivos:
        return pd.DataFrame()
    h = pd.concat([pd.read_csv(a).assign(rodada=a.stem) for a in arquivos], ignore_index=True)
    h["data"] = h.data.astype(str)
    return h


def usa_regra(h: pd.DataFrame, limiar_p90: float) -> np.ndarray:
    """A regra da partida a frio aplicada a pedidos já avaliados (para comparar com o roteador aprendido)."""
    return ((h.x_tem_codigo == 0) & (np.expm1(h.x_log_p90) < limiar_p90)).values


def avaliar(h: pd.DataFrame, limiar_p90: float | None = None) -> pd.DataFrame:
    """Custo e qualidade das estratégias nos mesmos pedidos (as duas respostas foram geradas para todos)."""
    lite_ok = h.lite_ok.astype(bool).values
    rota_lite = (h.rota == "lite").values
    estrategias = {
        "Sempre Flash": np.zeros(len(h), bool),
        "Sempre Lite": np.ones(len(h), bool),
        **({"Regra fixa": usa_regra(h, limiar_p90)} if limiar_p90 else {}),
        "Piloto automático": rota_lite,
        "Escolha perfeita": lite_ok,
    }
    linhas = []
    for nome, usa_lite in estrategias.items():
        linhas.append({
            "estrategia": nome,
            "custo": float(np.where(usa_lite, h.custo_lite, h.custo_flash).sum()),
            "qualidade": qualidade(usa_lite, lite_ok),
            "lite": float(usa_lite.mean()),
        })
    r = pd.DataFrame(linhas).set_index("estrategia")
    r["economia"] = 1 - r.custo / r.loc["Sempre Flash", "custo"]
    return r


def intervalo(valores: pd.DataFrame, estatistica, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Intervalo de confiança de 95% por bootstrap: reamostra os pedidos e recalcula a estatística."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(valores), size=(n, len(valores)))
    amostras = [estatistica(valores.iloc[i]) for i in idx]
    return float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))


def projetar(custo_piloto: np.ndarray, custo_flash: np.ndarray, volume: int, simulacoes: int = 2000, seed: int = 0):
    """Monte Carlo: economia mensal (US$) de uma aplicação com `volume` pedidos por mês, sorteando pedidos reais."""
    rng = np.random.default_rng(seed)
    n = min(volume, 5000)
    idx = rng.integers(0, len(custo_piloto), size=(simulacoes, n))
    return (custo_flash[idx].sum(1) - custo_piloto[idx].sum(1)) * volume / n


# ---------- uso real (tabela pedidos no Supabase) ----------

def razao_flash_lite(h: pd.DataFrame) -> float:
    """Quantas vezes a resposta do Flash é maior que a do Lite, medido no teste controlado.

    Serve para estimar quanto o Flash custaria num pedido real que foi para o Lite e não foi auditado.
    """
    if h.empty:
        return 1.0
    f = h.flash_saida + h.flash_raciocinio
    l = (h.lite_saida + h.lite_raciocinio).clip(lower=1)
    return float(np.median(f / l))


def custo_flash_real(r: pd.DataFrame) -> np.ndarray:
    """Quanto cada pedido real custaria no Flash: o custo pago, o da auditoria ou a estimativa."""
    auditado_lite = r.auditado & (r.rota == "lite") & r.custo_alt.notna()
    return np.where(r.rota == "flash", r.custo, np.where(auditado_lite, r.custo_alt, r.custo_flash_estimado))


def rotulos_reais(r: pd.DataFrame) -> pd.DataFrame:
    """Pedidos reais que viram dado de treino: os auditados (rótulo da juíza) e os do Lite com 👍/👎 do usuário."""
    if r.empty:
        return pd.DataFrame()
    auditados = r[r.auditado & r.lite_ok.notna() & ~r.forcado]
    com_feedback = r[~r.auditado & (r.rota == "lite") & r.feedback.notna() & ~r.forcado]
    base = pd.concat([auditados.assign(lite_ok=auditados.lite_ok.astype(bool)),
                      com_feedback.assign(lite_ok=com_feedback.feedback > 0)])
    if base.empty:
        return pd.DataFrame()
    x = pd.DataFrame(list(base.x)).add_prefix("x_")
    return pd.concat([x, pd.DataFrame({"lite_ok": base.lite_ok.values,
                                       "data": base.criado_em.dt.strftime("%Y-%m-%d").values})], axis=1)
