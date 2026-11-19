"""Modelo de previsão de tokens da resposta, treinado nos pares prompt → resposta do oasst1."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DADOS = Path(__file__).resolve().parent.parent / "data"
IDIOMAS = ["en", "es", "pt-BR", "fr", "de", "ru", "zh"]
NOMES_IDIOMAS = {
    "pt-BR": "Português", "en": "Inglês", "es": "Espanhol", "fr": "Francês",
    "de": "Alemão", "ru": "Russo", "zh": "Chinês",
}
PADRAO_CODIGO = r"```|def |function|código|code"

_enc = tiktoken.get_encoding("o200k_base")


def contar_tokens(texto: str) -> int:
    return len(_enc.encode(texto, disallowed_special=()))


def _features(prompt_tokens, tem_codigo, perguntas, continuacao, lang) -> pd.DataFrame:
    X = pd.DataFrame({
        "log_prompt_tokens": np.log1p(prompt_tokens),
        "tem_codigo": tem_codigo,
        "perguntas": perguntas,
        "continuacao": continuacao,
    })
    lang = pd.Series(lang)
    for l in IDIOMAS:
        X["idioma_" + l] = (lang == l).astype(int).values
    return X


def features_do_texto(texto: str, lang: str, continuacao: bool) -> pd.DataFrame:
    import re
    return _features(
        [contar_tokens(texto)],
        [int(bool(re.search(PADRAO_CODIGO, texto, re.IGNORECASE)))],
        [texto.count("?")],
        [int(continuacao)],
        [lang],
    )


@dataclass
class Previsao:
    prompt_tokens: int
    mediana: float
    p10: float
    p90: float
    media: float  # usada para custo (média esperada, com correção de smearing)


class ModeloTokens:
    def __init__(self):
        self.pares = pd.read_csv(DADOS / "pares_prompt_resposta.csv")
        self.idiomas = pd.read_csv(DADOS / "tokens_por_idioma.csv")
        p = self.pares
        X = _features(p.prompt_tokens, p.tem_codigo, p.perguntas, p.continuacao, p.lang)
        y = np.log1p(p.resp_tokens)
        X_tr, X_te, y_tr, y_te, r_tr, r_te = train_test_split(
            X, y, p.resp_tokens, test_size=0.25, random_state=42)

        self.gb = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
        self.q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=42).fit(X_tr, y_tr)
        self.q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=42).fit(X_tr, y_tr)
        self.smearing = float(np.mean(np.exp(y_tr - self.gb.predict(X_tr))))

        lin = LinearRegression().fit(X_tr, y_tr)
        base = np.full(len(y_te), np.median(y_tr))
        p_gb, p_lin = self.gb.predict(X_te), lin.predict(X_te)
        self.metricas = pd.DataFrame([
            {"modelo": "Baseline (mediana)", "r2": r2_score(y_te, base),
             "mae": mean_absolute_error(r_te, np.expm1(base))},
            {"modelo": "Regressão linear", "r2": r2_score(y_te, p_lin),
             "mae": mean_absolute_error(r_te, np.expm1(p_lin))},
            {"modelo": "Gradient Boosting", "r2": r2_score(y_te, p_gb),
             "mae": mean_absolute_error(r_te, np.expm1(p_gb))},
        ])
        lo, hi = np.expm1(self.q10.predict(X_te)), np.expm1(self.q90.predict(X_te))
        self.cobertura_faixa = float(np.mean((r_te.values >= lo) & (r_te.values <= hi)))
        self.teste = pd.DataFrame({
            "real": r_te.values,
            "previsto": np.expm1(p_gb),
            "previsto_media": np.expm1(p_gb) * self.smearing,
            "prompt_tokens": p.prompt_tokens.loc[X_te.index].values,
        })

    def prever(self, texto: str, lang: str, continuacao: bool = False) -> Previsao:
        X = features_do_texto(texto, lang, continuacao)
        m = float(np.expm1(self.gb.predict(X))[0])
        return Previsao(
            prompt_tokens=int(np.expm1(X.log_prompt_tokens.iloc[0]).round()),
            mediana=m,
            p10=float(np.expm1(self.q10.predict(X))[0]),
            p90=float(np.expm1(self.q90.predict(X))[0]),
            media=m * self.smearing,
        )

    def economia_com_limite(self, limite: int) -> tuple[float, float]:
        """Fração de tokens de saída economizada e fração de respostas cortadas, com base nas respostas reais."""
        r = self.pares.resp_tokens
        return float(1 - np.minimum(r, limite).sum() / r.sum()), float((r > limite).mean())

    def custo_mensal(self, req_mes: int, preco_in: float, preco_out: float,
                     simulacoes: int = 2000, seed: int = 42):
        """Monte Carlo (bootstrap) sobre o conjunto de teste: custo real × custo previsto pelo modelo."""
        rng = np.random.default_rng(seed)
        t = self.teste
        n = min(req_mes, 5000)
        idx = rng.integers(0, len(t), size=(simulacoes, n))
        escala = req_mes / n
        ent = t.prompt_tokens.values[idx].sum(1) * escala
        real = (ent * preco_in + t.real.values[idx].sum(1) * escala * preco_out) / 1e6
        prev = (ent * preco_in + t.previsto_media.values[idx].sum(1) * escala * preco_out) / 1e6
        return real, prev
