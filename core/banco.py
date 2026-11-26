"""Uso real do assistente, guardado no Supabase (tabela pedidos) pela API REST.

O Streamlit Cloud apaga os arquivos quando reinicia; o app e o workflow diário leem e gravam aqui.
"""

import json

import pandas as pd
import requests

TIMEOUT = 15


class Banco:
    def __init__(self, url: str, chave: str):
        base = url.strip().rstrip("/").removesuffix("/rest/v1")
        self.url = f"{base}/rest/v1/pedidos"
        self.cabecalho = {"apikey": chave, "Content-Type": "application/json"}

    def inserir(self, registro: dict) -> str:
        r = requests.post(self.url, headers={**self.cabecalho, "Prefer": "return=representation"},
                          data=json.dumps(registro, default=str), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()[0]["id"]

    def atualizar(self, id_: str, campos: dict):
        r = requests.patch(self.url, params={"id": f"eq.{id_}"}, headers=self.cabecalho,
                           data=json.dumps(campos, default=str), timeout=TIMEOUT)
        r.raise_for_status()

    def listar(self, colunas: str = "*", desde: str | None = None) -> pd.DataFrame:
        linhas, inicio, pagina = [], 0, 1000
        filtro = {"criado_em": f"gte.{desde}"} if desde else {}
        while True:
            r = requests.get(self.url, params={"select": colunas, "order": "criado_em", "limit": pagina,
                                               "offset": inicio, **filtro},
                             headers=self.cabecalho, timeout=TIMEOUT)
            r.raise_for_status()
            lote = r.json()
            linhas += lote
            if len(lote) < pagina:
                break
            inicio += pagina
        df = pd.DataFrame(linhas)
        if not df.empty:
            df["criado_em"] = pd.to_datetime(df.criado_em, utc=True)
        return df


def conectar(url: str | None, chave: str | None) -> Banco | None:
    return Banco(url, chave) if url and chave else None
