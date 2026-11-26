"""Gera data/validacao_humana.csv: 30 pares de respostas para uma pessoa julgar, como a IA juíza fez.

Preencha a coluna sua_escolha com A, B ou empate, sem olhar a última coluna (gabarito_a diz qual modelo é a A).
O app compara as suas escolhas com as da juíza e mostra a concordância.
"""

import random
import sys
from pathlib import Path

import pandas as pd

from core import piloto
from core.modelo import DADOS

ARQUIVO = DADOS / "validacao_humana.csv"

if ARQUIVO.exists():
    sys.exit(f"{ARQUIVO} já existe; apague-o para sortear outros pares.")
h = piloto.carregar_historico(DADOS / "execucoes")
if len(h) < 30:
    sys.exit("Ainda não há 30 pedidos avaliados pela juíza.")

linhas = []
for p in h.sample(30, random_state=1).itertuples():
    lite_e_a = random.Random(f"humano-{p.id}").random() < 0.5
    a, b = (p.resposta_lite, p.resposta_flash) if lite_e_a else (p.resposta_flash, p.resposta_lite)
    linhas.append({"id": p.id, "pedido": p.pedido, "resposta_a": a, "resposta_b": b, "sua_escolha": "",
                   "gabarito_a": "lite" if lite_e_a else "flash"})
pd.DataFrame(linhas).to_csv(ARQUIVO, index=False)
print(f"ok: {ARQUIVO}")
