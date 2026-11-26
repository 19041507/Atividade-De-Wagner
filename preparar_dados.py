"""Gera os arquivos auxiliares do app a partir da base oasst1.

- data/tokens_por_idioma.csv: tokens a cada 1.000 caracteres por idioma
- data/fila_pedidos.csv: perguntas reais que o piloto automático atende, uma rodada por dia

Os pares prompt → resposta (data/pares_prompt_resposta.csv) são gerados pelo notebook.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/oasst_ready.messages.jsonl.gz")
URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz"
IDIOMAS = ["en", "es", "pt-BR", "fr", "de", "ru", "zh"]

if not RAW.exists():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, RAW)

msgs = pd.read_json(RAW, lines=True)
enc = tiktoken.get_encoding("o200k_base")
msgs["tokens"] = [len(t) for t in enc.encode_batch(msgs.text.tolist(), disallowed_special=())]
msgs["chars"] = msgs.text.str.len()

idiomas = msgs[msgs.lang.isin(IDIOMAS)].groupby("lang")[["tokens", "chars"]].sum()
idiomas["tokens_por_1000_caracteres"] = (1000 * idiomas.tokens / idiomas.chars).round(1)
idiomas.to_csv("data/tokens_por_idioma.csv")

respostas = msgs[(msgs.role == "assistant") & (msgs["rank"] == 0)]

# Fila do piloto: perguntas do conjunto de TESTE do modelo de tokens (mesma ordem dos pares e mesmo
# train_test_split de core/modelo.py), para o roteador nunca decidir sobre algo que o previsor viu no treino.
prompts = msgs.set_index("message_id").loc[respostas.parent_id].reset_index()
_, teste = train_test_split(np.arange(len(prompts)), test_size=0.25, random_state=42)
fila = prompts.iloc[np.sort(teste)]
fila = fila[fila.lang.isin(["pt-BR", "en", "es"])           # idiomas que a juíza e as pessoas conseguem ler
            & fila.parent_id.isna()                          # primeira mensagem: o pedido se explica sozinho
            & fila.tokens.between(8, 400)]                   # sem "oi" e sem textos colados enormes
fila = pd.DataFrame({
    "id": fila.message_id.values,
    "lang": fila.lang.values,
    "texto": fila.text.values,
})
fila.to_csv("data/fila_pedidos.csv", index=False)
print(f"ok: {len(fila)} pedidos na fila")
