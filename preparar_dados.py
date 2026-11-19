"""Gera os arquivos auxiliares do app a partir da base oasst1.

- data/tokens_por_idioma.csv: tokens a cada 1.000 caracteres por idioma
- data/amostra_calibracao.csv: 300 respostas reais, usadas para calibrar o tokenizador do Gemini

Os pares prompt → resposta (data/pares_prompt_resposta.csv) são gerados pelo notebook.
"""

import urllib.request
from pathlib import Path

import pandas as pd
import tiktoken

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
respostas.sample(300, random_state=7)[["lang", "text", "tokens"]].to_csv("data/amostra_calibracao.csv", index=False)
print("ok")
