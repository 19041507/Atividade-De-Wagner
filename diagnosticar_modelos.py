"""Diagnóstico da chave do Gemini: quais modelos Flash, Lite e Pro respondem e qual a cota de cada um.

Faz uma chamada mínima por modelo (poucos tokens). Roda pelo workflow "Diagnóstico dos modelos" ou localmente:
GEMINI_API_KEY=... python diagnosticar_modelos.py
"""

import os
import re
import sys

from google import genai
from google.genai import types

from core import ia

chave = os.environ.get("GEMINI_API_KEY") or sys.exit("GEMINI_API_KEY não configurada.")
cliente = genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=60_000))
excluir = ("tts", "image", "live", "audio", "embedding", "computer", "robotics", "native")
# Modelos reservados para a demonstração ao vivo: o diagnóstico não gasta a cota deles.
reservados = os.environ.get("MODELOS_RESERVADOS", "gemini-3.8-flash,gemini-3.7-flash").split(",")
nomes = sorted(m.name.removeprefix("models/") for m in cliente.models.list()
               if "generateContent" in (m.supported_actions or [])
               and re.search(r"flash|pro", m.name) and not any(x in m.name for x in excluir))
for nome in [n for n in nomes if n in reservados]:
    print(f"{nome:40} 🔒 reservado          não testado, para não gastar a cota")
nomes = [n for n in nomes if n not in reservados]

linhas = []
for nome in nomes:
    try:
        r = cliente.models.generate_content(
            model=nome, contents="Responda apenas: ok",
            config=types.GenerateContentConfig(max_output_tokens=200))
        situacao, detalhe = "✅ responde", (r.text or "").strip()[:20]
    except Exception as e:
        texto = str(e)
        if "429" in texto:
            situacao, detalhe = "⏳ sem cota agora", ia.detalhe_da_cota(e)
        elif "404" in texto or "no longer available" in texto:
            situacao, detalhe = "⛔ indisponível", "não liberado para esta conta"
        else:
            situacao, detalhe = "❌ erro", texto[:120]
    linhas.append((nome, situacao, detalhe))
    print(f"{nome:40} {situacao:18} {detalhe}")

if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write("| Modelo | Situação | Detalhe |\n|---|---|---|\n")
        f.writelines(f"| `{n}` | {s} | {d} |\n" for n, s, d in linhas)
