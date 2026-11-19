"""Chamadas ao Gemini: gera a resposta e devolve os tokens que a própria API contabiliza."""

import time
from dataclasses import dataclass

import pandas as pd

from core.modelo import DADOS


@dataclass
class RespostaIA:
    texto: str
    prompt_tokens: int
    resposta_tokens: int
    raciocinio_tokens: int  # "thinking": cobrado como saída, mas não aparece no texto
    segundos: float


def _cliente(chave: str):
    from google import genai
    return genai.Client(api_key=chave)


def gerar(prompt: str, chave: str, modelo: str, max_saida: int | None = None) -> RespostaIA:
    from google.genai import types
    config = types.GenerateContentConfig(max_output_tokens=max_saida) if max_saida else None
    inicio = time.perf_counter()
    r = _cliente(chave).models.generate_content(model=modelo, contents=prompt, config=config)
    uso = r.usage_metadata
    return RespostaIA(
        texto=r.text or "",
        prompt_tokens=uso.prompt_token_count or 0,
        resposta_tokens=uso.candidates_token_count or 0,
        raciocinio_tokens=uso.thoughts_token_count or 0,
        segundos=time.perf_counter() - inicio,
    )


def fator_calibracao(chave: str, modelo: str) -> float:
    """Tokens do Gemini por token do o200k_base, medido numa amostra de 300 respostas reais da base.

    O modelo de previsão foi treinado em tokens o200k_base; este fator converte a previsão
    para a unidade que o Gemini cobra.
    """
    amostra = pd.read_csv(DADOS / "amostra_calibracao.csv")
    texto = "\n\n".join(amostra.text)
    r = _cliente(chave).models.count_tokens(model=modelo, contents=texto)
    return r.total_tokens / amostra.tokens.sum()
