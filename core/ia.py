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
    # O chamador precisa guardar o cliente numa variável: se ele for descartado,
    # a biblioteca fecha a conexão antes da requisição sair.
    from google import genai
    return genai.Client(api_key=chave)


TRANSITORIOS = {429, 500, 503, 504}  # sobrecarga ou limite: vale tentar de novo
ESPERAS = [2, 5]  # segundos entre as tentativas


def conversar(mensagens: list[dict], chave: str, modelo: str, instrucao: str,
              reserva: str | None = None) -> RespostaIA:
    """mensagens: [{"papel": "user" | "model", "texto": str}, ...], a última é a pergunta atual.

    Em erro transitório, tenta de novo no mesmo modelo e, se houver, por último no modelo reserva.
    """
    from google.genai import errors, types
    contents = [types.Content(role=m["papel"], parts=[types.Part(text=m["texto"])]) for m in mensagens]
    config = types.GenerateContentConfig(system_instruction=instrucao)
    cliente = _cliente(chave)
    tentativas = [modelo] * (len(ESPERAS) + 1) + ([reserva] if reserva else [])
    for i, nome in enumerate(tentativas):
        try:
            inicio = time.perf_counter()
            r = cliente.models.generate_content(model=nome, contents=contents, config=config)
            break
        except errors.APIError as e:
            if e.code not in TRANSITORIOS or i == len(tentativas) - 1:
                raise
            time.sleep(ESPERAS[i] if i < len(ESPERAS) else 0)
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
    cliente = _cliente(chave)
    r = cliente.models.count_tokens(model=modelo, contents=texto)
    return r.total_tokens / amostra.tokens.sum()
