"""Chamadas ao Gemini: gera a resposta e devolve os tokens que a própria API contabiliza."""

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.modelo import DADOS

TIMEOUT_MS = 60_000                  # sem resposta nesse tempo, a tentativa é abandonada
PRAZO_PARA_NOVA_TENTATIVA = 45       # segundos: depois disso, não vale fazer a pessoa esperar mais
TRANSITORIOS = {429, 500, 503, 504}  # sobrecarga ou limite: vale tentar de novo
ESPERAS = [2, 5]                     # segundos entre as tentativas


class GeminiIndisponivel(Exception):
    """O Gemini não respondeu (sobrecarga ou tempo esgotado) depois de todas as tentativas."""


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
    from google.genai import types
    return genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def _transitorio(e: Exception) -> bool:
    from google.genai import errors
    if isinstance(e, errors.APIError):
        return e.code in TRANSITORIOS
    # Tempo esgotado ou queda de rede. Compara pelo nome da classe porque o SDK pode usar httpx ou httpx2.
    return isinstance(e, TimeoutError) or any(
        c.__name__ in ("TimeoutException", "NetworkError") for c in type(e).__mro__)


def conversar(mensagens: list[dict], chave: str, modelo: str, instrucao: str,
              reserva: str | None = None, ao_receber: Callable[[str], None] | None = None) -> RespostaIA:
    """Gera a resposta em streaming: ao_receber é chamado com o texto acumulado a cada pedaço.

    mensagens: [{"papel": "user" | "model", "texto": str}, ...], a última é a pergunta atual.
    Em erro transitório antes do primeiro pedaço, tenta de novo no mesmo modelo e, se houver,
    por último no modelo reserva.
    """
    from google.genai import types
    contents = [types.Content(role=m["papel"], parts=[types.Part(text=m["texto"])]) for m in mensagens]
    config = types.GenerateContentConfig(system_instruction=instrucao)
    cliente = _cliente(chave)
    tentativas = [modelo] * (len(ESPERAS) + 1) + ([reserva] if reserva else [])
    comeco = time.perf_counter()

    for i, nome in enumerate(tentativas):
        texto, uso = "", None
        inicio = time.perf_counter()
        try:
            for pedaco in cliente.models.generate_content_stream(model=nome, contents=contents, config=config):
                if pedaco.text:
                    texto += pedaco.text
                    if ao_receber:
                        ao_receber(texto)
                if pedaco.usage_metadata:
                    uso = pedaco.usage_metadata
            break
        except Exception as e:
            ultima = i == len(tentativas) - 1
            sem_tempo = time.perf_counter() - comeco > PRAZO_PARA_NOVA_TENTATIVA
            if not _transitorio(e):
                raise
            if texto or ultima or sem_tempo:  # caiu no meio da resposta: não reenvia para não cobrar duas vezes
                raise GeminiIndisponivel(str(e)) from e
            time.sleep(ESPERAS[i] if i < len(ESPERAS) else 0)

    return RespostaIA(
        texto=texto,
        prompt_tokens=(uso and uso.prompt_token_count) or 0,
        resposta_tokens=(uso and uso.candidates_token_count) or 0,
        raciocinio_tokens=(uso and uso.thoughts_token_count) or 0,
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
