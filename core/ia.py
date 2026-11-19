"""Chamadas ao Gemini: gera a resposta e devolve os tokens que a própria API contabiliza."""

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.modelo import DADOS

TIMEOUT_MS = 40_000                  # sem receber nada nesse tempo, a tentativa é abandonada
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
    modelo: str


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


def _tempo_esgotado(e: Exception) -> bool:
    return isinstance(e, TimeoutError) or any(c.__name__ == "TimeoutException" for c in type(e).__mro__)


def _rejeita_raciocinio(e: Exception) -> bool:
    from google.genai import errors
    return isinstance(e, errors.APIError) and e.code == 400 and "think" in str(e).lower()


def escolher_reserva(chave: str, principal: str) -> str | None:
    """Entre os modelos que a chave tem, escolhe um Flash mais leve para usar se o principal falhar."""
    import re
    cliente = _cliente(chave)
    nomes = [m.name.removeprefix("models/") for m in cliente.models.list()
             if "generateContent" in (m.supported_actions or [])]
    excluir = ("tts", "image", "live", "audio", "embedding", "exp")
    candidatos = [n for n in nomes if "flash" in n and n != principal and not any(x in n for x in excluir)]
    if not candidatos:
        return None

    def ordem(n):
        versao = re.search(r"(\d+(?:\.\d+)?)", n)
        return ("lite" not in n, "preview" in n, -float(versao.group(1)) if versao else 0)
    return sorted(candidatos, key=ordem)[0]


def conversar(mensagens: list[dict], chave: str, modelo: str, instrucao: str,
              reserva: str | None = None, ao_receber: Callable[[str], None] | None = None) -> RespostaIA:
    """Gera a resposta em streaming: ao_receber é chamado com o texto acumulado a cada pedaço.

    mensagens: [{"papel": "user" | "model", "texto": str}, ...], a última é a pergunta atual.
    O modelo é instruído a raciocinar o mínimo (respostas começam rápido e custam menos); se o modelo
    não aceitar essa opção, tenta sem ela. Em erro transitório antes do primeiro pedaço, tenta de novo
    no mesmo modelo; se demorar demais ou esgotar as tentativas, passa para o modelo reserva.
    """
    from google.genai import types
    contents = [types.Content(role=m["papel"], parts=[types.Part(text=m["texto"])]) for m in mensagens]
    cliente = _cliente(chave)
    fila = [modelo] * (len(ESPERAS) + 1)
    reserva_pendente = bool(reserva) and reserva != modelo
    pensar_pouco, esperas = True, iter(ESPERAS)
    comeco = time.perf_counter()

    while True:
        nome = fila.pop(0)
        config = types.GenerateContentConfig(
            system_instruction=instrucao,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL) if pensar_pouco else None,
        )
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
            if pensar_pouco and _rejeita_raciocinio(e):
                pensar_pouco = False
                fila.insert(0, nome)
                continue
            if not _transitorio(e):
                raise
            if texto:  # caiu no meio da resposta: não reenvia para não cobrar duas vezes
                raise GeminiIndisponivel(str(e)) from e
            demorou = _tempo_esgotado(e) or time.perf_counter() - comeco > PRAZO_PARA_NOVA_TENTATIVA
            if (demorou or not fila) and reserva_pendente:
                fila, reserva_pendente = [reserva], False
            elif demorou:
                fila = []
            if not fila:
                raise GeminiIndisponivel(str(e)) from e
            if fila[0] == nome:
                time.sleep(next(esperas, 0))

    return RespostaIA(
        texto=texto,
        prompt_tokens=(uso and uso.prompt_token_count) or 0,
        resposta_tokens=(uso and uso.candidates_token_count) or 0,
        raciocinio_tokens=(uso and uso.thoughts_token_count) or 0,
        segundos=time.perf_counter() - inicio,
        modelo=nome,
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
