"""Chamadas ao Gemini: gera a resposta e devolve os tokens que a própria API contabiliza."""

import time
from dataclasses import dataclass
from typing import Callable

TIMEOUT_MS = 40_000                  # sem receber nada nesse tempo, a tentativa é abandonada
TIMEOUT_JUIZA_MS = 120_000          # o Pro raciocina antes de começar a responder
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
    cortada: bool = False  # a resposta bateu no teto de tokens


def _cliente(chave: str, timeout_ms: int = TIMEOUT_MS):
    # O chamador precisa guardar o cliente numa variável: se ele for descartado,
    # a biblioteca fecha a conexão antes da requisição sair.
    from google import genai
    from google.genai import types
    return genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=timeout_ms))


def _transitorio(e: Exception) -> bool:
    from google.genai import errors
    if isinstance(e, errors.APIError):
        return e.code in TRANSITORIOS
    # Tempo esgotado ou queda de rede. Compara pelo nome da classe porque o SDK pode usar httpx ou httpx2.
    return isinstance(e, TimeoutError) or any(
        c.__name__ in ("TimeoutException", "NetworkError") for c in type(e).__mro__)


def _tempo_esgotado(e: Exception) -> bool:
    return isinstance(e, TimeoutError) or any(c.__name__ == "TimeoutException" for c in type(e).__mro__)


def cota_diaria_esgotada(erro) -> bool:
    """429 por limite diário (não adianta esperar minutos, só no dia seguinte)."""
    texto = str(erro)
    return "429" in texto and "PerDay" in texto


def detalhe_da_cota(erro) -> str:
    """Qual cota estourou: métrica, modelo e limite, como o Google informa no corpo do 429."""
    import re
    texto = str(erro)
    partes = [f"{campo}={m.group(1)}" for campo, padrao in [
        ("cota", r"'quotaId': '([^']+)'"), ("modelo", r"'model': '([^']+)'"),
        ("limite", r"'quotaValue': '([^']+)'"), ("tentar em", r"'retryDelay': '([^']+)'")]
        if (m := re.search(padrao, texto))]
    return ", ".join(partes) or texto[:300]


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


def escolher_juiza(chave: str) -> str | None:
    """A juíza deve ser maior que os dois modelos avaliados e diferente deles: o Pro mais recente da chave."""
    import re
    cliente = _cliente(chave)
    nomes = [m.name.removeprefix("models/") for m in cliente.models.list()
             if "generateContent" in (m.supported_actions or [])]
    excluir = ("tts", "image", "live", "audio", "embedding", "exp", "computer")
    candidatos = [n for n in nomes if "pro" in n and not any(x in n for x in excluir)]
    if not candidatos:
        return None

    def ordem(n):
        versao = re.search(r"(\d+(?:\.\d+)?)", n)
        return ("preview" in n, -float(versao.group(1)) if versao else 0, len(n))
    # A lista traz modelos que a conta não pode usar ("no longer available to new users"): testa com uma
    # chamada mínima e fica com o primeiro liberado. Sem cota agora (429) conta como liberado: a juíza não
    # deve mudar de um dia para o outro só porque a cota acabou.
    from google.genai import types
    for nome in sorted(candidatos, key=ordem):
        try:
            cliente.models.generate_content(model=nome, contents="ok",
                                            config=types.GenerateContentConfig(max_output_tokens=1))
            return nome
        except Exception as e:
            texto = str(e)
            if "404" in texto or "403" in texto or "no longer available" in texto:
                continue
            return nome
    return None


def conversar(mensagens: list[dict], chave: str, modelo: str, instrucao: str,
              reserva: str | None = None, ao_receber: Callable[[str], None] | None = None,
              esquema_json: dict | None = None, timeout_ms: int = TIMEOUT_MS,
              max_tokens: int | None = None) -> RespostaIA:
    """Gera a resposta em streaming: ao_receber é chamado com o texto acumulado a cada pedaço.

    mensagens: [{"papel": "user" | "model", "texto": str}, ...], a última é a pergunta atual.
    O modelo é instruído a raciocinar o mínimo (respostas começam rápido e custam menos); se o modelo
    não aceitar essa opção, tenta sem ela. Em erro transitório antes do primeiro pedaço, tenta de novo
    no mesmo modelo; se demorar demais ou esgotar as tentativas, passa para o modelo reserva.
    """
    from google.genai import types
    contents = [types.Content(role=m["papel"], parts=[types.Part(text=m["texto"])]) for m in mensagens]
    cliente = _cliente(chave, timeout_ms)
    fila = [modelo] * (len(ESPERAS) + 1)
    reserva_pendente = bool(reserva) and reserva != modelo
    pensar_pouco, esperas = True, iter(ESPERAS)
    comeco = time.perf_counter()

    while True:
        nome = fila.pop(0)
        config = types.GenerateContentConfig(
            system_instruction=instrucao,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL) if pensar_pouco else None,
            response_mime_type="application/json" if esquema_json else None,
            response_json_schema=esquema_json,
            max_output_tokens=max_tokens,
        )
        texto, uso, fim = "", None, None
        inicio = time.perf_counter()
        try:
            for pedaco in cliente.models.generate_content_stream(model=nome, contents=contents, config=config):
                if pedaco.text:
                    texto += pedaco.text
                    if ao_receber:
                        ao_receber(texto)
                if pedaco.usage_metadata:
                    uso = pedaco.usage_metadata
                if pedaco.candidates and pedaco.candidates[0].finish_reason:
                    fim = pedaco.candidates[0].finish_reason
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
        cortada="MAX_TOKENS" in str(fim),
    )


INSTRUCAO_JUIZA = (
    "Você avalia respostas de assistentes de IA. Recebe um pedido e duas respostas, A e B. "
    "Julgue só se cada uma atende ao pedido: correção, completude e clareza. "
    "Não favoreça a resposta mais longa nem a que aparece primeiro: uma resposta curta e certa vale tanto quanto uma longa e certa. "
    "Dê nota de 1 a 5 para cada e diga qual é melhor, ou 'empate' se um usuário ficaria igualmente bem servido com qualquer uma. "
    "O motivo deve ter no máximo uma frase, em português."
)
ESQUEMA_JUIZA = {
    "type": "object",
    "properties": {
        "nota_a": {"type": "integer", "minimum": 1, "maximum": 5},
        "nota_b": {"type": "integer", "minimum": 1, "maximum": 5},
        "melhor": {"type": "string", "enum": ["A", "B", "empate"]},
        "motivo": {"type": "string"},
    },
    "required": ["nota_a", "nota_b", "melhor", "motivo"],
}


@dataclass
class Veredito:
    nota_a: int
    nota_b: int
    melhor: str  # "A", "B" ou "empate"
    motivo: str
    resposta: RespostaIA  # a chamada da juíza também gasta tokens


def julgar(pedido: str, resposta_a: str, resposta_b: str, chave: str, modelo: str,
           reserva: str | None = None) -> Veredito:
    """IA juíza: compara duas respostas ao mesmo pedido sem saber qual modelo gerou cada uma."""
    import json
    texto = f"PEDIDO:\n{pedido}\n\nRESPOSTA A:\n{resposta_a}\n\nRESPOSTA B:\n{resposta_b}"
    r = conversar([{"papel": "user", "texto": texto}], chave, modelo, INSTRUCAO_JUIZA,
                  reserva=reserva, esquema_json=ESQUEMA_JUIZA, timeout_ms=TIMEOUT_JUIZA_MS)
    v = json.loads(r.texto)
    return Veredito(int(v["nota_a"]), int(v["nota_b"]), v["melhor"], v["motivo"], r)


@dataclass
class Comparacao:
    nota_lite: int
    nota_flash: int
    vencedor: str  # "lite", "flash" ou "empate"
    motivo: str
    juiza: RespostaIA

    @property
    def lite_ok(self) -> bool:
        return self.vencedor != "flash"


def comparar(pedido: str, resposta_lite: str, resposta_flash: str, chave: str, modelo: str,
             lite_e_a: bool) -> Comparacao:
    """Julga Lite × Flash às cegas: lite_e_a (sorteado por quem chama) decide qual vai como resposta A."""
    a, b = (resposta_lite, resposta_flash) if lite_e_a else (resposta_flash, resposta_lite)
    v = julgar(pedido, a, b, chave, modelo)
    nota_lite, nota_flash = (v.nota_a, v.nota_b) if lite_e_a else (v.nota_b, v.nota_a)
    vencedor = "empate" if v.melhor == "empate" else ("lite" if (v.melhor == "A") == lite_e_a else "flash")
    return Comparacao(nota_lite, nota_flash, vencedor, v.motivo, v.resposta)

