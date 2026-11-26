"""O piloto em produção: decide o modelo de cada pedido real, responde e audita uma amostra em segundo plano."""

import random
from dataclasses import dataclass

import pandas as pd

from core import ia, piloto
from core.banco import Banco

INSTRUCAO = ("Você é um assistente prestativo e direto. Responda no idioma do pedido, "
             "com clareza, usando listas ou exemplos quando ajudarem.")
TAXA_AUDITORIA = 0.2  # 1 em cada 5 pedidos reais também é respondido pelo outro modelo e julgado


def modelos(config, chave: str) -> dict:
    flash = config("GEMINI_MODEL", "gemini-3.8-flash")
    lite = config("GEMINI_MODEL_LITE") or ia.escolher_reserva(chave, flash)
    if not lite:
        raise RuntimeError("Nenhum modelo Flash-Lite disponível na chave. Defina GEMINI_MODEL_LITE.")
    # Sem Pro na chave, o próprio Flash julga (com o risco de preferir o próprio estilo).
    juiza = config("GEMINI_MODEL_JUIZA") or ia.escolher_juiza(chave) or flash
    return {"flash": flash, "lite": lite, "juiza": juiza}


def precos(config) -> dict:
    return {tipo: (float(config(f"PRECO_{tipo.upper()}_ENTRADA", entrada)),
                   float(config(f"PRECO_{tipo.upper()}_SAIDA", saida)))
            for tipo, (entrada, saida) in piloto.PRECOS_PADRAO.items()}


@dataclass
class Decisao:
    rota: str
    motivo: str
    forcado: bool
    prob_lite_ok: float | None
    x: dict
    prev: object


def decidir(texto: str, roteador: piloto.Roteador, modelo_tokens, gasto_hoje: float, orcamento: float) -> Decisao:
    x, prev = piloto.caracteristicas(texto, "pt-BR", modelo_tokens)
    rota, motivo = roteador.decidir(x)
    forcado = False
    if rota == "flash" and gasto_hoje >= orcamento:
        rota, motivo, forcado = "lite", "orçamento do dia esgotado: só o Lite até amanhã", True
    return Decisao(rota, motivo, forcado, roteador.prob_lite_ok(x), x, prev)


def tipo_do_modelo(nome: str, modelos_: dict) -> str:
    return "lite" if nome == modelos_["lite"] else "flash"


def registro(sessao: str, pedido: str, d: Decisao, r: ia.RespostaIA, modelos_: dict, precos_: dict,
             razao_flash_lite: float, versao_roteador: int) -> dict:
    """A linha da tabela pedidos: decisão, resposta entregue, custo pago e quanto o Flash custaria."""
    rota, motivo, forcado = d.rota, d.motivo, d.forcado
    tipo = tipo_do_modelo(r.modelo, modelos_)
    if tipo != rota:  # o Gemini estava sobrecarregado e o outro modelo respondeu
        rota, motivo, forcado = tipo, f"{d.motivo} · o {d.rota.capitalize()} estava indisponível", True
    saida = r.resposta_tokens + r.raciocinio_tokens
    custo = piloto.custo(precos_[rota], r.prompt_tokens, saida)
    custo_flash = custo if rota == "flash" else piloto.custo(precos_["flash"], r.prompt_tokens, saida * razao_flash_lite)
    return {
        "sessao": sessao, "pedido": pedido, "lang": "pt-BR",
        "rota": rota, "motivo": motivo, "forcado": forcado, "prob_lite_ok": d.prob_lite_ok,
        "versao_roteador": versao_roteador, "previsto_p10": d.prev.p10, "previsto_mediana": d.prev.mediana,
        "previsto_p90": d.prev.p90, "x": d.x,
        "modelo": r.modelo, "entrada": r.prompt_tokens, "saida": r.resposta_tokens, "raciocinio": r.raciocinio_tokens,
        "custo": custo, "custo_flash_estimado": custo_flash, "resposta": r.texto,
    }


def auditar(banco: Banco, id_: str, conversa: list[dict], reg: dict, modelos_: dict, precos_: dict, chave: str):
    """Roda numa thread depois que o usuário já recebeu a resposta: gera a do outro modelo e pede o veredito."""
    rota = reg["rota"]
    outro = "flash" if rota == "lite" else "lite"
    try:
        alt = ia.conversar(conversa, chave, modelos_[outro], INSTRUCAO)
        lite, flash = (reg["resposta"], alt.texto) if rota == "lite" else (alt.texto, reg["resposta"])
        c = ia.comparar(reg["pedido"], lite, flash, chave, modelos_["juiza"], lite_e_a=random.random() < 0.5)
        banco.atualizar(id_, {
            "auditado": True, "modelo_alt": alt.modelo, "resposta_alt": alt.texto,
            "custo_alt": piloto.custo(precos_[outro], alt.prompt_tokens, alt.resposta_tokens + alt.raciocinio_tokens),
            "nota_lite": c.nota_lite, "nota_flash": c.nota_flash, "vencedor": c.vencedor, "lite_ok": c.lite_ok,
            "motivo_juiza": c.motivo,
            "custo_juiza": piloto.custo(precos_["juiza"], c.juiza.prompt_tokens,
                                        c.juiza.resposta_tokens + c.juiza.raciocinio_tokens),
        })
    except Exception:
        pass  # a auditoria é uma amostra: se falhar, o pedido só fica fora dela


def resumo_real(r: pd.DataFrame) -> dict | None:
    """Números do uso real: o que o piloto pagou, o que o Flash custaria e a qualidade estimada pela auditoria."""
    if r.empty:
        return None
    flash = piloto.custo_flash_real(r)
    aud = r[r.auditado & r.lite_ok.notna()]
    qualidade = piloto.qualidade((aud.rota == "lite").values, aud.lite_ok.astype(bool).values) if len(aud) else None
    fb = r.feedback.dropna()
    return {
        "pedidos": len(r), "custo": float(r.custo.sum()), "custo_flash": float(flash.sum()),
        "economia": float(1 - r.custo.sum() / flash.sum()) if flash.sum() else 0.0,
        "lite": float((r.rota == "lite").mean()), "auditados": len(aud), "qualidade": qualidade,
        "feedback": len(fb), "aprovacao": float((fb > 0).mean()) if len(fb) else None,
    }


def escrever_diario(rodada: pd.DataFrame, r, roteador, novo, reais_24h: pd.DataFrame, chave: str, modelo: str) -> str:
    """O próprio Gemini escreve o relatório do dia a partir dos números (não inventa: só recebe os números)."""
    erros = rodada[(rodada.rota == "lite") & ~rodada.lite_ok.astype(bool)]
    real = resumo_real(reais_24h)
    fatos = [
        f"Teste controlado: {len(rodada)} pedidos; {r.lite:.0%} foram para o Lite; economia de {r.economia:.0%} "
        f"sobre usar sempre o Flash; qualidade {r.qualidade:.0%}.",
        f"Erros (Lite escolhido e a juíza preferiu o Flash): {len(erros)}. Exemplos de pedidos: "
        + "; ".join(erros.pedido.str.slice(0, 80).head(3)) if len(erros) else "Nenhum erro do Lite na rodada.",
        f"Roteador usado: {roteador.tipo} v{roteador.versao}.",
        (f"Roteador retreinado para v{novo.versao}: {novo.versoes[-1]['pedidos']} rótulos, corte {novo.corte:.2f}, "
         f"AUC {novo.versoes[-1]['auc']:.2f}.") if novo else "O roteador não foi retreinado (poucos rótulos ainda).",
        (f"Uso real nas últimas 24 h: {real['pedidos']} pedidos, economia de {real['economia']:.0%}, "
         f"{real['auditados']} auditados.") if real else "Sem uso real nas últimas 24 h.",
    ]
    instrucao = ("Você é o piloto automático de custo de IA escrevendo o próprio diário de operação, em português, "
                 "na primeira pessoa. Use só os fatos recebidos, sem inventar números. Escreva 3 a 4 frases curtas: "
                 "o que decidiu, onde errou e o que mudou. Sem títulos, sem listas, sem emojis.")
    resp = ia.conversar([{"papel": "user", "texto": "\n".join(fatos)}], chave, modelo, instrucao)
    return resp.texto.strip()


def amostra_auditoria() -> bool:
    return random.random() < TAXA_AUDITORIA


def gasto_e_contagem_hoje(r: pd.DataFrame) -> tuple[float, int]:
    if r.empty:
        return 0.0, 0
    hoje = pd.Timestamp.now(tz="UTC").normalize()
    d = r[r.criado_em >= hoje]
    custo_aud = d.custo_alt.fillna(0).sum() + d.custo_juiza.fillna(0).sum()
    return float(d.custo.sum() + custo_aud), len(d)


def bootstrap_real(r: pd.DataFrame):
    """Intervalos de 95% da economia real e da qualidade estimada pela auditoria."""
    flash = pd.Series(piloto.custo_flash_real(r), index=r.index)
    eco = piloto.intervalo(r.assign(_f=flash), lambda s: 1 - s.custo.sum() / s._f.sum())
    aud = r[r.auditado & r.lite_ok.notna()]
    qual = (piloto.intervalo(aud, lambda s: piloto.qualidade((s.rota == "lite").values, s.lite_ok.astype(bool).values))
            if len(aud) >= 10 else None)
    return eco, qual

