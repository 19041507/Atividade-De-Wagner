"""Rodada diária do piloto automático (o GitHub Actions roda este script uma vez por dia).

Para cada pedido sorteado da fila:
  1. o roteador decide Lite ou Flash, antes de chamar a IA;
  2. modo avaliação: os dois modelos respondem, para medir o que teria acontecido com cada escolha;
  3. a IA juíza compara as duas respostas sem saber qual é qual.
No fim, grava data/execucoes/AAAA-MM-DD.csv e retreina o roteador com todos os vereditos acumulados.

Uso: python rodar_piloto.py [--n 50] [--orcamento 0.50] [--meta 0.90]
"""

import argparse
import os
import random
import sys
import time
import tomllib
from datetime import date
from pathlib import Path

import pandas as pd

from core import ia, piloto
from core.modelo import DADOS, ModeloTokens

INSTRUCAO = ("Você é um assistente prestativo e direto. Responda no idioma do pedido, "
             "com clareza, usando listas ou exemplos quando ajudarem.")


def config(nome, padrao=None):
    v = os.environ.get(nome)
    if v:
        return v
    segredos = Path(".streamlit/secrets.toml")
    if segredos.exists():
        return tomllib.loads(segredos.read_text()).get(nome, padrao)
    return padrao


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="pedidos por rodada")
    ap.add_argument("--orcamento", type=float, default=0.50, help="US$ que o piloto pode gastar no dia")
    ap.add_argument("--meta", type=float, default=0.90, help="qualidade mínima que o roteador deve manter")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre chamadas (limite por minuto da chave)")
    ap.add_argument("--saida", default=str(DADOS / "execucoes"))
    args = ap.parse_args()

    chave = config("GEMINI_API_KEY")
    if not chave:
        sys.exit("GEMINI_API_KEY não configurada (variável de ambiente ou .streamlit/secrets.toml).")
    hoje = date.today()
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / f"{hoje}.csv"
    if arquivo.exists():
        print(f"A rodada de {hoje} já foi feita.")
        return

    historico = piloto.carregar_historico(saida)
    fila = pd.read_csv(DADOS / "fila_pedidos.csv")
    if not historico.empty:
        fila = fila[~fila.id.isin(historico.id)]
    if fila.empty:
        sys.exit("A fila de pedidos acabou.")
    pedidos = fila.sample(min(args.n, len(fila)), random_state=int(hoje.strftime("%Y%m%d")))

    modelo = ModeloTokens()
    flash = config("GEMINI_MODEL", "gemini-3.8-flash")
    lite = config("GEMINI_MODEL_LITE") or ia.escolher_reserva(chave, flash)
    if not lite:
        sys.exit("Nenhum modelo Flash-Lite disponível na chave. Defina GEMINI_MODEL_LITE.")
    precos = {
        "flash": (float(config("PRECO_FLASH_ENTRADA", piloto.PRECOS_PADRAO["flash"][0])),
                  float(config("PRECO_FLASH_SAIDA", piloto.PRECOS_PADRAO["flash"][1]))),
        "lite": (float(config("PRECO_LITE_ENTRADA", piloto.PRECOS_PADRAO["lite"][0])),
                 float(config("PRECO_LITE_SAIDA", piloto.PRECOS_PADRAO["lite"][1]))),
    }
    caminho_roteador = saida / "roteador.json"
    roteador = piloto.Roteador.carregar(caminho_roteador) or piloto.regra_inicial(modelo, fila)
    print(f"{hoje}: {len(pedidos)} pedidos · Flash {flash} · Lite {lite} · roteador {roteador.tipo} v{roteador.versao}")

    linhas, gasto_piloto, falhas_seguidas = [], 0.0, 0
    for i, p in enumerate(pedidos.itertuples(), 1):
        x, prev = piloto.caracteristicas(p.texto, p.lang, modelo)
        rota, motivo = roteador.decidir(x)
        estimado_flash = piloto.custo(precos["flash"], prev.prompt_tokens, prev.media)
        if rota == "flash" and gasto_piloto + estimado_flash > args.orcamento:
            rota, motivo = "lite", "orçamento do dia quase no fim"

        pergunta = [{"papel": "user", "texto": p.texto}]
        lite_e_a = random.Random(p.id).random() < 0.5  # ordem sorteada: a juíza não sabe qual é qual
        try:
            rf = ia.conversar(pergunta, chave, flash, INSTRUCAO)
            time.sleep(args.pausa)
            rl = ia.conversar(pergunta, chave, lite, INSTRUCAO)
            time.sleep(args.pausa)
            a, b = (rl, rf) if lite_e_a else (rf, rl)
            v = ia.julgar(p.texto, a.texto, b.texto, chave, flash)
            time.sleep(args.pausa)
        except Exception as e:
            falhas_seguidas += 1
            print(f"  [{i}] falhou: {str(e)[:150]}")
            if falhas_seguidas >= 3:
                print("  3 falhas seguidas: rodada encerrada.")
                break
            continue
        falhas_seguidas = 0

        nota_lite, nota_flash = (v.nota_a, v.nota_b) if lite_e_a else (v.nota_b, v.nota_a)
        vencedor = "empate" if v.melhor == "empate" else ("lite" if (v.melhor == "A") == lite_e_a else "flash")
        custo_flash = piloto.custo(precos["flash"], rf.prompt_tokens, rf.resposta_tokens + rf.raciocinio_tokens)
        custo_lite = piloto.custo(precos["lite"], rl.prompt_tokens, rl.resposta_tokens + rl.raciocinio_tokens)
        jr = v.resposta
        gasto_piloto += custo_lite if rota == "lite" else custo_flash
        linhas.append({
            "data": str(hoje), "id": p.id, "lang": p.lang, "pedido": p.texto,
            "rota": rota, "motivo": motivo, "prob_lite_ok": roteador.prob_lite_ok(x), "versao_roteador": roteador.versao,
            "previsto_p10": prev.p10, "previsto_mediana": prev.mediana, "previsto_p90": prev.p90,
            "modelo_flash": rf.modelo, "modelo_lite": rl.modelo,
            "flash_entrada": rf.prompt_tokens, "flash_saida": rf.resposta_tokens, "flash_raciocinio": rf.raciocinio_tokens,
            "lite_entrada": rl.prompt_tokens, "lite_saida": rl.resposta_tokens, "lite_raciocinio": rl.raciocinio_tokens,
            "custo_flash": custo_flash, "custo_lite": custo_lite,
            "custo_juiza": piloto.custo(precos["flash"], jr.prompt_tokens, jr.resposta_tokens + jr.raciocinio_tokens),
            "nota_flash": nota_flash, "nota_lite": nota_lite, "vencedor": vencedor, "lite_ok": vencedor != "flash",
            "motivo_juiza": v.motivo, "resposta_flash": rf.texto, "resposta_lite": rl.texto,
            **{"x_" + c: x[c] for c in piloto.COLUNAS},
        })
        print(f"  [{i}] {rota:5} | juíza: {vencedor:6} | {motivo}")

    if not linhas:
        sys.exit("Nenhum pedido foi atendido; nada gravado.")
    rodada = pd.DataFrame(linhas)
    rodada.to_csv(arquivo, index=False)

    historico = pd.concat([historico, rodada], ignore_index=True)
    novo = piloto.treinar(historico, args.meta, roteador)
    (novo or roteador).salvar(caminho_roteador)

    r = piloto.avaliar(rodada)
    print(f"\nRodada: {len(rodada)} pedidos · piloto gastou US$ {r.loc['Piloto automático', 'custo']:.4f} "
          f"(economia de {r.loc['Piloto automático', 'economia']:.0%} sobre sempre Flash) "
          f"· qualidade {r.loc['Piloto automático', 'qualidade']:.0%}")
    if novo:
        ult = novo.versoes[-1]
        print(f"Roteador retreinado (v{novo.versao}): {ult['pedidos']} pedidos, AUC {ult['auc']:.2f}, corte {ult['corte']:.2f}")
    else:
        print("Roteador mantido: ainda faltam vereditos para treinar (mínimo de 40, com 5 de cada tipo).")


if __name__ == "__main__":
    main()
