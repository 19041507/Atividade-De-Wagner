"""Rodada diária do piloto automático (o GitHub Actions roda este script uma vez por dia).

Teste controlado: para cada pedido sorteado da fila,
  1. o roteador decide Lite ou Flash, antes de chamar a IA;
  2. os dois modelos respondem, para medir o que teria acontecido com cada escolha;
  3. a IA juíza compara as duas respostas sem saber qual é qual.
Depois retreina o roteador com tudo o que tem rótulo: as rodadas controladas e os pedidos reais do assistente
(auditados pela juíza ou com 👍/👎 do usuário), escreve o diário do dia e avisa se a qualidade caiu.

Uso: python rodar_piloto.py [--n 50] [--orcamento 0.50] [--meta 0.90] [--forcar]
"""

import argparse
import os
import random
import sys
import time
import tomllib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core import banco as banco_mod
from core import ia, piloto, producao
from core.modelo import DADOS, ModeloTokens

ESPERAS_SOBRECARGA = [30, 60, 120]  # segundos: a rodada não tem ninguém esperando, vale aguardar a sobrecarga passar


def com_paciencia(chamada):
    for espera in ESPERAS_SOBRECARGA + [None]:
        try:
            return chamada()
        except ia.GeminiIndisponivel as e:
            if espera is None or ia.cota_diaria_esgotada(e):
                raise
            print(f"    Gemini indisponível ({str(e)[:40]}…); nova tentativa em {espera}s")
            time.sleep(espera)


def config(nome, padrao=None):
    v = os.environ.get(nome)
    if v:
        return v
    segredos = Path(".streamlit/secrets.toml")
    if segredos.exists():
        return tomllib.loads(segredos.read_text()).get(nome, padrao)
    return padrao


def avisar(texto: str):
    """No GitHub Actions, vira a saída 'alerta' do passo e o workflow abre uma issue."""
    print(f"ALERTA: {texto}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"alerta={texto}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="pedidos por rodada (3 chamadas cada: cabe na cota gratuita)")
    ap.add_argument("--orcamento", type=float, default=0.50, help="US$ que o piloto pode gastar na rodada")
    ap.add_argument("--meta", type=float, default=0.90, help="qualidade mínima que o roteador deve manter")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre chamadas (limite por minuto da chave)")
    ap.add_argument("--forcar", action="store_true", help="faz outra rodada mesmo que a de hoje já exista")
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
        if not args.forcar:
            print(f"A rodada de {hoje} já foi feita (use --forcar para outra).")
            return
        arquivo = saida / f"{hoje}-{datetime.now():%H%M}.csv"

    historico = piloto.carregar_historico(saida)
    fila = pd.read_csv(DADOS / "fila_pedidos.csv")
    if not historico.empty:
        fila = fila[~fila.id.isin(historico.id)]
    if fila.empty:
        sys.exit("A fila de pedidos acabou.")
    pedidos = fila.sample(min(args.n, len(fila)), random_state=int(f"{datetime.now():%Y%m%d%H%M}") % 2**32)

    modelo = ModeloTokens()
    modelos = producao.modelos(config, chave)
    precos = producao.precos(config)
    caminho_roteador = saida / "roteador.json"
    roteador = piloto.Roteador.carregar(caminho_roteador) or piloto.regra_inicial(modelo, fila)
    print(f"{arquivo.stem}: {len(pedidos)} pedidos · Flash {modelos['flash']} · Lite {modelos['lite']} "
          f"· juíza {modelos['juiza']} · roteador {roteador.tipo} v{roteador.versao}")

    linhas, gasto_piloto, falhas_seguidas = [], 0.0, 0
    for i, p in enumerate(pedidos.itertuples(), 1):
        x, prev = piloto.caracteristicas(p.texto, p.lang, modelo)
        rota, motivo = roteador.decidir(x)
        estimado_flash = piloto.custo(precos["flash"], prev.prompt_tokens, prev.media)
        if rota == "flash" and gasto_piloto + estimado_flash > args.orcamento:
            rota, motivo = "lite", "orçamento do dia quase no fim"

        pergunta = [{"papel": "user", "texto": p.texto}]
        try:
            rf = com_paciencia(lambda: ia.conversar(pergunta, chave, modelos["flash"], producao.INSTRUCAO))
            time.sleep(args.pausa)
            rl = com_paciencia(lambda: ia.conversar(pergunta, chave, modelos["lite"], producao.INSTRUCAO))
            time.sleep(args.pausa)
            c = com_paciencia(lambda: ia.comparar(p.texto, rl.texto, rf.texto, chave, modelos["juiza"],
                                                  lite_e_a=random.Random(p.id).random() < 0.5))
            time.sleep(args.pausa)
        except Exception as e:
            if ia.cota_diaria_esgotada(e):
                print(f"  [{i}] cota diária do Gemini esgotada ({ia.detalhe_da_cota(e)}).")
                print(f"  Rodada encerrada com {len(linhas)} pedidos; os próximos ficam para amanhã.")
                break
            falhas_seguidas += 1
            print(f"  [{i}] falhou: {ia.detalhe_da_cota(e) if '429' in str(e) else str(e)[:300]}")
            if falhas_seguidas >= 3:
                print("  3 falhas seguidas: rodada encerrada.")
                break
            continue
        falhas_seguidas = 0

        custo_flash = piloto.custo(precos["flash"], rf.prompt_tokens, rf.resposta_tokens + rf.raciocinio_tokens)
        custo_lite = piloto.custo(precos["lite"], rl.prompt_tokens, rl.resposta_tokens + rl.raciocinio_tokens)
        gasto_piloto += custo_lite if rota == "lite" else custo_flash
        linhas.append({
            "data": str(hoje), "id": p.id, "lang": p.lang, "pedido": p.texto,
            "rota": rota, "motivo": motivo, "prob_lite_ok": roteador.prob_lite_ok(x), "versao_roteador": roteador.versao,
            "previsto_p10": prev.p10, "previsto_mediana": prev.mediana, "previsto_p90": prev.p90,
            "modelo_flash": rf.modelo, "modelo_lite": rl.modelo, "modelo_juiza": c.juiza.modelo,
            "flash_entrada": rf.prompt_tokens, "flash_saida": rf.resposta_tokens, "flash_raciocinio": rf.raciocinio_tokens,
            "lite_entrada": rl.prompt_tokens, "lite_saida": rl.resposta_tokens, "lite_raciocinio": rl.raciocinio_tokens,
            "custo_flash": custo_flash, "custo_lite": custo_lite,
            "custo_juiza": piloto.custo(precos["juiza"], c.juiza.prompt_tokens,
                                        c.juiza.resposta_tokens + c.juiza.raciocinio_tokens),
            "nota_flash": c.nota_flash, "nota_lite": c.nota_lite, "vencedor": c.vencedor, "lite_ok": c.lite_ok,
            "motivo_juiza": c.motivo, "resposta_flash": rf.texto, "resposta_lite": rl.texto,
            **{"x_" + col: x[col] for col in piloto.COLUNAS},
        })
        print(f"  [{i}] {rota:5} | juíza: {c.vencedor:6} | {motivo}")

    if not linhas:
        sys.exit("Nenhum pedido foi atendido; nada gravado.")
    rodada = pd.DataFrame(linhas)
    rodada.to_csv(arquivo, index=False)

    # Uso real das últimas 24 h e todos os rótulos reais acumulados.
    banco = banco_mod.conectar(config("SUPABASE_URL"), config("SUPABASE_KEY"))
    reais = pd.DataFrame()
    if banco:
        try:
            reais = banco.listar()
        except Exception as e:
            print(f"Não consegui ler o uso real do Supabase: {str(e)[:150]}")
    rotulados = piloto.rotulos_reais(reais)

    historico = pd.concat([historico, rodada], ignore_index=True)
    treino = pd.concat([historico[["x_" + c for c in piloto.COLUNAS] + ["lite_ok", "data"]], rotulados],
                       ignore_index=True)
    novo = piloto.treinar(treino, args.meta, roteador)
    (novo or roteador).salvar(caminho_roteador)

    r = piloto.avaliar(rodada).loc["Piloto automático"]
    print(f"\nRodada: {len(rodada)} pedidos · piloto gastou US$ {r.custo:.4f} "
          f"(economia de {r.economia:.0%} sobre sempre Flash) · qualidade {r.qualidade:.0%}")
    if novo:
        ult = novo.versoes[-1]
        print(f"Roteador retreinado (v{novo.versao}): {ult['pedidos']} rótulos ({len(rotulados)} do uso real), "
              f"AUC {ult['auc']:.2f}, corte {ult['corte']:.2f}")
    else:
        print("Roteador mantido: ainda faltam vereditos para treinar (mínimo de 40, com 5 de cada tipo).")

    ontem = datetime.now(timezone.utc) - timedelta(days=1)
    reais_24h = reais[reais.criado_em >= ontem] if not reais.empty else reais
    try:
        diario = producao.escrever_diario(rodada, r, roteador, novo, reais_24h, chave, modelos["flash"])
        arquivo.with_suffix(".md").write_text(diario)
        print(f"\nDiário:\n{diario}")
    except Exception as e:
        print(f"Diário não escrito: {str(e)[:150]}")

    if r.qualidade < args.meta:
        corte = f"; corte do roteador agora {novo.corte:.2f}" if novo else ""
        avisar(f"qualidade da rodada {arquivo.stem} caiu para {r.qualidade:.0%} (meta {args.meta:.0%}){corte}")


if __name__ == "__main__":
    main()
