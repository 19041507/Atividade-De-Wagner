# TokenScope: piloto automático de custo de IA

Atividade de IA aplicada e IA experimental (Prof. Wagner, 3º ciclo).

Uma aplicação com IA generativa paga por **token**, e o modelo mais capaz custa várias vezes mais que o mais simples. Mandar tudo para o modelo caro desperdiça dinheiro; mandar tudo para o barato derruba a qualidade.

O **TokenScope** é um piloto automático que resolve isso sozinho, todo dia, sem ninguém clicar:

1. o GitHub Actions dispara uma rodada com pedidos reais que o sistema ainda não atendeu;
2. para cada pedido, um **roteador** decide entre o Gemini **Flash-Lite** (barato) e o **Flash** (mais capaz), antes de chamar a IA;
3. os dois modelos respondem e uma **IA juíza** compara as respostas sem saber qual é qual;
4. as notas da juíza viram dados de treino e o roteador **se retreina**;
5. o resultado é salvo no repositório e o app publica: quanto o piloto economizou e quanta qualidade manteve.

## Roteiro da atividade

| Etapa | No projeto |
|---|---|
| **Entender o problema** | Gastar menos com IA sem piorar as respostas, decidindo o modelo de cada pedido. |
| **Base de dados** | [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache 2.0): 18.924 pares pergunta → resposta para treinar o previsor; os pedidos do piloto saem do conjunto de teste. |
| **Técnica** | Prever o tamanho da resposta, classificar se o modelo barato dá conta, avaliar com LLM como juiz e retreinar com os vereditos. |
| **Família de algoritmos** | Aprendizado supervisionado: **regressão** (Gradient Boosting e quantílica) e **classificação** (regressão logística). **LLM como juiz**, **aprendizado contínuo** e **Monte Carlo**. |
| **Token de IA** | Cada rodada gasta tokens reais do Gemini (duas respostas e a juíza por pedido), informados pela própria API. |
| **É válido?** | As duas respostas são geradas para todos os pedidos, então as estratégias são comparadas nos mesmos pedidos. A juíza recebe as respostas em ordem sorteada, o corte do roteador é escolhido com validação cruzada e uma pessoa avalia uma amostra para medir a concordância com a juíza. |
| **Resultado explicado** | Abas "Resultado" e "Rodadas" do app. |

## Como o resultado é medido

Nos mesmos pedidos, quatro estratégias:

| Estratégia | O que mostra |
|---|---|
| Sempre Flash | referência de qualidade e de custo máximo |
| Sempre Lite | o mais barato, e quanto de qualidade isso custa |
| **Piloto automático** | a decisão do roteador |
| Escolha perfeita | o teto: quanto daria para economizar sabendo as respostas de antemão |

Metas definidas antes da primeira rodada: economia de pelo menos 30% sobre sempre Flash, qualidade em pelo menos 90% dos pedidos, qualidade acima de sempre Lite e concordância da juíza com uma pessoa em pelo menos 80% dos casos.

## Estrutura

| Arquivo | O que é |
|---|---|
| `rodar_piloto.py` | A rodada diária: decide, gera, julga, grava e retreina |
| `core/piloto.py` | Roteador (regra inicial e regressão logística), retreino e avaliação das quatro estratégias |
| `core/ia.py` | Chamadas ao Gemini e a IA juíza |
| `core/modelo.py` | Previsor de tokens da resposta |
| `app.py` | Painel em Streamlit, só leitura: não chama o Gemini |
| `.github/workflows/piloto.yml` | Agendamento diário e commit automático do resultado |
| `validacao_humana.py` | Sorteia 30 pares para uma pessoa avaliar |
| `preparar_dados.py` | Gera a fila de pedidos e os tokens por idioma |
| `atividade_tokens_ia.ipynb` | Análise do previsor de tokens |
| `data/execucoes/` | Uma planilha por rodada e o `roteador.json` |

## Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py          # painel (não precisa de chave)
.venv/bin/python rodar_piloto.py --n 5  # uma rodada pequena (precisa de GEMINI_API_KEY)
```

A rodada lê a configuração das variáveis de ambiente ou de `.streamlit/secrets.toml` (fora do git):

```toml
GEMINI_API_KEY = "sua-chave"
# opcionais
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL_LITE = ""          # vazio: escolhe um Flash-Lite que a chave tenha
PRECO_FLASH_ENTRADA = 0.30      # US$ por 1M tokens
PRECO_FLASH_SAIDA = 2.50
PRECO_LITE_ENTRADA = 0.10
PRECO_LITE_SAIDA = 0.40
```

Opções da rodada: `--n` (pedidos, padrão 50), `--orcamento` (US$ que o piloto pode gastar no dia, padrão 0,50), `--meta` (qualidade mínima do roteador, padrão 0,90), `--pausa` (segundos entre chamadas, padrão 4).

## Automação no GitHub

1. Em **Settings → Secrets and variables → Actions**, crie o secret `GEMINI_API_KEY`.
2. Opcional, em **Variables**: `GEMINI_MODEL`, `GEMINI_MODEL_LITE`, `PILOTO_PEDIDOS`, `PILOTO_ORCAMENTO`.
3. A rodada roda todo dia às 8h (Brasília). Para rodar na hora: **Actions → Piloto automático → Run workflow**.

## Validação humana da juíza

Depois de 30 pedidos avaliados, rode `python validacao_humana.py`. Abra `data/validacao_humana.csv` e preencha `sua_escolha` com `A`, `B` ou `empate`, sem olhar a coluna `gabarito_a`. Faça commit do arquivo: o app calcula a concordância com a juíza.

## Publicar no Streamlit Community Cloud

Em [share.streamlit.io](https://share.streamlit.io), **Create app** → este repositório, branch `main`, arquivo `app.py`. O painel não precisa de secrets e se atualiza a cada commit de rodada.
