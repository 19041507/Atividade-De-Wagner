# TokenScope: piloto automático de custo de IA

Atividade de IA aplicada e IA experimental (Prof. Wagner, 3º ciclo).

Uma aplicação com IA generativa paga por **token**, e o modelo mais capaz custa várias vezes mais que o mais simples. Mandar tudo para o modelo caro desperdiça dinheiro; mandar tudo para o barato derruba a qualidade.

O **TokenScope** é um assistente em que essa escolha é feita por uma IA, sozinha, em cada pergunta:

1. **Decide:** antes de chamar o Gemini, um roteador escolhe o **Flash-Lite** (barato) ou o **Flash** (mais capaz). Só o escolhido responde, então a economia é real.
2. **Audita:** em 1 de cada 5 perguntas, depois que o usuário já recebeu a resposta, o outro modelo também responde e uma **IA juíza** (o Gemini Pro, maior e diferente dos dois modelos avaliados) compara as duas às cegas. O usuário também avalia com 👍/👎.
3. **Testa:** todo dia, o GitHub Actions roda um **teste controlado** com pedidos reais do OpenAssistant nos dois modelos e na juíza, comparando todas as estratégias nos mesmos pedidos.
4. **Aprende:** o roteador se retreina com os vereditos do teste e do uso real.
5. **Relata:** a IA escreve o diário do dia e, se a qualidade cair abaixo da meta, o workflow abre uma issue sozinho.

## Roteiro da atividade

| Etapa | No projeto |
|---|---|
| **Entender o problema** | Gastar menos com IA sem piorar as respostas, decidindo o modelo de cada pedido. |
| **Base de dados** | [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache 2.0) para treinar o previsor e para o teste controlado; perguntas reais do assistente, guardadas no Supabase. |
| **Técnica** | Prever o tamanho da resposta, classificar se o modelo barato dá conta, auditar com LLM como juiz e retreinar com os vereditos. |
| **Família de algoritmos** | Aprendizado supervisionado: **regressão** (Gradient Boosting e quantílica) e **classificação** (regressão logística). **LLM como juiz**, **aprendizado contínuo**, **bootstrap** e **Monte Carlo**. |
| **Token de IA** | Cada pergunta, auditoria e rodada gasta tokens reais do Gemini, informados pela própria API. |
| **É válido?** | Teste controlado com as duas respostas em todos os pedidos; juíza às cegas e com ordem sorteada; corte do roteador por validação cruzada; intervalos de confiança; comparação com uma regra fixa; concordância da juíza com o 👍/👎 dos usuários. |
| **Resultado explicado** | Telas "Resultados" e "Bastidores" do app. |

Metas definidas antes da primeira rodada: economia de pelo menos 30% sobre sempre Flash, qualidade em pelo menos 90% dos pedidos, qualidade acima de usar sempre o Lite, roteador aprendido melhor que a regra fixa e concordância da juíza com os usuários em pelo menos 80%.

## Estrutura

| Arquivo | O que é |
|---|---|
| `app.py` | Assistente, resultados e bastidores (Streamlit) |
| `core/producao.py` | O piloto no assistente: decisão, registro, auditoria em segundo plano, diário |
| `core/piloto.py` | Roteador, retreino, avaliação das estratégias, intervalos de confiança e projeção |
| `core/ia.py` | Chamadas ao Gemini e a IA juíza |
| `core/banco.py` | Leitura e gravação do uso real no Supabase |
| `core/modelo.py` | Previsor de tokens da resposta |
| `rodar_piloto.py` | Teste controlado diário + retreino + diário + alerta |
| `.github/workflows/piloto.yml` | Agendamento diário, commit do resultado e issue de alerta |
| `preparar_dados.py` | Gera a fila de pedidos e os tokens por idioma |
| `atividade_tokens_ia.ipynb` | Análise do previsor de tokens |
| `data/execucoes/` | Uma planilha e um diário por rodada, e o `roteador.json` |

## Configuração

### Supabase (uso real)
Crie um projeto e rode no SQL Editor:

```sql
create table pedidos (
  id uuid primary key default gen_random_uuid(),
  criado_em timestamptz not null default now(),
  sessao text, pedido text not null, lang text,
  rota text not null check (rota in ('lite', 'flash')), motivo text, forcado boolean not null default false,
  prob_lite_ok real, versao_roteador int, previsto_p10 real, previsto_mediana real, previsto_p90 real, x jsonb,
  modelo text, entrada int, saida int, raciocinio int, custo real, custo_flash_estimado real, resposta text,
  feedback smallint check (feedback in (-1, 1)),
  auditado boolean not null default false, modelo_alt text, resposta_alt text, custo_alt real,
  nota_lite smallint, nota_flash smallint, vencedor text check (vencedor in ('lite', 'flash', 'empate')),
  lite_ok boolean, motivo_juiza text, custo_juiza real
);
create index on pedidos (criado_em);
alter table pedidos enable row level security;
grant select, insert, update on pedidos to service_role;
```

### Secrets
No Streamlit (**Settings → Secrets**, formato TOML, com aspas) e no GitHub (**Settings → Secrets → Actions**, sem aspas):

```toml
GEMINI_API_KEY = "sua-chave"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "sb_secret_..."
# opcionais
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL_LITE = ""        # vazio: escolhe um Flash-Lite que a chave tenha
GEMINI_MODEL_JUIZA = ""       # vazio: escolhe o Gemini Pro mais recente da chave (sem Pro, o Flash julga)
PRECO_FLASH_ENTRADA = 0.30    # US$ por 1M tokens
PRECO_FLASH_SAIDA = 2.50
PRECO_LITE_ENTRADA = 0.10
PRECO_LITE_SAIDA = 0.40
PRECO_JUIZA_ENTRADA = 1.25
PRECO_JUIZA_SAIDA = 10.00
LIMITE_SESSAO = 20            # perguntas por visitante
LIMITE_DIA = 300              # perguntas por dia, somando todos
ORCAMENTO_DIA = 1.00          # US$: passou disso, só o Lite até o dia seguinte
```

## Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
.venv/bin/python rodar_piloto.py --n 5   # uma rodada pequena do teste controlado
```

Opções da rodada: `--n` (pedidos, padrão 50), `--orcamento` (US$ da rodada, padrão 0,50), `--meta` (qualidade mínima, padrão 0,90), `--pausa` (segundos entre chamadas, padrão 4), `--forcar` (outra rodada no mesmo dia).

## Automação no GitHub

A rodada roda todo dia às 8h (Brasília). Para rodar na hora: **Actions → Piloto automático → Run workflow** (com "Fazer outra rodada" marcado, dá para rodar várias no mesmo dia). Variáveis opcionais em **Settings → Variables**: `GEMINI_MODEL`, `GEMINI_MODEL_LITE`, `GEMINI_MODEL_JUIZA`, `PILOTO_PEDIDOS`, `PILOTO_ORCAMENTO`.

## Publicar no Streamlit Community Cloud

Em [share.streamlit.io](https://share.streamlit.io), **Create app** → este repositório, branch `main`, arquivo `app.py`, e os secrets acima.
