# TokenScope: quanto custa a IA da sua aplicação?

Atividade de IA aplicada e IA experimental (Prof. Wagner, 3º ciclo).

Uma aplicação com IA generativa paga por **token**. Ela conhece a pergunta que envia, mas **não sabe o tamanho da resposta**, que é a parte mais cara. O **TokenScope** é um assistente com IA que mostra o custo de cada resposta:

1. o visitante **conversa com o Gemini** (a chave fica só no servidor);
2. antes de cada resposta, um modelo treinado com 18.924 conversas reais **prevê** quantos tokens a IA vai gerar, com uma faixa de 80%;
3. depois, o app mostra os **tokens reais e o custo** informados pela API, e um **painel** acumula o consumo de todos os visitantes e mede o acerto da previsão;
4. a aba **O algoritmo** mostra os dados, a validação e a comparação com o chute ingênuo.

## Roteiro da atividade

| Etapa | No projeto |
|---|---|
| **Entender o problema** | Planejar o custo de usar IA dentro de uma aplicação sem saber o tamanho das respostas. |
| **Base de dados** | [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache 2.0): ~88 mil mensagens reais, 18.924 pares prompt → resposta. |
| **Técnica** | Inferir os tokens da resposta, comparar com um baseline e avaliar o impacto no custo. |
| **Família de algoritmos** | Aprendizado supervisionado, **regressão**: Gradient Boosting, regressão quantílica (faixa de 80%), regressão linear e baseline. **Monte Carlo** para o custo mensal. |
| **Token de IA** | O assistente usa a API do Gemini. Cada resposta gasta tokens reais, que são medidos, somados no painel e comparados com a previsão. |
| **É válido?** | Treino/teste 75/25, comparação com baseline, cobertura da faixa de 80% e validação ao vivo contra o Gemini. |
| **Resultado explicado** | Aba "O algoritmo" do app e seção 8 do notebook. |

## Resultados

| Modelo | R² (log) | Erro médio por resposta | Total previsto / real |
|---|---|---|---|
| Baseline (mediana) | -0,03 | 124 tokens | 0,997 |
| Regressão linear | 0,12 | 120 tokens | 1,038 |
| Gradient Boosting | 0,15 | 117 tokens | 1,022 |

- **Uma resposta isolada é difícil de prever** (R² ≈ 0,15): o tamanho depende do conteúdo, não só do tamanho da pergunta.
- **O custo do mês é previsível**: para 10 mil requisições, a estimativa fica a ~2% do real. A média histórica (baseline) chega no mesmo total, então o que importa para o orçamento é medir uma amostra real de uso.
- **A saída domina o custo**: em mediana, a resposta tem ~10 vezes mais tokens que o prompt, e o token de saída é mais caro.
- **O idioma muda o custo**: português gasta ~7% mais tokens por caractere que inglês.

## Estrutura

| Arquivo | O que é |
|---|---|
| `app.py` | Assistente, painel de consumo e abas do algoritmo (Streamlit) |
| `core/modelo.py` | Treino e previsão do modelo, custo e economia |
| `core/ia.py` | Chamadas ao Gemini e calibração de tokens |
| `atividade_tokens_ia.ipynb` | Análise completa, executada, com gráficos e explicação |
| `preparar_dados.py` | Gera os arquivos auxiliares de `data/` |
| `data/` | Pares processados, tokens por idioma e amostra de calibração |
| `figures/` | Gráficos do notebook |

## Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

A configuração fica em `.streamlit/secrets.toml` (fora do git) ou nos Secrets do Streamlit Cloud:

```toml
GEMINI_API_KEY = "sua-chave"
# opcionais (valores padrão abaixo)
GEMINI_MODEL = "gemini-2.5-flash"
PRECO_ENTRADA = 0.30   # US$ por 1M tokens
PRECO_SAIDA = 2.50     # US$ por 1M tokens
LIMITE_SESSAO = 15     # mensagens por visitante
LIMITE_DIA = 300       # mensagens por dia, somando todos
```

Para abrir o notebook: `.venv/bin/pip install ipykernel` e abra `atividade_tokens_ia.ipynb` no VS Code ou no Jupyter.

## Publicar no Streamlit Community Cloud

1. Em [share.streamlit.io](https://share.streamlit.io), entre com o GitHub e clique em **Create app**.
2. Escolha este repositório, a branch `main` e o arquivo `app.py`.
3. Em **Advanced settings → Secrets**, cole `GEMINI_API_KEY = "sua-chave"`.
4. Clique em **Deploy**.

Os limites por visitante e por dia protegem a cota da chave. O painel guarda o uso enquanto o servidor estiver no ar (o Streamlit Cloud reinicia o app depois de um tempo sem visitas).

![Distribuição de tokens](figures/01_distribuicao_tokens.png)
![Custo mensal](figures/04_custo_mensal.png)
