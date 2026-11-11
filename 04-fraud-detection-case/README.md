# Case de Prevenção a Fraudes: Classificação de Transações

**Autora:** Betsabet Sulamita Roque Sullca
**Vaga:** Pessoa Cientista de Dados (Vaga afirmativa para mulheres cis e trans)
**Entrega:** 03/11/2026
**Ferramentas:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter / Google Colab

---

## Sumário

1. [Resumo em 1 minuto](#1-resumo-em-1-minuto)
2. [O case](#2-o-case)
3. [Estrutura do projeto](#3-estrutura-do-projeto)
4. [Como rodar no seu computador](#4-como-rodar-no-seu-computador)
5. [Os dados](#5-os-dados)
6. [Passo a passo do notebook](#6-passo-a-passo-do-notebook)
7. [Decisões importantes e por quê](#7-decisões-importantes-e-por-quê)
8. [Resultados](#8-resultados)
9. [Insights e recomendações para o negócio](#9-insights-e-recomendações-para-o-negócio)
10. [Limitações](#10-limitações)
11. [Próximos passos](#11-próximos-passos)
12. [Índice das figuras](#12-índice-das-figuras)
13. [Glossário](#13-glossário)
14. [Perguntas prováveis na apresentação](#14-perguntas-prováveis-na-apresentação)
15. [Histórico do projeto](#15-histórico-do-projeto)

---

## 1. Resumo em 1 minuto

| Item | Valor |
|---|---|
| Transações de 2025 (com rótulo de fraude) | 523.525 |
| Taxa de fraude em 2025 | 3,9% |
| Transações de janeiro/2026 (sem rótulo) | 44.451 |
| Modelo escolhido | Random Forest |
| Average Precision no teste (nov–dez/2025) | **0,63** (aleatório = 0,05) |
| AUC-ROC no teste | **0,96** |
| Threshold (corte) escolhido | 0,2 |
| Fraudes detectadas no teste | **77%** (3.094 de 3.991) |
| Alertas que são fraude de verdade | ~50% |
| Transações que recebem alerta | **7,1%** |
| Transações de jan/2026 marcadas como fraude | **3.431 (7,7%)** |

**Em uma frase:** a fraude é rara, mas tem horário (22h–23h), perfil (aparelho novo, posto de combustível, alimentação) e comportamento (valores médios, parecidos com o gasto normal do cliente). O score interno atual não enxerga nada disso, e um Random Forest simples pega 3 de cada 4 fraudes revisando só 7% das transações.

---

## 2. O case

A área de Prevenção a Fraudes de uma instituição financeira avalia transações suspeitas de forma, em grande parte, manual. Com o aumento do volume, esse processo ficou ineficiente e pouco escalável, o que gera dois riscos:

- **Perda financeira:** fraudes que não são detectadas a tempo.
- **Experiência ruim:** clientes legítimos que têm a transação bloqueada por engano.

**Desafio:** explorar os dados, construir um modelo que classifique transações como "fraudulentas" ou "não fraudulentas" e apresentar os resultados de forma clara.

**Entregáveis pedidos:**
- um Jupyter Notebook com o desenvolvimento;
- um .zip com a base de janeiro/2026 classificada (CSV) e a apresentação em PDF.

**Regras do e-mail do processo:**
- todas as respostas em inglês;
- resolver da forma mais simples possível;
- nome dos arquivos no padrão `[Nome] - [Carreira]`;
- envio por Google Forms até 03/11/2026, 23h59.

O enunciado original está em [`docs/case_description.pdf`](docs/case_description.pdf).

---

## 3. Estrutura do projeto

```
fraud-detection-case/
│
├── data/
│   ├── processed/                    ← dados gerados pelo notebook
│   │   ├── .gitkeep
│   │   └── transactions_jan2026_classified.csv
│   └── raw/                          ← dados originais, nunca são alterados
│       ├── .gitkeep
│       ├── transacoes_historico_2025.csv.gz   (2025, com flag_fraude; compactado)
│       └── transacoes_jan2026.csv             (jan/2026, sem rótulo)
│
├── docs/
│   ├── case_description.pdf          ← enunciado original do case
│   └── presentation.pdf              ← apresentação entregue (13 slides)
│
├── images/                           ← os 18 gráficos do notebook em PNG
│
├── notebooks/
│   ├── fraud_detection_case.ipynb        ← notebook principal (versão local, já rodado)
│   └── fraud_detection_case_colab.ipynb  ← versão entregue no formulário (Google Colab)
│
├── src/
│   └── .gitkeep                      ← reservado para scripts Python reutilizáveis
│
├── .gitignore                        ← o que NÃO subir para o GitHub (dados)
├── README.md                         ← este documento
└── requirements.txt                  ← bibliotecas Python necessárias
```

**Por que separar assim?**

- **`data/raw/`**: os dados originais ficam intocados. Se algo der errado, dá para recomeçar do zero.
- **`data/processed/`**: tudo o que o código gera fica separado do original.
- **`docs/`**: os documentos (enunciado e apresentação em PDF).
- **`images/`**: os gráficos, prontos para usar no README, em slides ou no LinkedIn.
- **`notebooks/`**: o código e a análise.
- **`src/`**: lugar para funções reaproveitáveis (ex.: a `fraud_rate_chart`), se o projeto crescer.
- **`.gitkeep`**: um arquivo vazio que serve só para o Git guardar a pasta mesmo quando ela está sem dados.

**Diferenças entre as duas versões do notebook:**

| | `fraud_detection_case.ipynb` | `fraud_detection_case_colab.ipynb` |
|---|---|---|
| Onde roda | No seu computador | No Google Colab |
| Leitura dos dados | Caminho relativo (`../data/raw/`) | Google Drive (`drive.mount`) |
| Onde salva o CSV | `../data/processed/` | Pasta do Drive |
| Conteúdo e resultados | Iguais | Iguais |

> O arquivo de 2025 está compactado (`.csv.gz`, 12 MB em vez de 70 MB). O pandas lê direto, não precisa descompactar. Se quiser o CSV original, está no seu Google Drive em `testes/ATF-Data_Science/`.

---

## 4. Como rodar no seu computador

### 4.1 Pré-requisitos

- **Python 3.10 ou mais recente**: [python.org/downloads](https://www.python.org/downloads/). Na instalação do Windows, marque **"Add Python to PATH"**.
- Ou, se preferir, o **VS Code** com a extensão **Jupyter**.

### 4.2 Instalação (uma vez só)

Abra o **Prompt de Comando** ou o **PowerShell** e rode:

```bash
cd %USERPROFILE%\Documents\fraud-detection-case

# 1. cria um ambiente virtual (uma "caixinha" só para este projeto)
python -m venv .venv

# 2. ativa o ambiente
.venv\Scripts\activate

# 3. instala as bibliotecas
pip install -r requirements.txt
```

> No PowerShell, se aparecer erro de permissão ao ativar, rode antes:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 4.3 Rodar o notebook

```bash
cd %USERPROFILE%\Documents\fraud-detection-case
.venv\Scripts\activate
jupyter notebook notebooks/fraud_detection_case.ipynb
```

No Jupyter, use **Kernel → Restart & Run All**.

- **Tempo:** de 3 a 15 minutos, dependendo do computador. A seção 7.7 é a mais lenta, porque treina mais 4 modelos.
- **Resultado:** o arquivo `data/processed/transactions_jan2026_classified.csv` é gerado de novo.

**No VS Code:** abra a pasta do projeto, abra o notebook, escolha o kernel `.venv` no canto superior direito e clique em **Run All**.

### 4.4 Conferindo se rodou certo

Os números abaixo devem aparecer no notebook:

| Onde | Valor esperado |
|---|---|
| Seção 5 | Train: 436.034 linhas · Test: 87.491 linhas |
| Seção 7.1 | Random Forest: AP 0,632 · AUC 0,964 |
| Seção 7.6 | Frauds detected: 3094 of 3991 · alert 7.1% |
| Seção 9 | Transactions with alert: 3431 · 7.7% |

> O Gradient Boosting pode dar 0,624 ou 0,626 dependendo da versão das bibliotecas. É normal, e o texto ("about 0.62") cobre os dois.

---

## 5. Os dados

### 5.1 Visão geral

| | 2025 (histórico) | Jan/2026 |
|---|---|---|
| Arquivo | `transacoes_historico_2025.csv.gz` | `transacoes_jan2026.csv` |
| Linhas | 523.525 | 44.451 |
| Colunas | 15 | 14 (sem `flag_fraude`) |
| Período | 01/01/2025 00:00 a 31/12/2025 23:59 | 01/01/2026 a 31/01/2026 |
| Valores vazios | 0 | 0 |
| Linhas duplicadas | 0 | 0 |
| Fraudes | 20.485 (3,9%) | desconhecido (é o que o modelo prevê) |

### 5.2 Dicionário de dados

| Coluna | Tipo | Descrição | Valores / faixa | Usada no modelo? |
|---|---|---|---|---|
| `datahora_transacao` | data/hora | Momento da transação | 1 transação por minuto | Indiretamente (vira `hour` e `weekday`) |
| `nome_estabelecimento` | texto | Nome da loja | 10 lojas (ex.: Gas Station, Pharmacy Z, Online Shop 1) | Não (redundante com o tipo) |
| `tipo_estabelecimento` | texto | Categoria da loja | E-commerce, Food & Beverage, Fuel, Pharmacy, Retail, Supermarket | Sim |
| `valor_transacao` | número | Valor da compra (assumido em R$) | 0,50 a 722,90 · mediana 25,25 | Sim |
| `nome_cliente` | texto | Identificador do cliente | 26.214 clientes | Não (é um ID), mas usado para criar o histórico |
| `cidade` | texto | Cidade da transação | BH, Curitiba, Porto Alegre, Recife, Rio, Salvador, São Paulo | Sim |
| `flag_fraude` | booleano | **Alvo**: a transação é fraude? | True / False | É o que o modelo aprende |
| `device_type` | texto | Tipo de aparelho | desktop, mobile, tablet | Sim |
| `is_new_device` | booleano | Aparelho nunca usado antes pelo cliente? | True / False | Sim |
| `temperatura_celsius` | número | Temperatura no momento | −12,7 a 46,1 | Sim (para provar que não ajuda) |
| `codigo_postal` | texto | CEP | quase todos únicos (521.905 em 2025) | Não (sem padrão) |
| `score_interno` | inteiro | Score de risco atual da empresa | 0 a 1.000 | Sim (para comparar) |
| `turno` | texto | Período do dia | madrugada (0–5h), manha (6–11h), tarde (12–17h), noite (18–23h) | Sim |
| `valor_categoria` | texto | Faixa de valor | baixo (até 30), medio (30,01–100), alto (acima de 100) | Sim |
| `cidade_regiao` | texto | Região da cidade | Nordeste, Sudeste, Sul | Não (redundante com a cidade) |

### 5.3 Colunas criadas (feature engineering)

| Coluna | Como é calculada | Por quê |
|---|---|---|
| `hour` | Hora da transação (0–23) | O turno mistura horas de risco muito diferente |
| `weekday` | Dia da semana (0 = segunda … 6 = domingo) | Testar se o dia importa (não importou) |
| `n_previous_transactions` | Quantas compras o cliente fez **antes** desta | Clientes novos podem ter risco diferente |
| `previous_avg_amount` | Média das compras **anteriores** do cliente | Base para comparar o valor atual |
| `amount_to_avg_ratio` | Valor atual ÷ média anterior (1 = normal) | Mede se a compra foge do padrão do cliente |

> As colunas de histórico usam **só o passado** de cada transação: `cumcount` e `cumsum` menos o valor atual. Assim o modelo não "vê o futuro" (ver *data leakage* no glossário).

### 5.4 Observação sobre os dados

Alguns sinais indicam que a base é **sintética** (gerada para o case):
- exatamente uma transação por minuto;
- horas com 0% de fraude;
- score distribuído de forma uniforme;
- CEP praticamente único por linha.

Isso não muda a análise, mas vale comentar se perguntarem: padrões tão "limpos" provavelmente seriam mais suaves em dados reais.

---

## 6. Passo a passo do notebook

### Seção 0: Setup
Importa as bibliotecas e carrega as duas bases uma única vez. As outras seções reaproveitam `data2025` e `data2026`.

### Seção 1: Entendendo o problema
- É **classificação binária supervisionada**: o modelo aprende com exemplos já rotulados (fraude ou não).
- Existem dois tipos de erro:
  - **falso negativo:** deixar passar uma fraude, o que gera perda financeira;
  - **falso positivo:** bloquear um cliente legítimo, o que gera experiência ruim.
- Como a fraude é rara, **acurácia não serve**: um modelo que diz "nunca é fraude" acertaria 96%. Por isso o foco é em **precision** e **recall**.

### Seção 2: Qualidade dos dados
| Subseção | O que faz | Resultado |
|---|---|---|
| 2.1 | Colunas, tipos, nulos, duplicados | Nenhum nulo nem duplicado; a única diferença entre as bases é `flag_fraude` |
| 2.2 | Converte `datahora_transacao` para data | 2025 cobre o ano inteiro e jan/2026, o mês inteiro |
| 2.3 | `describe()` e histograma do valor | O valor é assimétrico: média 43,96, mediana 25,25, desvio 74,40, máximo 722,90 |
| 2.4 | Compara categorias e médias entre 2025 e jan/2026 | Nenhuma categoria nova (exceto CEP); médias quase iguais; todos os clientes de janeiro já existem em 2025 |

### Seção 3: Análise exploratória
| Subseção | Pergunta | Resposta |
|---|---|---|
| 3.1 | Quanto é fraude? | 3,9%: base desbalanceada |
| 3.2 | Muda ao longo do ano? | Estável (3,7–4,0%) de jan a nov; **dezembro sobe para 5,4%** (+44%) |
| 3.3 | Fraude tem valor alto? | **Não.** Medianas parecidas (25 x 27); fraudes ficam entre 19 e 38, e nenhuma passa de 219 |
| 3.4 | O score interno separa fraude? | **Não.** Medianas 500 x 501, distribuições iguais |
| 3.5 | Onde a fraude se concentra? | Noite 10,2% x manhã 0,7%; aparelho novo 12,9% x conhecido 2,1%; Fuel 7,5% e Food & Beverage 6,6% x Pharmacy 2,3%; Sudeste 4,9% x Nordeste 2,6%; valor médio 4,6% x alto 0,3% |

A função `fraud_rate_chart(coluna)` foi criada para não repetir código: ela calcula a taxa de fraude por categoria, desenha o gráfico com a linha da média geral e imprime a tabela.

### Seção 4: Feature engineering
| Subseção | O que faz | Resultado |
|---|---|---|
| 4.1 | Cria `hour` e `weekday` | Pico de **~30% às 22h e 23h**, pico menor de ~7% entre 12h e 14h, zero fraude entre 9h–11h e 15h–19h. O dia da semana não faz diferença |
| 4.2 | Cria o histórico do cliente (só com o passado) | A fraude é **maior quando o valor está perto da média do cliente** (5,5% na faixa do meio x 2,2% nos extremos): o fraudador imita o comportamento normal |
| 4.3 | Escolhe as 13 features | 7 numéricas + 6 categóricas |

**Features do modelo (13):**
- **Numéricas (7):** `valor_transacao`, `score_interno`, `temperatura_celsius`, `hour`, `weekday`, `n_previous_transactions`, `amount_to_avg_ratio`
- **Categóricas (6):** `tipo_estabelecimento`, `cidade`, `device_type`, `is_new_device`, `turno`, `valor_categoria`

**Ficaram de fora:**
- `codigo_postal`: não tem padrão;
- `nome_cliente`: é um identificador;
- `nome_estabelecimento` e `cidade_regiao`: são redundantes;
- `month`: para o modelo não "decorar" o calendário;
- colunas auxiliares (`previous_avg_amount`, `source`).

### Seção 5: Separação treino / teste
- **Treino:** janeiro a outubro de 2025, com 436.034 transações e 3,8% de fraude.
- **Teste:** novembro e dezembro de 2025, com 87.491 transações e 4,6% de fraude.
- A separação é **temporal**: treina no passado e testa no "futuro". É assim que o modelo seria usado na vida real, e o teste ainda inclui o dezembro atípico.

### Seção 6: Modelagem
Antes de treinar, `pd.get_dummies` transforma as colunas de texto em colunas de 0 e 1, o que dá 32 colunas no total. O `reindex` garante que o teste tenha as mesmas colunas do treino.

| Modelo | Como funciona | Configuração |
|---|---|---|
| 6.1 Regressão Logística | Dá um peso para cada feature e transforma em probabilidade. É a **linha de base** | `max_iter=1000`, `class_weight='balanced'`; números padronizados com `StandardScaler` (ajustado só no treino) |
| 6.2 Random Forest | 100 árvores de decisão, cada uma treinada numa parte diferente dos dados, que depois "votam" | `n_estimators=100`, `min_samples_leaf=20`, `random_state=42` |
| 6.3 Gradient Boosting | Árvores em sequência, cada uma corrigindo os erros da anterior | `HistGradientBoostingClassifier(random_state=42)` |

### Seção 7: Avaliação
| Subseção | O que faz |
|---|---|
| 7.1 | Compara Average Precision e AUC dos 3 modelos + score interno |
| 7.2 | Curva Precision-Recall |
| 7.3 | Curva ROC |
| 7.4 | Escolha do modelo (Random Forest) |
| 7.5 | Testa thresholds de 0,1 a 0,5 e escolhe 0,2 |
| 7.6 | Matriz de confusão com threshold 0,2 |
| 7.7 | Testa 3 variações para ver se dá para melhorar (não deu) |

### Seção 8: Interpretação do modelo
Calcula a importância de cada feature no Random Forest, somando as colunas 0/1 de volta na feature original (ex.: `turno_noite` + `turno_tarde` + … = `turno`).

### Seção 9: Classificação de janeiro/2026
1. Prepara todo o 2025 e jan/2026 do mesmo jeito (`get_dummies` + `reindex`).
2. **Retreina** o Random Forest com os mesmos parâmetros, mas com **o ano inteiro de 2025**, incluindo nov–dez, os meses mais recentes.
3. Calcula a probabilidade de fraude de cada transação de janeiro e aplica o corte de 0,2.
4. Salva o CSV com as 14 colunas originais + `fraud_probability` (0 a 1) + `predicted_fraud` (True/False).

### Seção 10: Conclusões e próximos passos
Resume os achados, o resultado do modelo, as recomendações, as limitações e os próximos passos (detalhados nas seções 8 a 11 deste README).

---

## 7. Decisões importantes e por quê

| # | Decisão | Alternativa | Por que escolhi assim |
|---|---|---|---|
| 1 | **Average Precision** como métrica principal | Acurácia, AUC-ROC | Com 3,9% de fraude, a acurácia engana (96% sem fazer nada) e a AUC tende a ser otimista. A AP foca em achar a classe rara |
| 2 | **Split temporal** (jan–out x nov–dez) | Split aleatório | Simula o uso real (prever o futuro) e evita o modelo "ver" transações do futuro |
| 3 | Histórico do cliente **só com o passado** | Média do cliente no ano todo | Usar o ano todo seria *data leakage*: o modelo veria informação que não existe na hora da compra |
| 4 | `StandardScaler` ajustado **só no treino** | Ajustar na base toda | Mesmo motivo: não vazar informação do teste |
| 5 | `pd.get_dummies` | `OneHotEncoder` com `Pipeline` | Mais simples e fácil de explicar, como pedia o case |
| 6 | **Random Forest** como modelo final | Gradient Boosting | Resultado igual ou um pouco melhor (0,632 x ~0,625) e mais fácil de explicar |
| 7 | `min_samples_leaf=20` | Padrão (1) | Evita folhas com poucos exemplos, ou seja, evita *overfitting* |
| 8 | `class_weight='balanced'` na Regressão Logística | Sem peso | Faz o modelo dar importância à classe rara |
| 9 | **Threshold 0,2** | 0,5 (padrão), 0,1 | Equilíbrio: 78% de recall com ~50% de precisão e 7% de alertas. O 0,1 pega mais (87%), mas só 40% dos alertas seriam fraude |
| 10 | **Retreinar com 2025 inteiro** para classificar janeiro | Usar o modelo treinado só até outubro | Nov–dez são os meses mais próximos de janeiro, e dezembro teve padrão diferente |
| 11 | Manter `score_interno` e `temperatura` no modelo | Tirar logo de início | Para **provar**, na importância, que não ajudam |
| 12 | Assumir valores em **R$** | — | Todas as cidades e CEPs são brasileiros; a base não informa a moeda (registrado no notebook) |

---

## 8. Resultados

### 8.1 Comparação dos modelos (teste: nov–dez/2025)

| Modelo | Average Precision | AUC-ROC |
|---|---|---|
| Aleatório (referência) | 0,046 | 0,50 |
| Score interno (atual) | 0,045 | 0,502 |
| Regressão Logística | 0,247 | 0,894 |
| **Random Forest** ✅ | **0,632** | **0,964** |
| Gradient Boosting | 0,626 | 0,958 |

**Como ler:** o Random Forest é **~14 vezes melhor que o aleatório** (0,632 ÷ 0,046). O score interno tem o mesmo desempenho do aleatório.

### 8.2 Escolha do threshold (Random Forest)

| Threshold | Precisão (alertas certos) | Recall (fraudes pegas) |
|---|---|---|
| 0,1 | 40% | 87% |
| **0,2** ✅ | **50%** | **78%** |
| 0,3 | 61% | 60% |
| 0,4 | 70% | 50% |
| 0,5 | 74% | 42% |

Quanto **menor** o corte, mais fraudes o modelo pega, mas mais clientes bons ele bloqueia. É uma decisão de negócio.

### 8.3 Matriz de confusão (threshold 0,2)

| | Previsto: não fraude | Previsto: fraude |
|---|---|---|
| **Real: não fraude** | 80.349 ✅ (verdadeiro negativo) | 3.151 ❌ (falso positivo) |
| **Real: fraude** | 897 ❌ (falso negativo) | 3.094 ✅ (verdadeiro positivo) |

- **Recall** = 3.094 ÷ 3.991 = **77,5%**
- **Precisão** = 3.094 ÷ 6.245 alertas = **49,5%**
- **Alertas** = 6.245 ÷ 87.491 = **7,1%** das transações

> Na tabela de thresholds aparece 78% e na matriz 77%: é o mesmo 77,5% arredondado de formas diferentes.

### 8.4 O modelo dá para melhorar? (seção 7.7)

| Variação | Average Precision |
|---|---|
| Random Forest escolhido | 0,632 |
| Sem as features fracas | 0,633 |
| Mais árvores (300) e folhas menores (5) | 0,618 |
| Gradient Boosting ajustado | 0,630–0,631 |

**Conclusão:** nenhuma variação melhorou. O limite está na **informação dos dados**, não no algoritmo. Para melhorar, seriam necessários dados novos (histórico de aparelho, localização etc.).

### 8.5 Importância das features

| Feature | Importância |
|---|---|
| `hour` | **0,398** (40%) |
| `is_new_device` | 0,136 |
| `tipo_estabelecimento` | 0,128 |
| `turno` | 0,085 |
| `cidade` | 0,077 |
| `valor_transacao` | 0,061 |
| `amount_to_avg_ratio` | 0,027 |
| `temperatura_celsius` | 0,022 |
| `score_interno` | 0,017 |
| `device_type` | 0,016 |
| `valor_categoria` | 0,012 |
| `n_previous_transactions` | 0,011 |
| `weekday` | 0,008 |

A ordem confirma a análise exploratória: a hora, o aparelho novo e o tipo de loja dominam; o score e o dia da semana quase não contam.

### 8.6 Janeiro/2026

- **3.431 transações marcadas como fraude (7,7%)**, perto dos 7,1% do período de teste, o que indica um comportamento consistente do modelo.
- Arquivo: `data/processed/transactions_jan2026_classified.csv` (44.451 linhas, 16 colunas).

---

## 9. Insights e recomendações para o negócio

**O perfil da fraude:**
1. **Horário:** 22h–23h concentram ~30% de fraude, 7 vezes a média.
2. **Aparelho novo:** 6 vezes mais risco que um aparelho conhecido.
3. **Tipo de loja:** combustível (7,5%) e alimentação (6,6%) têm cerca de 3 vezes o risco de farmácia (2,3%).
4. **Valor:** o fraudador evita valores altos e fica perto do gasto normal do cliente, para não chamar atenção.
5. **Sazonalidade:** dezembro teve 44% mais fraude que os outros meses.

**Recomendações:**
- **Usar o modelo com faixas de risco:**
  - probabilidade **alta**: bloqueio automático;
  - probabilidade **média**: revisão manual;
  - probabilidade **baixa**: aprovação automática.

  Assim a equipe revisa só os casos duvidosos, o que resolve o problema de escala.
- **Monitorar mais** o horário das 22h–23h e as transações com aparelho novo.
- **Revisar o score interno** com a equipe responsável, porque hoje ele não separa fraude de não fraude.

---

## 10. Limitações

- O threshold foi escolhido olhando o próprio conjunto de teste, então os resultados podem estar um pouco otimistas.
- O rótulo de fraude foi considerado confiável. Vale confirmar com a equipe **como e quando** uma transação é marcada como fraude.
- O padrão por hora é muito "limpo" e deve ser acompanhado com dados novos.
- O modelo final foi retreinado com 2025 inteiro, então os números da seção 7 são uma **estimativa** do desempenho dele.
- A base parece sintética (ver seção 5.4).

---

## 11. Próximos passos

1. **Retreinar** o modelo periodicamente (ex.: todo mês) com dados recentes.
2. **Definir o threshold pelo custo real** de cada erro (quanto custa uma fraude x um cliente bloqueado).
3. **Monitorar** o desempenho mês a mês, porque o padrão de fraude muda.
4. **Buscar dados novos**, como histórico de aparelho e localização, que é o que pode realmente melhorar o modelo.
5. *(Estudo)* Separar um período de **validação** para escolher o threshold sem usar o teste.

---

## 12. Índice das figuras

Todas estão em [`images/`](images/).

| # | Arquivo | O que mostra | Seção |
|---|---|---|---|
| 01 | `01_distribution_of_transaction_amounts.png` | Histograma do valor: maioria abaixo de 45, cauda longa até 723 | 2.3 |
| 02 | `02_share_of_fraudulent_transactions_2025.png` | 96,1% não fraude x 3,9% fraude | 3.1 |
| 03 | `03_fraud_rate_by_month_2025.png` | Taxa estável e salto em dezembro (5,4%) | 3.2 |
| 04 | `04_transaction_amount_fraud_vs_not_fraud.png` | Boxplot do valor (escala log): fraude concentrada entre 19 e 38 | 3.3 |
| 05 | `05_internal_score_fraud_vs_not_fraud.png` | Boxplot do score: distribuições iguais | 3.4 |
| 06 | `06_fraud_rate_by_turno.png` | Noite 10,2% x manhã 0,7% | 3.5 |
| 07 | `07_fraud_rate_by_is_new_device.png` | Aparelho novo 12,9% x conhecido 2,1% | 3.5 |
| 08 | `08_fraud_rate_by_device_type.png` | Mobile 4,9%, desktop 2,8%, tablet 2,1% | 3.5 |
| 09 | `09_fraud_rate_by_tipo_estabelecimento.png` | Fuel 7,5% e Food & Beverage 6,6% no topo | 3.5 |
| 10 | `10_fraud_rate_by_cidade_regiao.png` | Sudeste 4,9%, Sul 3,7%, Nordeste 2,6% | 3.5 |
| 11 | `11_fraud_rate_by_valor_categoria.png` | Médio 4,6%, baixo 3,9%, alto 0,3% | 3.5 |
| 12 | `12_fraud_rate_by_hour_2025.png` | **O gráfico principal:** pico de 30% às 22h–23h | 4.1 |
| 13 | `13_fraud_rate_by_weekday.png` | Dia da semana: sem diferença | 4.1 |
| 14 | `14_model_comparison.png` | Average Precision dos 4 "modelos" | 7.1 |
| 15 | `15_precision_recall_curve.png` | RF e GB sobrepostos no topo; score na linha do aleatório | 7.2 |
| 16 | `16_roc_curve.png` | AUC: RF 0,96 … score 0,50 | 7.3 |
| 17 | `17_confusion_matrix_random_forest_threshold_0_2.png` | 3.094 fraudes pegas, 897 perdidas, 3.151 falsos alertas | 7.6 |
| 18 | `18_feature_importance_random_forest.png` | A hora domina com 40% | 8 |

---

## 13. Glossário

| Termo | Explicação simples |
|---|---|
| **Classificação binária** | Prever uma entre duas respostas (fraude / não fraude) |
| **Supervisionado** | O modelo aprende com exemplos que já têm a resposta certa (`flag_fraude`) |
| **Base desbalanceada** | Uma classe é muito mais rara que a outra (3,9% de fraude) |
| **Acurácia** | % de acertos no total. Engana em bases desbalanceadas |
| **Precisão (precision)** | Dos alertas, quantos eram fraude de verdade |
| **Recall** | Das fraudes que existiam, quantas o modelo pegou |
| **Curva Precision-Recall** | Mostra a troca entre precisão e recall para todos os thresholds |
| **Average Precision (AP)** | Resume a curva Precision-Recall num número. Aleatório = taxa de fraude (~0,05); perfeito = 1 |
| **AUC-ROC** | Chance de o modelo dar nota maior para uma fraude do que para uma não fraude. 0,5 = aleatório, 1 = perfeito |
| **Threshold (corte)** | Probabilidade a partir da qual o modelo diz "é fraude" |
| **Matriz de confusão** | Tabela com acertos e erros: VP, VN, FP, FN |
| **Falso positivo** | Alerta em transação legítima (cliente bom bloqueado) |
| **Falso negativo** | Fraude que passou sem alerta |
| **Feature** | Uma coluna usada pelo modelo para aprender |
| **Feature engineering** | Criar colunas novas a partir das existentes (ex.: `hour`) |
| **Data leakage (vazamento)** | Quando o modelo usa, sem querer, informação que não existiria no momento da previsão. Deixa os resultados falsamente bons |
| **Overfitting** | O modelo "decora" o treino e vai mal em dados novos |
| **Split temporal** | Treinar com o passado e testar com o futuro |
| **`get_dummies` (one-hot)** | Transforma uma coluna de texto em várias colunas de 0 e 1 |
| **Padronização (`StandardScaler`)** | Coloca os números na mesma escala (média 0, desvio 1). Necessária para a Regressão Logística |
| **`class_weight='balanced'`** | Dá mais peso à classe rara durante o treino |
| **Regressão Logística** | Modelo linear que dá um peso para cada feature e transforma a soma em probabilidade |
| **Árvore de decisão** | Sequência de perguntas do tipo "hora ≥ 22?" que leva a uma resposta |
| **Random Forest** | Muitas árvores treinadas em partes diferentes dos dados, que votam juntas |
| **Gradient Boosting** | Árvores criadas em sequência, cada uma corrigindo os erros da anterior |
| **Importância de feature** | Quanto cada feature ajudou o Random Forest a separar fraude de não fraude |
| **Mediana** | Valor do meio quando os dados estão em ordem. Não é afetada por extremos |
| **Desvio padrão** | Quanto os valores se espalham em torno da média |
| **Boxplot** | Gráfico com mediana, quartis (25% e 75%) e extremos |
| **Escala log** | Eixo em que cada marca multiplica o valor (1, 10, 100…). Ajuda a ver dados com valores muito diferentes |

---

## 14. Perguntas prováveis na apresentação

**"Por que não usou acurácia?"**
Porque com 3,9% de fraude, um modelo que nunca alerta teria 96% de acurácia e seria inútil. Usei Average Precision, que foca na classe rara.

**"Por que Random Forest e não Gradient Boosting?"**
Os dois empataram (0,63 x 0,62). Escolhi o Random Forest porque teve o melhor resultado e é mais simples de explicar.

**"Como você evitou vazamento de dados?"**
Com três cuidados:
- split temporal (treino no passado, teste no futuro);
- histórico do cliente calculado só com as compras anteriores;
- padronização ajustada só no treino.

**"Por que threshold 0,2?"**
É o equilíbrio entre pegar fraude (78%) e não incomodar clientes (só 7% de alertas, metade deles corretos). O ideal seria definir pelo custo real de cada erro.

**"O que faria com mais tempo ou dados?"**
- Validar o threshold num período separado.
- Buscar dados de aparelho e localização: testei variações do modelo e nenhuma melhorou, então o ganho tem que vir de informação nova.
- Monitorar o modelo mês a mês.

**"E o score interno?"**
Ele tem o mesmo desempenho do aleatório (AP 0,05, AUC 0,50). Recomendo entender com a equipe o que ele mede.

**"O que o dezembro atípico muda?"**
O teste já inclui dezembro, então o modelo foi avaliado num mês "difícil". Além disso, o modelo final foi retreinado com nov–dez para ficar mais próximo de janeiro.

---

## 15. Histórico do projeto

| Data | Etapa |
|---|---|
| 31/10/2026 | Recebimento do case; estudo guiado começa (qualidade dos dados, exploração) |
| 01–02/11/2026 | Feature engineering, modelagem (LR, RF, GB), avaliação e threshold |
| 03/11/2026 | Revisão dos textos, versão em inglês, seção 7.7, apresentação (13 slides), validação final e entrega pelo Google Forms |
| 04/11/2026 | Preparação dos arquivos para o portfólio no GitHub |
| 10/11/2026 | Organização local do projeto (esta pasta) |

**Arquivos relacionados no Google Drive** (`testes/ATF-Data_Science/`):
- `ATF_Case_v2.ipynb`: versão de trabalho em português;
- `Betsabet Sulamita Roque Sullca - Data Scientist (FINAL v2).ipynb`: versão final em inglês, sem os resultados;
- os CSVs originais.

---

> **Antes de publicar no GitHub:** o `.gitignore` já impede que `data/raw/`, `data/processed/` e os CSVs subam. Os dados são da empresa do processo seletivo, então confirme que o case pode ser público antes de deixar o repositório aberto.
