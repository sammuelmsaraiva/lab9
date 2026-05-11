# Lab 8 - Alinhamento Humano com DPO

Pipeline completo de alinhamento de um LLM utilizando Direct Preference Optimization (DPO), substituindo o complexo pipeline de RLHF. O objetivo é garantir que o modelo seja Útil, Honesto e Inofensivo (HHH — Helpful, Honest, Harmless), # Lab 9 - Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

Pipeline de Retrieval-Augmented Generation (RAG) de nível de produção para busca em manuais médicos. O sistema traduz queries coloquiais de pacientes para jargão técnico via HyDE, busca rapidamente via índice HNSW e refina os resultados com um Cross-Encoder.

## Pré-requisitos

```
pip install faiss-cpu sentence-transformers scikit-learn openai
```

## Estrutura do Projeto

```
lab9/
├── lab9.py      # Pipeline completo: HNSW + HyDE + Cross-Encoder
└── README.md
```

## Como Executar

```bash
# Com OpenAI (HyDE via GPT-3.5):
export OPENAI_API_KEY="sua_chave_aqui"
python3 lab9.py

# Sem OpenAI (fallback local automático):
python3 lab9.py
```

O script detecta automaticamente a disponibilidade dos modelos e usa fallbacks locais quando necessário, permitindo testar o pipeline completo sem dependências externas.

## Arquitetura do Pipeline

```
Query coloquial do usuário
         │
    [PASSO 2] HyDE (Hypothetical Document Embedding)
         │   LLM alucina um documento técnico hipotético
         │   O vetor desse documento serve de âncora geométrica
         ▼
    [PASSO 3] Busca Bi-Encoder via HNSW
         │   Top-10 candidatos recuperados (funil largo)
         │   Busca aproximada em O(log N) ao invés de O(N)
         ▼
    [PASSO 4] Re-ranking Cross-Encoder
         │   Cada par (query, doc) avaliado com atenção cruzada
         │   Top-3 documentos finais selecionados
         ▼
    Contexto injetado no LLM gerador
```

## HNSW: M e ef_construction vs KNN Exato

O **KNN exato** compara a query contra todos os N vetores do índice, com complexidade O(N × d) e consumo de memória O(N × d × 4 bytes). Para 1 milhão de vetores de dimensão 768 em float32, isso representa ~3 GB só para os vetores, mais o custo de varredura linear a cada consulta.

O **HNSW** (Hierarchical Navigable Small World) organiza os vetores em um grafo multicamada onde cada nó conecta-se a no máximo **M** vizinhos por camada. O parâmetro **M** controla diretamente o consumo de memória: cada aresta ocupa ~8 bytes, então o overhead do grafo é aproximadamente `N × M × 8 bytes` por nível (em média 2 níveis). Para M=32 e 1 milhão de vetores isso adiciona ~512 MB de overhead de grafo, mas reduz a busca para O(log N) consultas, tornando a latência de busca constante independente do tamanho do índice.

O parâmetro **ef_construction** define o tamanho da lista de candidatos durante a construção do grafo: valores maiores produzem um grafo de maior qualidade (recall mais alto) ao custo de mais tempo de indexação e RAM temporária durante a build. Em produção, M=32 e ef_construction=200 são valores padrão que equilibram recall (~99%) e latência de busca abaixo de 1ms para bases de milhões de vetores, o que seria impossível com KNN exato.suprimindo respostas tóxicas ou maliciosas.

## Pré-requisitos

```
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

> **Hardware recomendado:** Google Colab Pro com GPU A100 (40 GB VRAM).

## Estrutura do Projeto

```
lab8/
├── lab8.py              # Pipeline completo DPO (4 passos)
├── dataset_hhh.jsonl    # Dataset de preferências HHH (31 pares)
└── README.md
```

## Como Executar

```bash
huggingface-cli login
python3 lab8.py
```

## Dataset de Preferências (Passo 1)

O arquivo `dataset_hhh.jsonl` contém 31 pares no formato obrigatório com as chaves `prompt`, `chosen` e `rejected`, cobrindo categorias de segurança como:

- Ataques a sistemas e bancos de dados
- Violação de privacidade e interceptação de comunicações
- Geração de malware e ransomware
- Fraude financeira e lavagem de dinheiro
- Discurso de ódio e incitação à violência
- Desinformação e deepfakes
- Manipulação psicológica e assédio

## Configurações Implementadas (Passos 2 e 3)

| Parâmetro           | Valor             |
|---------------------|-------------------|
| beta (DPO)          | 0.1               |
| LoRA rank           | 64                |
| LoRA alpha          | 16                |
| LoRA dropout        | 0.1               |
| otimizador          | paged_adamw_32bit |
| lr_scheduler        | cosine            |
| warmup_ratio        | 0.03              |
| quantização         | 4-bit NF4         |

## O Papel Matemático do Parâmetro Beta (β)

O DPO otimiza diretamente a política do modelo sem um modelo de recompensa explícito. A função objetivo é:

```
L_DPO(π_θ) = -E[ log σ( β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

onde `y_w` é a resposta escolhida (*chosen*), `y_l` é a resposta rejeitada (*rejected*) e `π_ref` é o modelo de referência congelado.

O **β funciona como um imposto de regularização** sobre a divergência de Kullback-Leibler (KL) entre o modelo em treinamento (ator) e o modelo de referência. Matematicamente, ele controla o trade-off entre duas forças opostas: maximizar a probabilidade das respostas *chosen* em relação às *rejected* e manter a distribuição do modelo próxima à distribuição original pré-treinada.

Um **β próximo de zero** torna o modelo agressivo na otimização de preferências, podendo colapsar a fluência e diversidade da linguagem — o modelo "esquece" como escrever de forma natural ao focar apenas em preferir *chosen* sobre *rejected*. Um **β muito alto** mantém o modelo quase idêntico à referência, tornando o alinhamento ineficaz. O valor **β = 0.1** é o padrão da literatura (paper original DPO, Rafailov et al., 2023) por equilibrar alinhamento efetivo com preservação da fluência e capacidade geral do modelo base.