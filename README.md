# rag_assistente

Assistente de perguntas e respostas sobre documentos usando RAG (Retrieval-Augmented Generation) com LLM local via Ollama.

## O que faz

Carrega um arquivo de texto, divide em chunks, gera embeddings vetoriais e armazena no ChromaDB. Ao receber uma pergunta, busca os trechos mais relevantes do documento e usa o LLM para gerar uma resposta fundamentada — sem alucinação, sem API paga, 100% local.

## Arquitetura

```
Documento (.txt)
    └── RecursiveCharacterTextSplitter → chunks
        └── nomic-embed-text (Ollama) → vetores
            └── ChromaDB → banco vetorial

Pergunta do usuário
    └── nomic-embed-text → vetor da pergunta
        └── ChromaDB → chunks mais similares
            └── llama3.2 (Ollama) → resposta final
```

## Stack

- **LangChain** — orquestração do pipeline RAG
- **ChromaDB** — banco de dados vetorial local
- **Ollama** — execução de LLMs localmente
- **nomic-embed-text** — modelo de embeddings
- **llama3.2** — modelo de linguagem para geração de respostas

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando

## Instalação

```bash
# Clone o repositório
git clone https://github.com/Arthur-Baptista-dos-Santos/rag_assistente.git
cd rag_assistente

# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt

# Baixe os modelos necessários
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Uso

```bash
# Garanta que o Ollama está rodando
ollama serve

# Execute o assistente
python src/rag.py
```

Digite perguntas sobre o documento. Digite `sair` para encerrar.

## Exemplo

```
Você: Qual foi a receita total em janeiro?
Assistente: A receita total em janeiro de 2024 foi de R$ 196.520.

Você: Quem foi o vendedor com maior margem?
Assistente: O vendedor com maior margem foi João, com 44,75%.
```

## Estrutura

```
rag_assistente/
├── dados/
│   └── vendas.txt        # documento de exemplo
├── src/
│   └── rag.py            # pipeline RAG completo
├── .gitignore
├── requirements.txt
└── README.md
```

## Conceitos aplicados

- **RAG**: recupera contexto real antes de gerar resposta, eliminando alucinações
- **Embeddings**: representação vetorial de texto para busca por similaridade semântica
- **Chunking**: divisão do documento em pedaços com overlap para preservar contexto
- **LCEL**: LangChain Expression Language — pipeline declarativo com operador `|`
- **Vector store**: banco otimizado para busca por similaridade (ChromaDB)
