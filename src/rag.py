import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuração
DOCUMENTO = Path("dados/vendas.txt")
CHROMA_DIR = Path("dados/chroma")
MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_LLM = "llama3.2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

PROMPT_TEMPLATE = (
    "Você é um assistente de análise de vendas. "
    "Use apenas o contexto abaixo para responder. "
    "Se a resposta não estiver no contexto, diga que não encontrou a informação.\n\n"
    "Contexto:\n{contexto}\n\n"
    "Pergunta: {pergunta}\n\n"
    "Resposta:"
)


def carregar_documento(caminho: Path) -> list:
    if not caminho.exists():
        print(f"Erro: arquivo '{caminho}' não encontrado.")
        sys.exit(1)
    loader = TextLoader(str(caminho), encoding="utf-8")
    return loader.load()


def criar_chunks(documentos: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)
    print(f"{len(chunks)} chunks criados")
    return chunks


def criar_banco_vetorial(chunks: list) -> Chroma:
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def criar_chain(db: Chroma):
    llm = OllamaLLM(model=MODELO_LLM)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return (
        {"contexto": db.as_retriever(), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    print("Carregando documento...")
    documentos = carregar_documento(DOCUMENTO)

    print("Dividindo em chunks...")
    chunks = criar_chunks(documentos)

    print("Gerando embeddings e salvando no ChromaDB...")
    db = criar_banco_vetorial(chunks)

    print("Conectando ao LLM...")
    chain = criar_chain(db)

    print("\nAssistente pronto. Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() == "sair":
            break
        try:
            resposta = chain.invoke(pergunta)
            print(f"\nAssistente: {resposta}\n")
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}")
            print("Verifique se o Ollama está rodando: ollama serve\n")


if __name__ == "__main__":
    main()
