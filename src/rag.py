import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

DADOS_DIR = Path("dados")
CHROMA_DIR = Path("dados/chroma")
MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_LLM = "llama3.2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 4

PROMPT_TEMPLATE = """\
Você é um assistente especialista em análise de dados de vendas.
Responda em português usando APENAS as informações fornecidas no contexto abaixo.
Se a informação não estiver no contexto, diga claramente que não encontrou a informação.
Seja direto e preciso. Cite números quando disponíveis no contexto.

{historico_bloco}Contexto recuperado dos documentos:
{contexto}

Pergunta: {pergunta}

Resposta:"""


def _formatar_historico(historico: list[tuple[str, str]]) -> str:
    if not historico:
        return ""
    linhas = ["Histórico da conversa (últimas trocas):"]
    for user, assistente in historico[-3:]:
        linhas.append(f"Usuário: {user}")
        linhas.append(f"Assistente: {assistente}")
    return "\n".join(linhas) + "\n\n"


def _carregar_documentos() -> list:
    docs = []
    for arquivo in DADOS_DIR.iterdir():
        if not arquivo.is_file():
            continue
        if arquivo.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(arquivo), encoding="utf-8").load())
        elif arquivo.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(arquivo)).load())
    return docs


def indexar() -> tuple[Chroma, int]:
    """Carrega documentos de DADOS_DIR, gera chunks e persiste no ChromaDB."""
    documentos = _carregar_documentos()
    if not documentos:
        print("Nenhum documento encontrado em dados/")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    print(f"{len(documentos)} documento(s) → {len(chunks)} chunks gerados")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DIR))
    return db, len(chunks)


def carregar_banco() -> Chroma:
    """Carrega banco ChromaDB existente do disco sem reindexar."""
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def consultar(
    pergunta: str,
    db: Chroma,
    historico: list[tuple[str, str]] | None = None,
) -> tuple[str, list]:
    """Executa busca vetorial + geração. Retorna (resposta, chunks_recuperados)."""
    historico = historico or []

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    chunks = retriever.invoke(pergunta)

    contexto = "\n\n---\n\n".join(c.page_content for c in chunks)
    hist_bloco = _formatar_historico(historico)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model=MODELO_LLM)
    chain = prompt | llm | StrOutputParser()

    resposta = chain.invoke({
        "historico_bloco": hist_bloco,
        "contexto": contexto,
        "pergunta": pergunta,
    })

    return resposta, chunks
