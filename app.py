import shutil
from pathlib import Path

import gradio as gr

from src.rag import CHROMA_DIR, DADOS_DIR, carregar_banco, consultar, indexar

db = None

EXEMPLOS = [
    "Qual é a receita total de janeiro?",
    "Qual vendedor tem a melhor margem?",
    "Quais são os produtos mais lucrativos?",
    "Como está o desempenho do Nordeste comparado ao Sudeste?",
    "Quais anomalias foram detectadas nos dados?",
    "Quais são os principais insights estratégicos?",
    "Qual produto tem maior giro de estoque?",
]


def _banco_existe() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.glob("*.sqlite3"))


def inicializar() -> str:
    global db
    if _banco_existe():
        db = carregar_banco()
        n = db._collection.count()
        return f"Base vetorial carregada: {n} chunks indexados."
    db, n = indexar()
    return f"Base indexada com sucesso: {n} chunks gerados."


def responder(
    mensagem: str,
    historico: list[dict],
) -> tuple[list[dict], list[dict], str, str]:
    global db

    if not mensagem or not mensagem.strip():
        return historico, historico, "", _renderizar_fontes([])

    if db is None:
        novo = historico + [
            {"role": "user", "content": mensagem},
            {"role": "assistant", "content": "Base não inicializada. Recarregue a página."},
        ]
        return novo, novo, "", ""

    # Converte formato Gradio 6 (dicts) para tuplas aceitas pelo consultar()
    hist_tuples = []
    msgs = [m for m in historico]
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            hist_tuples.append((msgs[i]["content"], msgs[i + 1]["content"]))

    resposta, chunks = consultar(mensagem, db, hist_tuples)

    novo = historico + [
        {"role": "user", "content": mensagem},
        {"role": "assistant", "content": resposta},
    ]
    return novo, novo, "", _renderizar_fontes(chunks)


def _renderizar_fontes(chunks: list) -> str:
    if not chunks:
        return "_Nenhum trecho recuperado ainda. Faça uma pergunta._"
    linhas = []
    for i, chunk in enumerate(chunks, 1):
        fonte = Path(chunk.metadata.get("source", "?")).name
        preview = chunk.page_content.strip().replace("\n", " ")
        linhas.append(f"**Trecho {i}** | `{fonte}`\n\n> {preview}")
    return "\n\n---\n\n".join(linhas)


def reindexar() -> str:
    global db
    db, n = indexar()
    docs = len(list(DADOS_DIR.glob("*.txt")) + list(DADOS_DIR.glob("*.pdf")))
    return f"Reindexado: {n} chunks de {docs} documento(s)."


def adicionar_documento(arquivo) -> str:
    if arquivo is None:
        return "Nenhum arquivo selecionado."
    origem = Path(arquivo)
    if origem.suffix.lower() not in (".txt", ".pdf"):
        return "Formato não suportado. Envie .txt ou .pdf."
    destino = DADOS_DIR / origem.name
    shutil.copy(str(origem), destino)
    return f"'{destino.name}' adicionado. Clique em **Reindexar** para incluí-lo."


CSS = """
.fonte-panel { background: var(--background-fill-secondary); border-radius: 8px; padding: 12px; }
.status-ok { color: var(--color-green-500); }
"""

with gr.Blocks(title="RAG Assistente de Vendas") as demo:
    gr.Markdown(
        "# `RAG Assistente de Análise de Vendas`\n"
        "Assistente com recuperação semântica de documentos. LangChain · ChromaDB · Ollama (llama3.2 + nomic-embed-text)"
    )

    with gr.Row():
        status_box = gr.Textbox(
            label="Status da base vetorial",
            interactive=False,
            scale=4,
        )
        reindexar_btn = gr.Button("Reindexar", variant="secondary", scale=1)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversa",
                height=460,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Digite sua pergunta sobre os dados de vendas…",
                    label="",
                    scale=5,
                    container=False,
                    autofocus=True,
                )
                enviar_btn = gr.Button("Enviar", variant="primary", scale=1)

            with gr.Row():
                limpar_btn = gr.Button("Limpar conversa", variant="secondary", scale=1)

            gr.Examples(examples=EXEMPLOS, inputs=msg_input, label="Exemplos de perguntas")

        with gr.Column(scale=2):
            gr.Markdown("### Trechos recuperados")
            gr.Markdown(
                "_Os trechos do documento usados como contexto para gerar cada resposta aparecem aqui._",
                elem_classes=["fonte-panel"],
            )
            fontes_md = gr.Markdown(
                "_Faça uma pergunta para ver os trechos._",
                elem_classes=["fonte-panel"],
            )

            with gr.Accordion("Adicionar documento", open=False):
                gr.Markdown("Adicione arquivos `.txt` ou `.pdf` à base. Após o upload, clique em **Reindexar**.")
                upload = gr.File(
                    label="Documento",
                    file_types=[".txt", ".pdf"],
                    type="filepath",
                )
                upload_status = gr.Textbox(label="", interactive=False)

    historico_state = gr.State([])

    # Eventos
    msg_input.submit(
        responder,
        inputs=[msg_input, historico_state],
        outputs=[chatbot, historico_state, msg_input, fontes_md],
    )
    enviar_btn.click(
        responder,
        inputs=[msg_input, historico_state],
        outputs=[chatbot, historico_state, msg_input, fontes_md],
    )
    limpar_btn.click(
        lambda: ([], [], "", "_Conversa limpa._"),  # type: ignore[return-value]
        outputs=[chatbot, historico_state, msg_input, fontes_md],
    )
    reindexar_btn.click(reindexar, outputs=status_box)
    upload.change(adicionar_documento, inputs=upload, outputs=upload_status)

    demo.load(inicializar, outputs=status_box)


if __name__ == "__main__":
    demo.launch(css=CSS)
