#!/usr/bin/env python3
"""
Aplicacao Streamlit para chat com PDFs usando RAG.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import streamlit as st
import yaml
from dotenv import load_dotenv

from chat_interface import ChatInterface
from pdf_processor import PDFProcessor
from vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if os.getenv("OPENAI_CHAT_MODEL"):
        config["chat"]["model"] = os.getenv("OPENAI_CHAT_MODEL")

    if os.getenv("OPENAI_EMBEDDING_MODEL"):
        config["embeddings"]["model"] = os.getenv("OPENAI_EMBEDDING_MODEL")

    return config


def save_uploaded_files(uploaded_files) -> List[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            saved_paths.append(Path(tmp_file.name))

    return saved_paths


def process_documents(saved_paths: List[Path], config: Dict[str, Any]) -> Dict[str, Any]:
    pdf_config = config["pdf_processing"]
    processor = PDFProcessor(
        chunk_size=pdf_config["chunk_size"],
        chunk_overlap=pdf_config["chunk_overlap"],
    )

    all_documents = []
    for file_path in saved_paths:
        all_documents.extend(processor.process_pdf(str(file_path), filename=file_path.name))

    if not all_documents:
        raise ValueError("Nao foi possivel extrair conteudo dos PDFs enviados.")

    embeddings_config = config["embeddings"]
    use_openai_embeddings = embeddings_config["provider"].lower() == "openai"

    if use_openai_embeddings and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "O provider de embeddings esta configurado como openai, mas OPENAI_API_KEY nao foi definida."
        )
    embedding_model = (
        embeddings_config["model"] if use_openai_embeddings else embeddings_config["local_model"]
    )

    vector_store = VectorStore(
        embedding_model=embedding_model,
        use_openai=use_openai_embeddings,
        dimension=embeddings_config["dimension"],
    )
    vector_store.create_index(all_documents)

    chat_config = config["chat"]
    chat_interface = ChatInterface(
        vector_store=vector_store,
        model_name=chat_config["model"],
        temperature=chat_config["temperature"],
        max_tokens=chat_config["max_tokens"],
        memory_window=chat_config["memory_window"],
    )

    return {
        "documents": all_documents,
        "vector_store": vector_store,
        "chat_interface": chat_interface,
        "stats": processor.get_document_stats(all_documents),
    }


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("Fontes utilizadas", expanded=False):
        for source in sources:
            st.markdown(
                f"**{source['filename']}** | pagina **{source['page']}** | score **{source['score']:.3f}**"
            )
            st.caption(source["content"])


def main() -> None:
    load_dotenv()
    config = load_config()

    st.set_page_config(
        page_title=config["interface"]["page_title"],
        page_icon="📚",
        layout=config["interface"]["layout"],
        initial_sidebar_state=config["interface"]["sidebar_state"],
    )

    st.title("Chatbot baseado em conteudo de PDFs")
    st.caption("Upload, indexacao vetorial e perguntas com citacao de fontes.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False

    with st.sidebar:
        st.header("Configuracao")
        st.write(f"Embeddings: **{config['embeddings']['provider']}**")
        st.write(f"Modelo de chat: **{config['chat']['model']}**")

        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY encontrada.")
        else:
            st.warning("OPENAI_API_KEY ausente. O app funcionara em modo extrativo.")

        uploaded_files = st.file_uploader(
            "Envie um ou mais PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("Processar PDFs", type="primary", use_container_width=True)
        clear_clicked = st.button("Limpar conversa", use_container_width=True)

        if clear_clicked and st.session_state.get("rag_ready"):
            st.session_state.chat_interface.clear_memory()
            st.session_state.chat_messages = []
            st.success("Historico limpo.")

    if process_clicked:
        if not uploaded_files:
            st.error("Envie pelo menos um PDF antes de processar.")
        else:
            try:
                with st.spinner("Processando documentos e criando indice vetorial..."):
                    saved_paths = save_uploaded_files(uploaded_files)
                    result = process_documents(saved_paths, config)

                    st.session_state.documents = result["documents"]
                    st.session_state.vector_store = result["vector_store"]
                    st.session_state.chat_interface = result["chat_interface"]
                    st.session_state.stats = result["stats"]
                    st.session_state.rag_ready = True
                    st.session_state.chat_messages = []

                st.success("Documentos processados com sucesso.")
            except Exception as exc:
                st.session_state.rag_ready = False
                st.error(f"Falha ao processar os PDFs: {exc}")

    if st.session_state.get("rag_ready"):
        stats = st.session_state["stats"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks", stats["total_chunks"])
        col2.metric("Paginas unicas", stats["unique_pages"])
        col3.metric("Arquivos", stats["unique_sources"])

        st.subheader("Perguntas sugeridas")
        suggestions = st.session_state.chat_interface.suggest_questions()
        st.write(" | ".join(suggestions[:4]))

        st.subheader("Conversa")
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_sources(message.get("sources", []))

        user_query = st.chat_input("Pergunte algo sobre os PDFs carregados")
        if user_query:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Consultando documentos..."):
                        response = st.session_state.chat_interface.get_response(user_query)
                    st.markdown(response["answer"])
                    render_sources(response.get("sources", []))
                except Exception as exc:
                    response = {"answer": f"Erro ao responder a pergunta: {exc}", "sources": []}
                    st.error(response["answer"])

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                }
            )
    else:
        st.info("Envie PDFs na barra lateral e clique em 'Processar PDFs' para iniciar.")


if __name__ == "__main__":
    main()
