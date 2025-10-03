#!/usr/bin/env python3
"""
PDF Processor - Modulo para processamento de documentos PDF
Extrai texto e cria chunks semanticos para indexacao vetorial
"""

import os
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processador de documentos PDF com chunking inteligente"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa o processador de PDF
        
        Args:
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposicao entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configurar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"PDFProcessor inicializado - chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    
    def extract_text_pymupdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extrai texto usando PyMuPDF (melhor para texto simples)
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Dicionario com numero da pagina e texto
        """
        text_by_page = {}
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_by_page[page_num + 1] = text
            
            doc.close()
            logger.info(f"PyMuPDF extraiu texto de {len(text_by_page)} paginas")
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto com PyMuPDF: {str(e)}")
        
        return text_by_page
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Dict[int, str]:
        """
        Extrai texto usando pdfplumber (melhor para tabelas e layout complexo)
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Dicionario com numero da pagina e texto
        """
        text_by_page = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        text_by_page[page_num + 1] = text
                        
                        # Extrair tabelas se existirem
                        tables = page.extract_tables()
                        if tables:
                            table_text = self._format_tables(tables)
                            text_by_page[page_num + 1] += f"\n\nTabelas:\n{table_text}"
            
            logger.info(f"pdfplumber extraiu texto de {len(text_by_page)} paginas")
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto com pdfplumber: {str(e)}")
        
        return text_by_page
    
    def _format_tables(self, tables: List[List[List[str]]]) -> str:
        """
        Formata tabelas extraidas em texto legivel
        
        Args:
            tables: Lista de tabelas extraidas
            
        Returns:
            Texto formatado das tabelas
        """
        formatted_text = ""
        
        for table_idx, table in enumerate(tables):
            formatted_text += f"\nTabela {table_idx + 1}:\n"
            
            for row in table:
                if row:  # Verificar se a linha nao esta vazia
                    row_text = " | ".join([cell or "" for cell in row])
                    formatted_text += f"{row_text}\n"
            
            formatted_text += "\n"
        
        return formatted_text
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza o texto extraido
        
        Args:
            text: Texto bruto extraido
            
        Returns:
            Texto limpo e normalizado
        """
        if not text:
            return ""
        
        # Remover quebras de linha excessivas
        text = text.replace('\n\n\n', '\n\n')
        text = text.replace('\r\n', '\n')
        
        # Remover espacos excessivos
        text = ' '.join(text.split())
        
        # Remover caracteres especiais problematicos
        text = text.replace('\x00', '')
        text = text.replace('\ufeff', '')  # BOM
        
        # Normalizar aspas e apostrofes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Divide texto em chunks semanticos
        
        Args:
            text: Texto para dividir
            metadata: Metadados do documento
            
        Returns:
            Lista de documentos LangChain
        """
        if not text.strip():
            return []
        
        # Limpar texto
        cleaned_text = self.clean_text(text)
        
        # Criar chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Converter em documentos LangChain
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Apenas chunks com conteudo
                doc_metadata = metadata.copy()
                doc_metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
        
        return documents
    
    def process_pdf(self, pdf_path: str, filename: str = None) -> List[Document]:
        """
        Processa um arquivo PDF completo
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            filename: Nome do arquivo (opcional)
            
        Returns:
            Lista de documentos processados
        """
        if not os.path.exists(pdf_path):
            logger.error(f"Arquivo nao encontrado: {pdf_path}")
            return []
        
        if not filename:
            filename = os.path.basename(pdf_path)
        
        logger.info(f"Processando PDF: {filename}")
        
        # Tentar pdfplumber primeiro (melhor para layout complexo)
        text_by_page = self.extract_text_pdfplumber(pdf_path)
        
        # Se falhar, tentar PyMuPDF
        if not text_by_page:
            logger.info("pdfplumber falhou, tentando PyMuPDF...")
            text_by_page = self.extract_text_pymupdf(pdf_path)
        
        if not text_by_page:
            logger.error(f"Nao foi possivel extrair texto de: {filename}")
            return []
        
        # Processar paginas
        all_documents = []
        
        for page_num, page_text in text_by_page.items():
            metadata = {
                'source': filename,
                'page': page_num,
                'file_path': pdf_path,
                'processing_method': 'hybrid'
            }
            
            # Criar chunks para a pagina
            page_documents = self.create_chunks(page_text, metadata)
            all_documents.extend(page_documents)
        
        logger.info(f"PDF processado: {filename} - {len(all_documents)} chunks de {len(text_by_page)} paginas")
        
        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Obtem estatisticas dos documentos processados
        
        Args:
            documents: Lista de documentos
            
        Returns:
            Dicionario com estatisticas
        """
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Paginas unicas
        pages = set()
        sources = set()
        
        for doc in documents:
            if 'page' in doc.metadata:
                pages.add(doc.metadata['page'])
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        
        return {
            'total_chunks': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chunk_size': total_chars // len(documents) if documents else 0,
            'unique_pages': len(pages),
            'unique_sources': len(sources),
            'sources': list(sources)
        }

# Funcoes auxiliares para uso direto
def process_single_pdf(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
    """
    Funcao utilitaria para processar um unico PDF
    
    Args:
        pdf_path: Caminho para o PDF
        chunk_size: Tamanho dos chunks
        
    Returns:
        Lista de documentos processados
    """
    processor = PDFProcessor(chunk_size=chunk_size)
    return processor.process_pdf(pdf_path)

def batch_process_pdfs(pdf_paths: List[str], chunk_size: int = 1000) -> List[Document]:
    """
    Funcao utilitaria para processar multiplos PDFs
    
    Args:
        pdf_paths: Lista de caminhos para PDFs
        chunk_size: Tamanho dos chunks
        
    Returns:
        Lista combinada de documentos processados
    """
    processor = PDFProcessor(chunk_size=chunk_size)
    all_documents = []
    
    for pdf_path in pdf_paths:
        documents = processor.process_pdf(pdf_path)
        all_documents.extend(documents)
    
    return all_documents