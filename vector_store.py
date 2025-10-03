#!/usr/bin/env python3
"""
Vector Store - Modulo para gerenciamento de embeddings e busca vetorial
Utiliza FAISS para indexacao e busca eficiente de similaridade
"""

import os
import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    """Gerenciador de embeddings e busca vetorial com FAISS"""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-ada-002",
                 use_openai: bool = True,
                 dimension: int = 1536):
        """
        Inicializa o vector store
        
        Args:
            embedding_model: Nome do modelo de embedding
            use_openai: Se deve usar OpenAI ou modelo local
            dimension: Dimensao dos embeddings
        """
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.embeddings_cache = {}
        
        # Inicializar modelo de embeddings
        if use_openai:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY nao configurada")
            self.embedder = OpenAIEmbeddings(model=embedding_model)
            logger.info(f"Usando OpenAI embeddings: {embedding_model}")
        else:
            # Usar modelo local (Sentence Transformers)
            model_name = "all-MiniLM-L6-v2"  # Modelo leve e eficiente
            self.embedder = SentenceTransformer(model_name)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Usando modelo local: {model_name}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para lista de textos
        
        Args:
            texts: Lista de textos para embeddings
            
        Returns:
            Array numpy com embeddings
        """
        if self.use_openai:
            # Usar LangChain OpenAI embeddings
            embeddings = self.embedder.embed_documents(texts)
            return np.array(embeddings)
        else:
            # Usar Sentence Transformers
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            return embeddings
    
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """
        Gera embedding para um unico texto
        
        Args:
            text: Texto para embedding
            
        Returns:
            Array numpy com embedding
        """
        if self.use_openai:
            embedding = self.embedder.embed_query(text)
            return np.array(embedding)
        else:
            embedding = self.embedder.encode([text], convert_to_numpy=True)
            return embedding[0]
    
    def create_index(self, documents: List[Document]) -> None:
        """
        Cria indice FAISS a partir dos documentos
        
        Args:
            documents: Lista de documentos LangChain
        """
        if not documents:
            raise ValueError("Lista de documentos vazia")
        
        logger.info(f"Criando indice para {len(documents)} documentos...")
        
        # Extrair textos
        texts = [doc.page_content for doc in documents]
        
        # Gerar embeddings
        logger.info("Gerando embeddings...")
        embeddings = self._get_embeddings(texts)
        
        # Criar indice FAISS
        logger.info("Criando indice FAISS...")
        
        # Usar IndexFlatIP para busca por produto interno (cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Adicionar embeddings ao indice
        self.index.add(embeddings)
        
        # Armazenar documentos e metadados
        self.documents = documents
        
        logger.info(f"Indice criado com sucesso - {self.index.ntotal} vetores indexados")
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Busca documentos similares a consulta
        
        Args:
            query: Consulta de busca
            k: Numero de resultados desejados
            score_threshold: Score minimo de similaridade
            
        Returns:
            Lista de resultados com documentos e scores
        """
        if not self.index or not self.documents:
            raise ValueError("Indice nao foi criado. Execute create_index() primeiro.")
        
        # Gerar embedding da consulta
        query_embedding = self._get_single_embedding(query)
        
        # Normalizar para cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Buscar no indice
        scores, indices = self.index.search(query_embedding, k)
        
        # Processar resultados
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= score_threshold:
                doc = self.documents[idx]
                
                result = {
                    'document': doc,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'rank': i + 1,
                    'filename': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown')
                }
                
                results.append(result)
        
        logger.info(f"Busca por '{query[:50]}...' retornou {len(results)} resultados")
        
        return results
    
    def similarity_search_with_threshold(self, 
                                       query: str, 
                                       k: int = 10, 
                                       score_threshold: float = 0.7) -> List[Document]:
        """
        Busca por similaridade com threshold de score
        
        Args:
            query: Consulta de busca
            k: Numero maximo de resultados
            score_threshold: Score minimo
            
        Returns:
            Lista de documentos similares
        """
        results = self.search(query, k, score_threshold)
        return [result['document'] for result in results]
    
    def get_relevant_context(self, 
                           query: str, 
                           max_tokens: int = 2000,
                           k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Obtem contexto relevante para uma consulta
        
        Args:
            query: Consulta de busca
            max_tokens: Numero maximo de tokens no contexto
            k: Numero de documentos para buscar
            
        Returns:
            Tupla com (contexto_concatenado, lista_de_fontes)
        """
        # Buscar documentos relevantes
        results = self.search(query, k)
        
        if not results:
            return "", []
        
        # Concatenar contexto respeitando limite de tokens
        context_parts = []
        total_tokens = 0
        sources = []
        
        for result in results:
            content = result['content']
            # Estimativa simples: ~4 chars = 1 token
            estimated_tokens = len(content) // 4
            
            if total_tokens + estimated_tokens <= max_tokens:
                context_parts.append(content)
                total_tokens += estimated_tokens
                
                # Adicionar fonte
                source_info = {
                    'filename': result['filename'],
                    'page': result['page'],
                    'score': result['score'],
                    'content': content[:200] + "..." if len(content) > 200 else content
                }
                sources.append(source_info)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Contexto gerado: {total_tokens} tokens estimados de {len(sources)} fontes")
        
        return context, sources
    
    def save_index(self, filepath: str) -> None:
        """
        Salva o indice em arquivo
        
        Args:
            filepath: Caminho para salvar o indice
        """
        if not self.index or not self.documents:
            raise ValueError("Nenhum indice para salvar")
        
        # Criar estrutura de dados para salvar
        save_data = {
            'documents': self.documents,
            'embedding_model': self.embedding_model,
            'use_openai': self.use_openai,
            'dimension': self.dimension
        }
        
        # Salvar indice FAISS
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Salvar metadados
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Indice salvo em: {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Carrega indice de arquivo
        
        Args:
            filepath: Caminho do arquivo de indice
        """
        # Carregar indice FAISS
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Carregar metadados
        with open(f"{filepath}.pkl", 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data['documents']
        self.embedding_model = save_data['embedding_model']
        self.use_openai = save_data['use_openai']
        self.dimension = save_data['dimension']
        
        logger.info(f"Indice carregado de: {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtem estatisticas do vector store
        
        Returns:
            Dicionario com estatisticas
        """
        if not self.index or not self.documents:
            return {'status': 'empty'}
        
        # Contar documentos por fonte
        sources = {}
        pages = set()
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
            
            page = doc.metadata.get('page')
            if page:
                pages.add(page)
        
        return {
            'status': 'ready',
            'total_documents': len(self.documents),
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'unique_sources': len(sources),
            'unique_pages': len(pages),
            'sources': sources,
            'embedding_model': self.embedding_model,
            'uses_openai': self.use_openai
        }

# Funcoes utilitarias
def create_vector_store_from_docs(documents: List[Document], 
                                 use_openai: bool = True) -> VectorStore:
    """
    Funcao utilitaria para criar vector store a partir de documentos
    
    Args:
        documents: Lista de documentos
        use_openai: Se deve usar OpenAI embeddings
        
    Returns:
        Vector store configurado e indexado
    """
    vs = VectorStore(use_openai=use_openai)
    vs.create_index(documents)
    return vs