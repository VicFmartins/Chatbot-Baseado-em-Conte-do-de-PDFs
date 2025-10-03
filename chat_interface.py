#!/usr/bin/env python3
"""
Chat Interface - Modulo para interface de conversacao com RAG
Integra busca vetorial com modelos de linguagem para respostas contextuais
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)

class ChatInterface:
    """Interface de chat com capacidades RAG"""
    
    def __init__(self,
                 vector_store,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 memory_window: int = 10):
        """
        Inicializa a interface de chat
        
        Args:
            vector_store: Instance do VectorStore
            model_name: Nome do modelo OpenAI
            temperature: Temperatura para geracao
            max_tokens: Limite de tokens por resposta
            memory_window: Janela de memoria de conversacao
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY nao configurada")
        
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializar modelo de chat
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False
        )
        
        # Configurar memoria de conversacao
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Sistema de prompts
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"ChatInterface inicializada - Modelo: {model_name}, Temperatura: {temperature}")
    
    def _create_system_prompt(self) -> str:
        """
        Cria o prompt do sistema para o assistente
        
        Returns:
            String com o prompt do sistema
        """
        return """Voce e um assistente academico especializado em analise de documentos PDF. 
        Sua funcao e ajudar estudantes e pesquisadores a extrair informacoes relevantes de textos academicos.

        INSTRUCOES:
        1. Responda sempre em portugues brasileiro
        2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido
        3. Se a informacao nao estiver no contexto, diga claramente que nao foi encontrada
        4. Cite as fontes sempre que possivel (nome do arquivo e pagina)
        5. Mantenha um tom academico e profissional
        6. Se apropriado, sugira conexoes entre diferentes partes dos documentos
        7. Para perguntas complexas, estruture a resposta em topicos

        FORMATO DE RESPOSTA:
        - Resposta clara e objetiva
        - Citacoes das fontes entre parenteses: (nome_arquivo.pdf, p. X)
        - Se relevante, mencione limitacoes ou areas para investigacao adicional

        Lembre-se: voce esta ajudando na elaboracao de trabalhos academicos, entao precisao e confiabilidade sao fundamentais."""
    
    def _format_context(self, context: str, sources: List[Dict[str, Any]]) -> str:
        """
        Formata o contexto para o prompt
        
        Args:
            context: Texto do contexto
            sources: Lista de fontes
            
        Returns:
            Contexto formatado
        """
        if not context:
            return "Nenhum contexto relevante encontrado nos documentos."
        
        formatted_context = f"CONTEXTO DOS DOCUMENTOS:\n{context}\n\n"
        
        if sources:
            formatted_context += "FONTES:\n"
            for i, source in enumerate(sources, 1):
                formatted_context += f"{i}. {source['filename']} - Pagina {source['page']} (Score: {source['score']:.3f})\n"
        
        return formatted_context
    
    def _create_prompt(self, query: str, context: str, sources: List[Dict[str, Any]]) -> str:
        """
        Cria o prompt completo para o modelo
        
        Args:
            query: Pergunta do usuario
            context: Contexto relevante
            sources: Lista de fontes
            
        Returns:
            Prompt completo formatado
        """
        formatted_context = self._format_context(context, sources)
        
        prompt = f"""{self.system_prompt}

{formatted_context}

PERGUNTA DO USUARIO: {query}

RESPOSTA:"""
        
        return prompt
    
    def get_response(self, query: str, use_memory: bool = True) -> Dict[str, Any]:
        """
        Gera resposta para uma consulta usando RAG
        
        Args:
            query: Pergunta do usuario
            use_memory: Se deve usar memoria de conversacao
            
        Returns:
            Dicionario com resposta e metadados
        """
        try:
            # Buscar contexto relevante
            logger.info(f"Processando consulta: {query[:100]}...")
            
            context, sources = self.vector_store.get_relevant_context(
                query, 
                max_tokens=1500,  # Deixar espaco para resposta
                k=5
            )
            
            if not context:
                return {
                    'answer': "Desculpe, nao encontrei informacoes relevantes nos documentos carregados para responder sua pergunta. Verifique se os PDFs contem o conteudo relacionado ao que voce esta perguntando.",
                    'sources': [],
                    'context_used': False
                }
            
            # Criar prompt
            full_prompt = self._create_prompt(query, context, sources)
            
            # Gerar resposta
            messages = [HumanMessage(content=full_prompt)]
            
            # Adicionar memoria se solicitado
            if use_memory and hasattr(self.memory, 'chat_memory'):
                # Obter historico recente
                chat_history = self.memory.chat_memory.messages[-6:]  # Ultimas 3 trocas
                if chat_history:
                    messages = chat_history + messages
            
            # Chamar modelo
            response = self.llm(messages)
            answer = response.content.strip()
            
            # Salvar na memoria
            if use_memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(answer)
            
            logger.info(f"Resposta gerada com {len(sources)} fontes")
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': True,
                'model_info': {
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'context_length': len(context)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            return {
                'answer': f"Ocorreu um erro ao processar sua pergunta: {str(e)}",
                'sources': [],
                'context_used': False,
                'error': str(e)
            }
    
    def get_streaming_response(self, query: str) -> str:
        """
        Gera resposta com streaming (para interfaces em tempo real)
        
        Args:
            query: Pergunta do usuario
            
        Yields:
            Chunks da resposta conforme gerada
        """
        # Configurar streaming callback
        streaming_llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Buscar contexto
        context, sources = self.vector_store.get_relevant_context(query)
        
        if not context:
            yield "Desculpe, nao encontrei informacoes relevantes nos documentos."
            return
        
        # Criar prompt
        full_prompt = self._create_prompt(query, context, sources)
        
        # Gerar resposta com streaming
        messages = [HumanMessage(content=full_prompt)]
        
        try:
            for chunk in streaming_llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            yield f"Erro no streaming: {str(e)}"
    
    def summarize_documents(self, max_length: int = 500) -> str:
        """
        Cria um resumo dos documentos carregados
        
        Args:
            max_length: Comprimento maximo do resumo
            
        Returns:
            Resumo dos documentos
        """
        # Buscar amostras representativas
        sample_queries = [
            "resumo principal",
            "temas centrais",
            "conclusoes importantes",
            "metodologia utilizada"
        ]
        
        all_contexts = []
        for query in sample_queries:
            context, _ = self.vector_store.get_relevant_context(query, max_tokens=300, k=2)
            if context:
                all_contexts.append(context)
        
        combined_context = "\n\n".join(all_contexts)
        
        if not combined_context:
            return "Nao foi possivel gerar resumo dos documentos."
        
        # Prompt para resumo
        summary_prompt = f"""Com base no seguinte conteudo dos documentos, crie um resumo academico conciso:

{combined_context[:2000]}

Crie um resumo estruturado de ate {max_length} palavras cobrindo:
1. Temas principais
2. Metodologias mencionadas
3. Principais conclusoes
4. Areas de conhecimento abordadas

RESUMO:"""
        
        try:
            messages = [HumanMessage(content=summary_prompt)]
            response = self.llm(messages)
            return response.content.strip()
        except Exception as e:
            return f"Erro ao gerar resumo: {str(e)}"
    
    def suggest_questions(self) -> List[str]:
        """
        Sugere perguntas baseadas no conteudo dos documentos
        
        Returns:
            Lista de perguntas sugeridas
        """
        # Buscar contexto geral
        context, sources = self.vector_store.get_relevant_context(
            "principais temas conclusoes metodologia", 
            max_tokens=1000,
            k=3
        )
        
        if not context:
            return [
                "Quais sao os principais temas abordados?",
                "Que metodologias sao mencionadas?",
                "Quais as principais conclusoes?",
                "Existem recomendacoes para trabalhos futuros?"
            ]
        
        # Prompt para sugestoes
        suggestion_prompt = f"""Com base no seguinte conteudo dos documentos:

{context[:1500]}

Sugira 6 perguntas pertinentes que um estudante poderia fazer sobre este conteudo.
As perguntas devem ser:
- Especificas ao conteudo presente
- Adequadas para pesquisa academica
- Variadas em escopo (teoricas, metodologicas, aplicadas)

Liste apenas as perguntas, uma por linha, sem numeracao:"""
        
        try:
            messages = [HumanMessage(content=suggestion_prompt)]
            response = self.llm(messages)
            
            # Processar resposta
            questions = [q.strip() for q in response.content.split('\n') if q.strip()]
            return questions[:6]  # Maximo 6 sugestoes
            
        except Exception as e:
            logger.error(f"Erro ao sugerir perguntas: {str(e)}")
            return [
                "Quais sao os principais argumentos apresentados?",
                "Como os autores justificam suas conclusoes?",
                "Que evidencias sao apresentadas?",
                "Existem limitacoes mencionadas no estudo?"
            ]
    
    def clear_memory(self):
        """Limpa a memoria de conversacao"""
        self.memory.clear()
        logger.info("Memoria de conversacao limpa")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Obtem o historico de conversacao
        
        Returns:
            Lista de mensagens do historico
        """
        if not hasattr(self.memory, 'chat_memory'):
            return []
        
        messages = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                messages.append({'role': 'user', 'content': message.content})
            elif isinstance(message, AIMessage):
                messages.append({'role': 'assistant', 'content': message.content})
        
        return messages