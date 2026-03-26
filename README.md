# Chatbot Baseado em Conteudo de PDFs

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/arquitetura-RAG-0a7ea4)
![Streamlit](https://img.shields.io/badge/interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/busca-FAISS-2e8b57)

Aplicação RAG para conversar com o conteúdo de documentos PDF, com upload via Streamlit, extração de texto, chunking, indexação vetorial e respostas com fontes.

O projeto foi melhorado para sair do estado de módulos isolados e virar uma aplicação utilizável de ponta a ponta.

## Visão Geral

Este chatbot foi pensado para cenários como:

- revisão bibliográfica para TCC;
- leitura orientada de artigos científicos;
- consulta rápida a apostilas e relatórios;
- exploração de múltiplos PDFs em um único espaço de conversa.

O fluxo principal é:

1. o usuário envia um ou mais PDFs;
2. o sistema extrai e limpa o texto;
3. o conteúdo é dividido em chunks;
4. os chunks são indexados em FAISS;
5. a pergunta do usuário recupera os trechos mais relevantes;
6. a resposta é gerada com base no contexto encontrado.

## Funcionalidades

- Upload de múltiplos PDFs
- Extração híbrida de texto com `pdfplumber` e `PyMuPDF`
- Chunking configurável
- Indexação vetorial com FAISS
- Busca semântica por similaridade
- Respostas com indicação de fontes
- Interface de chat em Streamlit
- Modo degradado sem OpenAI:
  recuperação e exibição de trechos relevantes mesmo sem chave de API

## Arquitetura Atual

```text
.
├── app.py
├── streamlit_app.py
├── chat_interface.py
├── pdf_processor.py
├── vector_store.py
├── config.yaml
├── .env.example
├── requirements.txt
└── data/
    └── uploads/
```

## Principais Componentes

### `pdf_processor.py`

Responsável por:

- extrair texto de PDFs;
- comparar resultados de extração;
- limpar ruídos comuns;
- gerar chunks com metadados por página.

### `vector_store.py`

Responsável por:

- gerar embeddings;
- indexar vetores com FAISS;
- buscar trechos semanticamente relevantes;
- devolver contexto e fontes para o chat.

### `chat_interface.py`

Responsável por:

- construir o prompt;
- consultar os trechos relevantes;
- gerar respostas com OpenAI quando disponível;
- operar em modo extrativo quando não houver chave configurada.

### `streamlit_app.py`

Responsável por:

- interface web;
- upload dos arquivos;
- processamento dos PDFs;
- exibição da conversa, sugestões e fontes.

## Stack

- Python
- Streamlit
- LangChain
- FAISS
- OpenAI
- Sentence Transformers
- pdfplumber
- PyMuPDF

## Como Executar

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd Chatbot-Baseado-em-Conte-do-de-PDFs
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```bash
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

No PowerShell:

```powershell
Copy-Item .env.example .env
```

Se você definir `OPENAI_API_KEY`, o sistema poderá gerar respostas sintetizadas com LLM.

Sem a chave, o app continua funcionando em modo extrativo, mostrando os trechos mais relevantes encontrados.

### 5. Execute a aplicação

```bash
streamlit run streamlit_app.py
```

## Configuração

O arquivo `config.yaml` controla:

- provider de embeddings;
- modelo de chat;
- tamanho de chunk e overlap;
- quantidade de resultados;
- mensagens padrão da interface.

Por padrão, os embeddings estão configurados para modo local, o que reduz atrito inicial.

## Exemplo de Uso

1. Envie 2 ou 3 PDFs pela barra lateral.
2. Clique em `Processar PDFs`.
3. Espere a criação do índice vetorial.
4. Faça perguntas como:

- "Qual é o objetivo principal dos documentos?"
- "Quais metodologias aparecem com mais frequência?"
- "Existe consenso entre os autores?"
- "Resuma as conclusões com indicação das páginas"

## Melhorias Aplicadas Nesta Versão

- criação da aplicação Streamlit que faltava no repositório;
- alinhamento do README com a estrutura real;
- correção da limpeza de texto no processamento de PDFs;
- extração híbrida mais robusta;
- tratamento melhor de erros na indexação e no chat;
- modo degradado quando a OpenAI não está configurada;
- simplificação das dependências para facilitar instalação.

## Próximos Passos

- persistir índices FAISS entre sessões;
- adicionar suporte a DOCX e TXT;
- incluir OCR para PDFs escaneados;
- exportar respostas e referências;
- adicionar testes automatizados;
- permitir troca de modelos pela interface.

## Observação

Este projeto é uma base prática de RAG aplicada a documentos. Ele já pode ser usado como MVP e também serve como ponto de partida para evoluir para produto acadêmico, assistente interno ou ferramenta de pesquisa documental.
