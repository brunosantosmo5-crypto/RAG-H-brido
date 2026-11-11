# 🚀 Hybrid RAG Assistant: Busca Local (Vector DB) + Fallback Web (LLM)

Uma aplicação de **Retrieval-Augmented Generation (RAG) Híbrido** desenvolvida em Python, com interface web interativa em Streamlit. O sistema prioriza a busca em dados locais estruturados via vetorização e executa fallback automático para busca na internet quando a informação local é insuficiente.

---

## 🛠️ Stack Tecnológica

* **Linguagem:** Python 3.14+
* **Interface (UI):** Streamlit
* **Vector Store:** ChromaDB (Persistente)
* **Manipulação de Dados:** Pandas
* **Inferência de LLM:** OpenRouter API (Meta Llama 3.1 8B Instruct)
* **Busca na Web (Fallback):** DuckDuckGo Search API

---

## 🧠 Arquitetura da Solução

```text
[ Usuário (Streamlit UI) ]
            │
            ▼
┌─────────────────────────┐
│  Busca Vetorial Local   │ ──► [ Encontrou contexto ] ──► [ Prompt + Contexto ] ──┐
│       (ChromaDB)        │                                                      │
└─────────────────────────┘                                                      │
            │                                                                    │
            ▼ (Sem contexto relevante)                                           ▼
┌─────────────────────────┐                                             ┌─────────────────┐
│     Busca na Web        │ ──► [ Contexto da Internet ] ───────────────► │ OpenRouter API  │
│    (DuckDuckGo API)     │                                             │ (Llama 3.1 8B)  │
└─────────────────────────┘                                             └─────────────────┘
                                                                                 │
                                                                                 ▼
                                                                        [ Resposta ao Usuário ]