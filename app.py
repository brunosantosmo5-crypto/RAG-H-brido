import streamlit as st
import chromadb
from openai import OpenAI
from duckduckgo_search import DDGS

# Configuração da Página do Streamlit (sem CSS customizado que oculta o texto)
st.set_page_config(
    page_title="RAG Híbrido Assistant", 
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Bate-papo com RAG Híbrido")
st.write("Converse com seus dados locais (CSV) ou pesquise na internet em tempo real.")

# 1. Configurar recursos com cache
@st.cache_resource
def carregar_recursos():
    client_openrouter = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key= "SUA-CHAVE-API", # Cole sua chave real da OpenRouter aqui
    )
    client_chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = client_chroma.get_or_create_collection(name="meus_dados_rag")
    return client_openrouter, collection

client_openrouter, collection = carregar_recursos()

# --- BARRA LATERAL (SIDEBAR) CUSTOMIZADA ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.header("Painel de Controle")
    st.info("Status do Sistema: RAG Híbrido Ativo 🟢")
    
    # Contar quantos documentos estão salvos no ChromaDB
    try:
        total_docs = collection.count()
        st.metric(label="Documentos na Base Local", value=total_docs)
    except Exception:
        st.metric(label="Documentos na Base Local", value=0)
    
    st.markdown("---")
    
    # Botão para limpar histórico do chat
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Pilha Tecnológica:**")
    st.markdown("- Streamlit (interface do usuário)")
    st.markdown("- ChromaDB (Vetorial)")
    st.markdown("- Pandas (Dados)")
    st.markdown("- OpenRouter (LLM)")
    st.markdown("- DuckDuckGo (Web)")

# Função para buscar na internet usando DuckDuckGo
def buscar_na_web(termo_busca):
    try:
        with DDGS() as ddgs:
            resultados = [r['body'] for r in ddgs.text(termo_busca, max_results=3)]
            return "\n\n".join(resultados)
    except Exception as e:
        return f"Não foi possível buscar na web: {e}"

# 2. Inicializar histórico de mensagens na tela principal
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Caixa de entrada do chat
if pergunta := st.chat_input("Faça uma pergunta (sobre o CSV ou sobre a internet)..."):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando e pesquisando..."):
            
            # Passo A: Tentar buscar no ChromaDB (CSV local)
            resultados_chroma = collection.query(
                query_texts=[pergunta],
                n_results=2
            )
            
            documentos_recuperados = []
            if resultados_chroma and 'documents' in resultados_chroma and resultados_chroma['documents']:
                documentos_recuperados = resultados_chroma['documents'][0]  # type: ignore

            origem_dado = "Base Local (CSV)"
            contexto = "\n\n".join(documentos_recuperados)

            # Passo B: Se o ChromaDB não achar contexto relevante, busca na Internet!
            if not contexto.strip():
                origem_dado = "Internet (Web Search)"
                contexto = buscar_na_web(pergunta)

            # Montar prompt dinâmico informando a origem
            prompt_sistema = f"""
            Você é um assistente prestativo. Responda à pergunta do usuário baseando-se no contexto fornecido abaixo.
            A origem destes dados é: {origem_dado}.

            Contexto:
            {contexto}

            Pergunta: {pergunta}
            """

            # Gerar resposta com a OpenRouter
            resposta = client_openrouter.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt_sistema}]
            )
            
            resposta_ia = resposta.choices[0].message.content
            
            # Exibir a resposta e a fonte utilizada
            st.markdown(resposta_ia)
            st.caption(f"🔍 *Fonte utilizada: {origem_dado}*")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"{resposta_ia}\n\n*(Fonte: {origem_dado})*"
    })