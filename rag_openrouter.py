import chromadb
from openai import OpenAI

# 1. Configurar o cliente da OpenAI apontando para a OpenRouter
# (Lembre-se de colocar sua chave real da OpenRouter abaixo)
client_openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="SUA-CHAVE-API",  
)

# 2. Conectar ao ChromaDB que já criamos antes
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="meus_dados_rag")

# 3. Definir a pergunta do usuário
pergunta = "O que é o ChromaDB e para que ele serve?"
print(f"Pergunta do usuário: {pergunta}\n")

# 4. Buscar os trechos mais relevantes no ChromaDB (Retrieval)
resultados = collection.query(
    query_texts=[pergunta],
    n_results=2
)

# Extrair os textos encontrados (com tratamento seguro e type: ignore para o Pylance)
documentos_recuperados = []
if resultados and 'documents' in resultados and resultados['documents']:
    documentos_recuperados = resultados['documents'][0]  # type: ignore

print("--- Trechos recuperados do banco vetorial ---")
for i, doc in enumerate(documentos_recuperados):
    print(f"{i+1}. {doc}")
print("-" * 50)

# 5. Montar o "Prompt de Contexto" (Augmented Generation)
contexto = "\n\n".join(documentos_recuperados)

prompt_sistema = f"""
Você é um assistente de IA prestativo. Responda à pergunta do usuário baseando-se APENAS no contexto fornecido abaixo. Se a resposta não estiver no contexto, diga que não sabe.

Contexto:
{contexto}

Pergunta: {pergunta}
"""

# 6. Enviar para a OpenRouter usando um modelo gratuito/disponível
print("Gerando resposta via OpenRouter...\n")

resposta = client_openrouter.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",  # Você pode alterar o modelo se preferir
    messages=[
        {"role": "user", "content": prompt_sistema}
    ]
)

# 7. Exibir a resposta final gerada pelo RAG
print("=== Resposta Final da IA (OpenRouter) ===")
print(resposta.choices[0].message.content)