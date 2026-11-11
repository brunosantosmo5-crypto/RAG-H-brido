import chromadb
from openai import OpenAI

# 1. Configurar o cliente da OpenAI apontando para a OpenRouter
client_openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Cole sua chave real da OpenRouter aqui
)

# 2. Conectar ao ChromaDB (onde os dados do seu CSV estão salvos)
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="meus_dados_rag")

# 3. Defina a sua pergunta sobre os dados do CSV
pergunta = "O que o editor de código Visual Studio Code faz?"
print(f"Pergunta do usuário: {pergunta}\n")

# 4. Buscar os trechos mais relevantes no ChromaDB (Retrieval)
resultados = collection.query(
    query_texts=[pergunta],
    n_results=2  # Busca os 2 trechos mais próximos do CSV
)

# Extrair os textos encontrados com segurança
documentos_recuperados = []
if resultados and 'documents' in resultados and resultados['documents']:
    documentos_recuperados = resultados['documents'][0]  # type: ignore

print("--- Trechos recuperados do arquivo CSV (via ChromaDB) ---")
for i, doc in enumerate(documentos_recuperados):
    print(f"{i+1}. {doc}")
print("-" * 50)

# 5. Montar o "Prompt de Contexto" para o RAG
contexto = "\n\n".join(documentos_recuperados)

prompt_sistema = f"""
Você é um assistente prestativo. Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nas informações do contexto fornecido abaixo. Se a informação não estiver lá, diga que não sabe.

Contexto (extraído do CSV):
{contexto}

Pergunta: {pergunta}
"""

# 6. Enviar para a OpenRouter gerar a resposta final
print("Gerando resposta baseada no CSV via OpenRouter...\n")

resposta = client_openrouter.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",  # Modelo gratuito e ativo na OpenRouter
    messages=[
        {"role": "user", "content": prompt_sistema}
    ]
)

# 7. Exibir a resposta final da IA
print("=== Resposta Final da IA (Baseada no seu CSV) ===")
print(resposta.choices[0].message.content)