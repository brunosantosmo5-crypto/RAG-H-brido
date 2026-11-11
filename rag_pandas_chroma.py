import pandas as pd
import chromadb

# 1. Criar um DataFrame de exemplo
dados = {
    "id": ["1", "2", "3"],
    "texto": [
        "O Pandas é uma biblioteca em Python para manipulação e análise de dados.",
        "O ChromaDB é um banco de dados vetorial leve, ideal para aplicações de RAG.",
        "RAG (Retrieval-Augmented Generation) melhora as respostas de LLMs buscando em documentos externos."
    ],
    "categoria": ["Python", "Banco de Dados", "IA"]
}

df = pd.DataFrame(dados)
print("DataFrame original:")
print(df)
print("-" * 50)

# 2. Inicializar o ChromaDB com persistência local
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="meus_dados_rag")

# 3. Preparar os dados
documents = df["texto"].tolist()
metadatas = [{"categoria": str(cat)} for cat in df["categoria"]]
ids = df["id"].tolist()

# 4. Adicionar os dados na coleção (ignorando o aviso chato de tipagem do Pylance)
collection.add(
    documents=documents,
    metadatas=metadatas,  # type: ignore
    ids=ids
)

print("Dados do DataFrame salvos com sucesso no ChromaDB!")
print("-" * 50)

# 5. Fazendo uma busca de RAG
pergunta = "Como o ChromaDB ajuda em inteligência artificial?"
print(f"Buscando por: '{pergunta}'\n")

resultados = collection.query(
    query_texts=[pergunta],
    n_results=2
)

# Validando e exibindo os resultados (com type: ignore para acalmar o linter)
if resultados and resultados.get('documents'):
    docs = resultados['documents'][0]  # type: ignore
    metas = resultados['metadatas'][0] if resultados.get('metadatas') else []  # type: ignore
    distances = resultados['distances'][0] if resultados.get('distances') else []  # type: ignore

    for i, doc in enumerate(docs):
        print(f"Resultado {i+1}:")
        print(f"  - Texto: {doc}")
        meta_atual = metas[i] if i < len(metas) else {}
        print(f"  - Metadados: {meta_atual}")
        dist_atual = distances[i] if i < len(distances) else "N/A"
        print(f"  - Distância/Relevância: {dist_atual}")
        print()
else:
    print("Nenhum resultado encontrado.")