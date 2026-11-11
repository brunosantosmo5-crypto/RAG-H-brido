import pandas as pd
import chromadb

# 1. Ler o arquivo CSV usando o Pandas
print("Lendo o arquivo CVS...")
df = pd.read_csv("dados.csv")

# Exibindo o DataFrame no terminal paa conferir
print(df)
print("-" * 50)

# 2. Inicializar o ChromaDB (conectando na mesma pasta ' chroma_db')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="meus_dados_rag")

# 3. Converter as colunas do DataFrame em listas que o ChromaDB exige
documents = df["texto"].tolist()
# Criamos metadados baseados nas colunas 'id' e 'categoria' do CSV
metadatas = [{"id_original": str(row["id"]),  "categoria": str(row["categoria"])} for _, row in df.iterrows()]
ids = [str(row["id"]) for _, row in df.iterrows()]

# 4. Adicionar os dados na coleção do ChromaDB
collection.upsert(  # Usamos 'upsert' para atualizar caso o ID já exista
    documents=documents,
    metadatas=metadatas,  # type: ignore
    ids=ids
)

print("Sucesso! Todos os dados do arquivo CSV foram salvos no ChromaDB.")

# 5. Testando uma busca rápida nos dados do CSV
pergunta = "Como o editor de código funciona?"
print(f"\nTestando busca no CSV para: '{pergunta}'")

resultados = collection.query(
    query_texts=[pergunta],
    n_results=1
)

if resultados and resultados.get('documents'):
    print(f"Resultado mais relevante encontrado: {resultados['documents'][0][0]}") # type: ignore




