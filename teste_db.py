import pandas as pd
import chromadb

# Cria um cliente do ChromaDB rodando localmente na memória (ou salvando na pasta)
chroma_client = chromadb.Client()

# Cria uma coleção de teste
collection = chroma_client.create_collection(name="meu_teste_rag")

print("Tudo instalado com sucesso! Pandas e ChromaDB prontos para uso.")