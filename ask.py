from query_transform import _generate
import requests
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("my_docs")

""" def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return resp.json()["embedding"] """

def ask(query, n_results=4):
    query_embedding = _generate(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    chunks = results["documents"][0]
    #print(results)
    print("--- RETRIEVED CHUNKS ---")
    for c in chunks:
        print(c)
    print("------------------------")
    sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n".join(f"[{s}]\n{c}" for s, c in zip(sources, chunks))
    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}"""

    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return resp.json()["response"]

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ")
        if q.lower() == "quit":
            break
        print("\n" + ask(q))