import os
import chromadb

def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 150) -> list[str]:
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def main():
    if hasattr(chromadb, "PersistentClient"):
        client = chromadb.PersistentClient(path="./chromadb")
    elif hasattr(chromadb, "Client"):
        client = chromadb.Client()
    else:
        raise ImportError("Не удалось инициализировать ChromaDB. Проверьте отсутствие файла chromadb.py в директории.")

    try:
        client.delete_collection('kbase')
    except Exception:
        pass
        
    collection = client.create_collection(
        name='kbase', 
        metadata={"hnsw:space": "cosine"}
    )

    data_dir = 'data'
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath) and fname.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
                
            chunks = chunk_with_overlap(text)
            
            if not chunks:
                continue

            collection.add(
                documents=chunks,
                ids=[f"{fname}_{i}" for i in range(len(chunks))],
                metadatas=[{'source': fname} for _ in chunks]
            )
            print(f"{fname}: {len(chunks)} чанков")

    print("\n--- Проверка фильтрации ---")
    res = collection.get(where={'source': 'transformer_notes.txt'})
    print("чанков из transformer_notes.txt:", len(res['ids']))

if __name__ == "__main__":
    main()