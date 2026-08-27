# %%
import os
import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def tokenize(text: str) -> list[str]:
    return re.findall(r'[\wа-яё]+', text.lower())

def search(query: str, top_k: int = 3):
    query_vec = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k
    )
    print(f"\n=== Запрос: {query} ===")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"[dist={dist:.4f}] {doc[:150]}")
    return results['documents'][0]

def rrf_fusion(rankings: list[list[str]], k: int = 60, top_k: int = 5) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key = scores.get, reverse=True)[:top_k]

def hybrid_search(query: str, top_k: int = 3) -> list[str]:
    query_vec = model.encode(query).tolist()
    semantic_res = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k * 2
    )
    semantic_ids = semantic_res['ids'][0] if semantic_res['ids'] else []

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    id_with_scores = list(zip(doc_ids, bm25_scores))
    id_with_scores.sort(key=lambda x: x[1], reverse=True)
    lexical_ids = [doc_id for doc_id, score in id_with_scores [:top_k * 2]]
    final_ids = rrf_fusion([semantic_ids, lexical_ids], top_k=top_k)

    if not final_ids:
            return []
    
    retrieved = collection.get(ids=final_ids)
    id_to = {i: (d, m['source']) for i, d, m in 
                 zip(retrieved['ids'], retrieved['documents'], retrieved['metadatas'])}
    return [id_to[i] for i in final_ids if i in id_to]

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection('kbase') 

data = collection.get()
doc_ids = data['ids']                          
docs = data['documents']
bm25 = BM25Okapi([tokenize(d) for d in docs])  

if __name__ == "__main__":
    questions = [
        "Какие векторные базы данных упоминаются?",
        "Что такое overlap?"
    ]
    for q in questions:
        print(f"\n======================")
        print(f"ЗАПРОС: {q}")
        print(f"======================")
        
        sem = collection.query(query_embeddings=[model.encode(q).tolist()], n_results=3)
        print("SEMANTIC:")
        for doc, src in sem['documents'][0]:
            print("  ", doc,src[:100].replace('\n', ' '))
            
        print("\nHYBRID:")
        for doc, src in hybrid_search(q, top_k=3):
            print("  ", doc,src[:100].replace('\n', ' '))



