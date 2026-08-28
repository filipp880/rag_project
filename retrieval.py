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

def semantic_sources(query, top_k=3):
    res = collection.query(query_embeddings=[model.encode(query).tolist()],
                           n_results=top_k, include=['metadatas'])
    return [m['source'] for m in res['metadatas'][0]]

def hybrid_sources(query,top_k=3):
    return [src for _, src in hybrid_search(query, top_k=top_k)]

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection('kbase') 

data = collection.get()
doc_ids = data['ids']                          
docs = data['documents']
bm25 = BM25Okapi([tokenize(d) for d in docs])  

if __name__ == "__main__":
    questions = [
    ("Что такое overlap и зачем он нужен?", "article.txt"),
    ("Почему LLM галлюцинируют?", "article.txt"),
    ("Какие эмбеддеры актуальны для русского языка?", "article.txt"),
    ("Когда RAG действительно нужен?", "article.txt"),
    ("Что такое ingest?", "article.txt"),
    ("Какие базы данных наиболее часто встречаются?", "article.txt"),
    ("В чем отличие гибридного и семантического поиска?", "article.txt"),
    ("Приведи пример реккурентных нейросетей", "transformer_notes.txt"),
    ("Что такое self-attention?", "transformer_notes.txt"),
    ("Что такое функция softmax?", "transformer_notes.txt"),
    ("Что такое multi-head-attention?", "transformer_notes.txt"),
    ("Что позволяет модели ориентироваться в структуре текста?", "transformer_notes.txt"),
    ("С какими задачами отлично справляется нейросеть на архитектуре transformer?", "transformer_notes.txt")
]
    sem_hits = hyb_hits = 0
    for q, expected in questions:
        sem_ok = expected in semantic_sources(q)
        hyb_ok = expected in hybrid_sources(q)
        sem_hits += sem_ok
        hyb_hits += hyb_ok
        print(f"[sem {'+' if sem_ok else '-'} | hyb {'+' if hyb_ok else '-'}] {q}")
    n = len(questions)
    print(f"Recall@3 semantic: {sem_hits}/{len(questions)} = {sem_hits/len(questions):.2f}")
    print(f"Recall@3 hybrid: {hyb_hits}/{len(questions)} = {hyb_hits/len(questions):.2f}")



