# %%
import os
import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def rerank(query: str, docs: list[str], top_k: int = 3) -> list[str]:
    if not docs:
        return []
    pairs = [(query,doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs,scores), key = lambda x: -x[1])
    return [doc for doc, _ in ranked[:top_k]]

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

def hybrid_search(query: str, top_k: int = 3, use_reranker: bool = True) -> list[str]:
    query_vec = model.encode(query).tolist()
    semantic_res = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k * 4
    )
    semantic_ids = semantic_res['ids'][0] if semantic_res['ids'] else []
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    id_with_scores = list(zip(doc_ids, bm25_scores))
    id_with_scores.sort(key=lambda x: x[1], reverse=True)
    lexical_ids = [i for i, _ in sorted(zip(doc_ids, bm25_scores),
                                        key=lambda x: x[1], reverse=True)[:top_k*4]]
    pool = 20 if use_reranker else top_k
    final_ids = rrf_fusion([semantic_ids, lexical_ids], top_k=pool)
    if not final_ids:
            return []
    
    retrieved = collection.get(ids=final_ids)
    docs = retrieved['documents']
    sources = [m['source'] for m in retrieved['metadatas']]
    if use_reranker:
        ranked = rerank(query, docs, top_k=top_k)
        doc_to_src = dict(zip(docs, sources))
        return[(d, doc_to_src[d]) for d in ranked]
    return list(zip(docs, sources))


def semantic_sources(query, top_k=3):
    res = collection.query(query_embeddings=[model.encode(query).tolist()],
    n_results=top_k, include=['metadatas'])
    return [m['source'] for m in res['metadatas'][0]]

def hybrid_sources(query,top_k=3, use_reranker = True):
    return [src for _, src in hybrid_search(query, top_k=top_k, use_reranker=use_reranker)]

def calculate_mrr(questions: list[tuple[str,str]], top_k: int = 3) -> tuple[float,float]:
    sem_mrr_sum = 0.0
    hyb_mrr_sum = 0.0
    for q, expected in questions:
        sem_sources_list = semantic_sources(q, top_k = top_k)
        hyb_sources_list = hybrid_sources(q, top_k=top_k)
        if expected in sem_sources_list:
            rank = sem_sources_list.index(expected) + 1
            sem_mrr_sum += 1.0/rank
        if expected in hyb_sources_list:
            rank = hyb_sources_list.index(expected) + 1
            hyb_mrr_sum += 1.0 / rank
    n = len(questions)
    return sem_mrr_sum / n, hyb_mrr_sum / n

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
    sem_mrr, hyb_mrr = calculate_mrr(questions, top_k =3)
    for use_reranker in (False, True):
        sem_hits = hyb_hits = 0
        sem_mrr = hyb_mrr = 0.0
        for q, expected in questions:
            sem_list = semantic_sources(q)
            hyb_list = hybrid_sources(q, use_reranker=use_reranker)
            if expected in sem_list:
                sem_hits += 1; sem_mrr += 1.0 / (sem_list.index(expected) +1)
            if expected in hyb_list:
                hyb_hits += 1; hyb_mrr += 1.0 / (hyb_list.index(expected) + 1)
        n = len(questions)
        print(f"\n==={'с реранкером' if use_reranker else 'без реранкера'}===")
        print(f"Recall@3: semantic {sem_hits/n:.2f} | hybrid {hyb_hits/n:.2f}")
        print(f"MRR@3: semantic {sem_mrr/n:.3f} | hybrid {hyb_mrr/n:.3f}")


