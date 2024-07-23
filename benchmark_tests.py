import unittest
import requests
import timeit
import statistics

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.embedding_url = "http://localhost:6000/embeddings"
        self.splade_doc_url = "http://localhost:4000/embed_sparse"
        self.splade_query_url = "http://localhost:5000/embed_sparse"
        self.reranking_url = "http://localhost:8000/rerank"

    def test_embedding_service(self):
        payload = {
            "input": "This is a test sentence for embedding.",
            "model": "jinaai/jina-embeddings-v2-base-en"
        }
        times = timeit.repeat(lambda: requests.post(self.embedding_url, json=payload), number=1, repeat=32)
        avg_time = statistics.mean(times)
        print(f"Embedding Service Average Time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0)  # Oczekujemy, że średni czas będzie poniżej 1 sekundy

    def test_splade_doc_service(self):
        payload = {
            "inputs": "This is a test sentence for SPLADE document embedding."
        }
        times = timeit.repeat(lambda: requests.post(self.splade_doc_url, json=payload), number=1, repeat=32)
        avg_time = statistics.mean(times)
        print(f"SPLADE Doc Service Average Time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0)  # Oczekujemy, że średni czas będzie poniżej 1 sekundy

    def test_splade_query_service(self):
        payload = {
            "inputs": "This is a test query for SPLADE."
        }
        times = timeit.repeat(lambda: requests.post(self.splade_query_url, json=payload), number=1, repeat=32)
        avg_time = statistics.mean(times)
        print(f"SPLADE Query Service Average Time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0)  # Oczekujemy, że średni czas będzie poniżej 1 sekundy

    def test_reranking_service(self):
        payload = {
            "query": "This is a test query for reranking.",
            "texts": [
                "This is the first document for reranking.",
                "This is the second document for reranking.",
                "This is the third document for reranking."
            ],
            "truncate": False
        }
        times = timeit.repeat(lambda: requests.post(self.reranking_url, json=payload), number=1, repeat=32)
        avg_time = statistics.mean(times)
        print(f"Reranking Service Average Time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0)  # Oczekujemy, że średni czas będzie poniżej 1 sekundy

if __name__ == '__main__':
    unittest.main()