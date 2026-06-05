JD_TEXT = """
Senior AI Engineer with 5 to 9 years of experience in applied machine learning 
and information retrieval at product companies. Not a research role — this person 
has shipped real systems to real users.

They have production experience building embedding-based retrieval systems using 
models like sentence-transformers, BGE, E5, or OpenAI embeddings. They have handled 
embedding drift, index refresh, and retrieval quality regression in production — not 
just toy projects.

They have hands-on experience with vector databases and hybrid search infrastructure: 
Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, or OpenSearch. They know 
the operational side, not just the API calls.

They have designed and shipped ranking systems, recommendation engines, or semantic 
search systems. They understand NDCG, MRR, MAP — evaluation metrics for ranking. 
They have set up offline benchmarks and online A/B tests for search quality.

Strong Python skills. They write clean, production-quality code. Not just scripts — 
actual systems.

They have worked at product companies or startups — not their entire career at 
IT services firms like TCS, Wipro, Infosys, Accenture, or Cognizant. They have 
built things that users actually use.

Ideally located in Pune, Noida, Delhi NCR, Mumbai, Hyderabad, or Bangalore in India. 
Or willing to relocate there. Open to work and available soon — short notice period preferred.

Bonus: LLM fine-tuning experience with LoRA or QLoRA. Learning-to-rank models. 
Open source contributions in AI/ML. Experience with BM25 and hybrid retrieval.

They are a builder, not a titleholder. They have shipped a v2 of something based 
on real user feedback. They think about latency, scale, and eval — not just accuracy 
on a benchmark.

NOT a fit: pure computer vision, speech recognition, or robotics background without 
NLP or information retrieval experience. NOT a fit: AI experience that is only 
LangChain tutorials calling OpenAI APIs in the last 12 months. NOT a fit: someone 
who has not written production code in 18 months because they moved into architecture.
"""

if __name__ == "__main__":
    print("JD text length:", len(JD_TEXT.split()), "words")
    print("\nFirst 300 chars:")
    print(JD_TEXT[:300])