from src.loaders.kaggle_loader import load_kaggle_data
from src.processing.chunker import chunk_text
from src.vectorstore.faiss_store import create_faiss_index
from src.retriever.hybrid_retriever import Retriever
from src.generator.llm_generator import generate_answer

print("🚀 Building RAG System...")

docs = load_kaggle_data()
chunks = chunk_text(docs)
index, _ = create_faiss_index(chunks)

retriever = Retriever(chunks, index)

print("\n✅ RAG Ready! Ask anything (type 'exit' to quit)\n")

while True:
    query = input("🧑 Ask: ")

    if query.lower() == "exit":
        break

    results = retriever.search(query)
    context = "\n".join(results[:3])

    answer = generate_answer(context, query)

    print("\n🤖 Answer:\n", answer)
    print("\n" + "-"*50)