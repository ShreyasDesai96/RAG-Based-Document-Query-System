# RAG-Based-Document-Query-System
 Retrieval-Augmented Generation (RAG) System using AWS Bedrock, LangChain, FAISS, and LLaMA 3
This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for intelligent document querying and contextual answering. It leverages cutting-edge tools like AWS Bedrock, Titan Embeddings, LLaMA 3, FAISS, and LangChain to build a scalable and performant question-answering pipeline over custom document data.

🔧 Tech Stack
LLM: LLaMA 3

Embeddings: AWS Titan Embeddings

Vector Store: FAISS (Facebook AI Similarity Search)

Framework: LangChain

Frontend/UI: Streamlit

Orchestration: AWS Bedrock


🚀 Features


✅ Accepts user queries and retrieves relevant document chunks using semantic search

✅ Uses Titan Embeddings to vectorize documents and store them in FAISS

✅ Passes retrieved context into LLaMA 3 via LangChain RAG pipeline

✅ Offers a clean, interactive interface via Streamlit

✅ Easily extendable to support additional LLMs or embedding models via LangChain modularity

✅ Hosted and orchestrated using AWS Bedrock for enterprise scalability

📂 Project Structure

├── data                 # Dataset is uploaded here
├── faiss_index          # Indexes are created and stored in vector store
├── venv                 # Virtual Enviorment is created using this
├── app.py               # Code is written in this portion 
├── benchmarks.json      # Used for testing purpose
|── benchmark.py         # testing and graphs are made here
|── llama3.py            # llama3 model is invoked
└── requiremets.txt      # all the libraries are installed using requirements
└──test.py               # to test the perfromance of the ap


Streamlit UI allowing users to query documents with real-time LLM-generated responses.

📈 Use Cases
Internal document search and Q&A

Enterprise knowledge base assistance

Chatbot support over structured/unstructured data

Legal/Medical/Research paper summarization

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Note: Requires AWS credentials with access to Bedrock APIs and LLaMA 3.

📄 Future Improvements
Add support for multi-document ingestion and chunking

Integrate user feedback loop for continuous learning

Replace FAISS with scalable vector DB like Pinecone or Weaviate

Add authentication and role-based access to the Streamlit app
