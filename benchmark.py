import json
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3

# Load benchmark questions
with open("benchmark_queries.json", "r") as f:
    benchmark_queries = json.load(f)

# Initialize bedrock
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)
llm = Bedrock(
    model_id='meta.llama3-70b-instruct-v1:0',
    client=bedrock,
    model_kwargs={'max_gen_len':512}
)

# Load FAISS
faiss_index = FAISS.load_local("faiss_index", bedrock_embedding, allow_dangerous_deserialization=True)

# Prompt
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Metrics storage
results = []

for entry in benchmark_queries:
    query = entry["query"]
    expected = entry["expected_answer_contains"]
    
    try:
        # measure retrieval+generation latency
        start_time = time.time()
        result = qa_chain({"query": query})
        end_time = time.time()
        
        total_latency = end_time - start_time
        
        answer = result['result']
        
        # simple precision-like metric: do expected keywords appear
        hits = sum(1 for keyword in expected if keyword.lower() in answer.lower())
        precision = hits / len(expected)
        
        results.append({
            "query": query,
            "latency_sec": round(total_latency, 2),
            "precision": precision,
            "answer_snippet": answer[:200]
        })
        
    except Exception as e:
        results.append({
            "query": query,
            "error": str(e)
        })

# Report
print("=== BENCHMARK RESULTS ===")
for r in results:
    print(json.dumps(r, indent=2))

import matplotlib.pyplot as plt

# Filter out only successful queries
successful_results = [r for r in results if "latency_sec" in r]

# Collect data
queries = [r["query"] for r in successful_results]
latencies = [r["latency_sec"] for r in successful_results]
precisions = [r["precision"] for r in successful_results]

# Latency bar plot
plt.figure(figsize=(10, 5))
plt.bar(queries, latencies, color="skyblue")
plt.xlabel("Query")
plt.ylabel("Latency (s)")
plt.title("Query Latency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Precision bar plot
plt.figure(figsize=(10, 5))
plt.bar(queries, precisions, color="salmon")
plt.xlabel("Query")
plt.ylabel("Precision (0–1)")
plt.title("Query Answer Precision")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

