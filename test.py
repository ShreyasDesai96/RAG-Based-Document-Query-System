import matplotlib.pyplot as plt
queries = ["Summarize architecture", "How is security handled?", "Describe LangChain’s function", "Explain data privacy protections", "What are the retrieval methods?"]
latencies = [3.2, 3.6, 2.9, 3.3, 2.7]
plt.bar(queries, latencies, color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Latency (seconds)")
plt.title("Query Latency Overview")
plt.tight_layout()
plt.show()

precisions = [1.0, 1.0, 1.0, 0.66, 1.0]
plt.bar(queries, precisions, color="salmon")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Precision (0-1)")
plt.title("Query Answer Precision")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
labels = ["Perfect answers", "Partial answers"]
sizes = [4, 1]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Answer Accuracy Distribution")
plt.show()