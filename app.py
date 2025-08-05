import json
import os
import sys
import boto3
import streamlit as st

## Titan Embeddings model to generate embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

## Libraries for data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embeddings And Vector store
from langchain_community.vectorstores import FAISS

## LLM Model
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

## calling Bedrock Client
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=bedrock)

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader('Data')
    documents = loader.load()

    #recurssive character text split
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000,
                                                  chunk_overlap = 1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

## Vector embeddings and vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    llm = Bedrock(
        model_id='meta.llama3-70b-instruct-v1:0',
        client=bedrock,
        model_kwargs={'max_gen_len':512}
    )
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with the PDF💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Output"):
        with st.spinner("Processing..."):
            
            llm=get_llama3_llm()
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding, allow_dangerous_deserialization=True)
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()



    
