import boto3
import streamlit as st
import os
import uuid

## S3_Client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock Embeddings
from langchain_community.embeddings import BedrockEmbeddings

## Bedrock
from langchain.llms.bedrock import Bedrock

## For RAG
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

## PDF Loader
from langchain_community.document_loaders import PyPDFLoader

## Import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

## Unique ID Generation Function
def get_unique_id():
    return str (uuid.uuid4())

## Load Index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}myfaiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

## Get LLMs
def get_llm():
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample':512})
    return llm

## Get Response
def get_response(llm):
    ## Create Prompt / Template
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5}),
        return_source_documents=True
        chain_type_kwargs = {"prompt": PROMPT}
    )

    answer = qa({"query":question})
    return answer['result']

## Main method 
def main():
    st.write("This is client site for chat with pdf demo using Bedrock, RAG etc.")

    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    ## Create Index
    faiss_index = FAISS.load_local(index_name="my_faiss",
                                   folder_path = folder_path,
                                   embeddings = bedrock_embeddings,
                                   allow_dangerous_deserialization=True)

    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")
    if st.button("Ask Question"):
        with st.spinner("Querying....")

        llm = get_llm()

        # get_response
        st.write(get_response(llm, faiss_index, question))
        st.success("Done")

if __name__ == "__main__":
    main()
