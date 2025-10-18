import boto3
import streamlit as st
import os
import uuid

## S3_Client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock Embeddings
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

## PDF Loader
from langchain_community.document_loaders import PyPDFLoader

## Import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

## Unique ID Generation Function
def get_unique_id():
    return str (uuid.uuid4())

## Main method 
def main():
    st.write("This is client site for chat with pdf demo using Bedrock, RAG etc.")

if __name__ == "__main__":
    main()
