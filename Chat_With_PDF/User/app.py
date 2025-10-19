import boto3
import streamlit as st
import os
import uuid

## s3 Client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Main Method
def main():
    st.header("This is Client Site for Chat with PDF demo using Bedrock, RAG etc.")

if __name__ == "__main__":
    main()