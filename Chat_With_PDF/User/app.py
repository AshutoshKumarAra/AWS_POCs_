
import boto3
import streamlit as st
import os
import uuid

## S3_Client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock Embeddings
from langchain_aws import BedrockEmbeddings

## Bedrock LLM
from langchain_aws import BedrockLLM

## For RAG
from langchain_core.prompts import PromptTemplate

## Import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

## Unique ID Generation Function
def get_unique_id():
    return str(uuid.uuid4())

## Load Index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

## Get LLM
def get_llm():
    return BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            'maxTokenCount': 2048,
            'temperature': 0.2,
            'topP': 0.9,
            'stopSequences': []
        }
    )

## Get Response
def get_response(llm, vectorstore, question):
    prompt_template = """You are a helpful assistant that answers questions based on the provided context from a PDF document.

Context from PDF Document:
{context}

Question: {question}

Instructions:
- Use ONLY the context above to answer.
- If context is insufficient, reply with: "OUT_OF_CONTEXT".
- Be detailed if information is available.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Proper similarity search
    docs = vectorstore.similarity_search(question, k=5)
    
    if not docs:
        return "I cannot answer this question as it is not covered in the uploaded PDF document."
    
    # Debug: show retrieved chunks
    debug_chunks = "\n---\n".join([d.page_content[:300] for d in docs])
    st.expander("🔍 Retrieved Chunks").write(debug_chunks)
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create final prompt
    formatted_prompt = PROMPT.format(context=context, question=question)
    
    # Get response from LLM
    answer = llm.invoke(formatted_prompt)
    
    if "OUT_OF_CONTEXT" in answer:
        return "I cannot answer this question as it is not covered in the uploaded PDF document."
    
    return answer

## Main method
def main():
    st.title("📄 Chat with PDF")
    st.write("Ask questions about your uploaded PDF document")
    st.write("---")

    load_index()

    # Load FAISS index
    faiss_index = FAISS.load_local(
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        index_name="my_faiss",
        allow_dangerous_deserialization=True
    )

    st.success("✅ PDF Index is ready! You can now ask questions.")
    st.write("---")
    
    question = st.text_input("Please ask your question about the PDF document:")
    
    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Analyzing document and generating answer..."):
                llm = get_llm()
                response = get_response(llm, faiss_index, question)
                
                if "cannot answer this question" in response.lower():
                    st.warning(response)
                else:
                    st.success("Answer:")
                    st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
