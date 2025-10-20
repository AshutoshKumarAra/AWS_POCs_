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

## For RAG - Using modern approach without RetrievalQA
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

## Get LLMs
def get_llm():
    llm = BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            'maxTokenCount': 2048,
            'temperature': 0.2,
            'topP': 0.9,
            'stopSequences': []
        }
    )
    return llm

## Get Response - BALANCED APPROACH
def get_response(llm, vectorstore, question):
    ## Create Prompt / Template - STRICT BUT REASONABLE
    prompt_template = """You are a helpful assistant that answers questions based on the provided context from a PDF document.

Context from PDF Document:
{context}

Question: {question}

IMPORTANT INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above
- If the context contains relevant information, provide a detailed and comprehensive answer
- If the context does NOT contain enough information to answer the question, respond EXACTLY with: "OUT_OF_CONTEXT"
- Do not use external knowledge - only use what is explicitly stated in the context
- Be thorough in your answer if the information is available in the context

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Get retriever - removed threshold for better recall
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    
    # Get relevant documents using invoke
    docs = retriever.invoke(question)
    
    # Check if any documents were retrieved
    if not docs or len(docs) == 0:
        return "I cannot answer this question as it is not covered in the uploaded PDF document. Please ask questions related to the document content."
    
    # Format context from documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(doc.page_content)
    
    context = "\n\n".join(context_parts)
    
    # Create the prompt with context and question
    formatted_prompt = PROMPT.format(context=context, question=question)
    
    # Get response from LLM
    answer = llm.invoke(formatted_prompt)
    
    # Check if the model explicitly said OUT_OF_CONTEXT
    if "OUT_OF_CONTEXT" in answer:
        return "I cannot answer this question as it is not covered in the uploaded PDF document. Please ask questions related to the document content."
    
    return answer

## Main method
def main():
    st.write("Chat with PDF - Ask questions about your uploaded PDF document")
    st.write("---")

    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files loaded from S3: {len([f for f in dir_list if f.endswith('.faiss') or f.endswith('.pkl')])} files")
    
    ## Create Index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("✅ PDF Index is ready! You can now ask questions about your document.")
    st.write("---")
    
    question = st.text_input("Please ask your question about the PDF document:")
    
    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Analyzing document and generating answer..."):
                llm = get_llm()
                response = get_response(llm, faiss_index, question)
                
                # Display the response with appropriate styling
                if "cannot answer this question" in response.lower():
                    st.warning(response)
                else:
                    st.success("Answer:")
                    st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()