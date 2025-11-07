import boto3
import streamlit as st
import os
import uuid

## S3 Client
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

# Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    return BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            'maxTokenCount': 2048,
            'temperature': 0.5,   # encourage synthesis
            'topP': 0.9,
            'stopSequences': []
        }
    )

def build_rag_chain(llm, vectorstore):
    prompt_template = """You are a helpful assistant that answers questions based on retrieved context from a PDF.

Context (from PDF):
{context}

Question:
{question}

Instructions:
- Use ONLY the context above to answer.
- If the context does not contain enough information, reply EXACTLY with: "OUT_OF_CONTEXT".
- Do not use external knowledge.
- When answering, synthesize and explain in your own words. Do not just copy text.
- Provide a clear, structured answer if context is available.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def rag_chain_fn(inputs):
        docs = retriever.invoke(inputs["question"])

        # If no docs retrieved → OUT_OF_CONTEXT
        if not docs or len(docs) == 0:
            return "OUT_OF_CONTEXT"

        # Build context string
        context = "\n\n".join([f"Chunk {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

        # Format prompt
        formatted_prompt = PROMPT.format(context=context, question=inputs["question"])

        # Get LLM answer
        answer = llm.invoke(formatted_prompt).strip()

        # Guardrail: if model ignored instructions and hallucinated
        if "OUT_OF_CONTEXT" in answer:
            return "OUT_OF_CONTEXT"

        # Extra check: if answer is too short (likely echo), force OUT_OF_CONTEXT
        if len(answer.split()) < 3:
            return "OUT_OF_CONTEXT"

        return answer

    return rag_chain_fn

def main():
    st.title("📄 Chat with PDF")
    st.write("Ask questions about your uploaded PDF document")
    st.write("---")

    load_index()

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
                rag_chain = build_rag_chain(llm, faiss_index)
                
                response = rag_chain({"question": question})
                
                if response.strip() == "OUT_OF_CONTEXT":
                    st.warning("I cannot answer this question as it is not covered in the uploaded PDF document.")
                else:
                    st.success("Answer:")
                    st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
