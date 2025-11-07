import boto3
import streamlit as st
import os
import uuid

# -----------------------------
# AWS & LangChain setup
# -----------------------------
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

# -----------------------------
# Utilities
# -----------------------------
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
            "maxTokenCount": 3072,  # allow richer answers
            "temperature": 0.6,     # encourage synthesis, still controlled
            "topP": 0.9,
            "stopSequences": []
        }
    )

# -----------------------------
# RAG chain with strict OOC and richer synthesis
# -----------------------------
def build_rag_chain(llm, vectorstore):
    prompt_template = """You are a helpful assistant that answers questions using only the retrieved context from a PDF.

Context (from PDF):
{context}

Question:
{question}

Instructions:
- Use ONLY the context above to answer; do not use external knowledge.
- If the context does not contain enough information, reply EXACTLY with: "OUT_OF_CONTEXT".
- Summarize the relevant information and connect ideas in your own words.
- Provide a clear, structured response with short sections or bullet points where helpful.
- Avoid copying text verbatim unless quoting is necessary.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def rag_chain_fn(inputs):
        question = inputs["question"].strip()
        if not question:
            return "OUT_OF_CONTEXT"

        # Retrieve relevant docs via Runnable retriever
        docs = retriever.invoke(question)

        # If no docs → strict out of context
        if not docs or len(docs) == 0:
            return "OUT_OF_CONTEXT"

        # Build richer, labeled context to nudge synthesis
        labeled_chunks = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                labeled_chunks.append(f"Chunk {i+1}:\n{content}")
        context = "\n\n".join(labeled_chunks).strip()

        # If context somehow empty → out of context
        if not context:
            return "OUT_OF_CONTEXT"

        # Format prompt and invoke LLM
        formatted_prompt = PROMPT.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt).strip()

        # Guardrails: enforce strict OOC and avoid trivial echoes
        if answer == "OUT_OF_CONTEXT" or "OUT_OF_CONTEXT" in answer:
            return "OUT_OF_CONTEXT"

        # If answer looks too short or copy-like, ask LLM to elaborate within same constraints
        if len(answer.split()) < 10:
            elaboration_prompt = formatted_prompt + "\n\nPlease elaborate with more detail, structure, and clear reasoning while staying within the provided context."
            answer = llm.invoke(elaboration_prompt).strip()
            if answer == "OUT_OF_CONTEXT" or "OUT_OF_CONTEXT" in answer or len(answer.split()) < 10:
                return "OUT_OF_CONTEXT"

        return answer

    return rag_chain_fn

# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.title("📄 Chat with PDF (Strict RAG)")
    st.write("Ask questions about your uploaded PDF. Answers are synthesized from retrieved context only. Outside questions return Out of Context.")
    st.write("---")

    # Load FAISS index from S3 to local
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

    # Optional: toggle to show retrieved chunks for transparency
    show_chunks = st.checkbox("Show retrieved chunks")

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Analyzing document and generating answer..."):
                llm = get_llm()
                rag_chain = build_rag_chain(llm, faiss_index)

                # For optional debug: retrieve first to show chunks
                docs_preview = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(question.strip())
                if show_chunks:
                    debug_text = "\n\n---\n\n".join([f"Chunk {i+1}:\n{d.page_content[:1000]}" for i, d in enumerate(docs_preview)])
                    st.expander("🔍 Retrieved Chunks").write(debug_text if debug_text else "No chunks retrieved.")

                # Invoke RAG chain
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
