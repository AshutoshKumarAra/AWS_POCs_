'''
#--------------------------User Site only for PDF---------------------------
import boto3
import streamlit as st
import os
import uuid
import gzip,shutil

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
    #s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    #s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

    # Download compressed files from S3. 
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss.gz", Filename=f"{folder_path}my_faiss.faiss.gz")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl.gz", Filename=f"{folder_path}my_faiss.pkl.gz")

    # Decompress to usable FAISS + PKL
    with gzip.open(f"{folder_path}my_faiss.faiss.gz", "rb") as f_in, open(f"{folder_path}my_faiss.faiss", "wb") as f_out:
        shutil.copyfileobj(f_in,f_out)

    with gzip.open(f"{folder_path}my_faiss.pkl.gz", "rb") as f_in, open(f"{folder_path}my_faiss.pkl", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def get_llm():
    return BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            "maxTokenCount": 3072,
            "temperature": 0.2,   # stricter, closer to context
            "topP": 0.9,
            "stopSequences": []
        }
    )

# -----------------------------
# Image utilities (strict filtering)
# -----------------------------
def list_images_for_pages(pages, s3_prefix="pdf_images/"):
    if not pages:
        return []
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix)
    except Exception:
        return []
    keys = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            for p in pages:
                token = f"page{p}_"
                if token in key:
                    keys.append(key)
                    break
    return keys

# -----------------------------
# RAG chain
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
- Keep wording faithful to the PDF. Prefer short quotes from the context when possible.
- Summarize minimally and clearly; do not invent beyond the provided text.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def rag_chain_fn(inputs):
        question = inputs["question"].strip()
        if not question:
            return "OUT_OF_CONTEXT"

        docs = retriever.invoke(question)
        if not docs or len(docs) == 0:
            return "OUT_OF_CONTEXT"

        labeled_chunks = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                labeled_chunks.append(f"Chunk {i+1}:\n{content}")
        context = "\n\n".join(labeled_chunks).strip()

        if not context:
            return "OUT_OF_CONTEXT"

        formatted_prompt = PROMPT.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt).strip()

        if answer == "OUT_OF_CONTEXT" or "OUT_OF_CONTEXT" in answer:
            return "OUT_OF_CONTEXT"

        return answer

    return rag_chain_fn

# -----------------------------
# Streamlit app (UI unchanged)
# -----------------------------
def main():
    st.title("📄 Chat with PDF")
    st.write("Ask questions about your uploaded PDF. Answers are synthesized from retrieved context only. Outside questions return Out of Context.")
    st.write("---")

    try:
        load_index()
    except Exception as e:
        st.error(f"Failed to load FAISS index from S3. Check BUCKET_NAME and object keys.{e}")
        return

    try:
        faiss_index = FAISS.load_local(
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            index_name="my_faiss",
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load local FAISS index. Verify files exist in /tmp and index_name matches.{e}")
        return

    st.success("✅ PDF Index is ready! You can now ask questions.")
    st.write("---")

    question = st.text_input("Please ask your question about the PDF document:")
    show_chunks = st.checkbox("Show retrieved chunks")

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Analyzing document and generating answer..."):
                llm = get_llm()
                rag_chain = build_rag_chain(llm, faiss_index)

                # Retrieve docs
                docs_preview = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(question.strip())

                # Show retrieved chunks
                if show_chunks:
                    debug_text = "\n\n---\n\n".join([f"Chunk {i+1}:\n{d.page_content[:1000]}" for i, d in enumerate(docs_preview)])
                    st.expander("🔍 Retrieved Chunks").write(debug_text if debug_text else "No chunks retrieved.")

                # Get answer
                response = rag_chain({"question": question})

                if response.strip() == "OUT_OF_CONTEXT":
                    st.warning("I cannot answer this question as it is not covered in the uploaded PDF document.")
                    return

                st.success("Answer:")
                st.write(response)

                # Collect relevant page numbers
                relevant_pages = set()
                for d in docs_preview:
                    if "page_number" in d.metadata and isinstance(d.metadata["page_number"], int):
                        relevant_pages.add(d.metadata["page_number"])
                    elif "page" in d.metadata and isinstance(d.metadata["page"], int):
                        relevant_pages.add(d.metadata["page"] + 1)

                # Show only images for those pages
                image_keys = list_images_for_pages(relevant_pages)
                if image_keys:
                    st.info("📷 Relevant images from the PDF:")
                    for key in image_keys:
                        try:
                            url = s3_client.generate_presigned_url(
                                "get_object",
                                Params={"Bucket": BUCKET_NAME, "Key": key},
                                ExpiresIn=3600
                            )
                            st.image(url, caption=key)
                        except Exception:
                            pass
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
'''

# ----------------------------User Site for Advance Code--------------
import boto3
import streamlit as st
import os
import uuid
import gzip, shutil, zlib

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
    st.info("Downloading FAISS index files from S3.")

    # Download compressed files from S3
    try:
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss.gz", Filename=f"{folder_path}my_faiss.faiss.gz")
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl.gz", Filename=f"{folder_path}my_faiss.pkl.gz")
        st.success("Downloaded compressed FAISS files from S3.")
    except Exception as e:
        st.error(f"Failed to download FAISS files from S3: {e}")
        raise

    # Decompress FAISS
    try:
        with open(f"{folder_path}my_faiss.faiss.gz", "rb") as f_in, open(f"{folder_path}my_faiss.faiss", "wb") as f_out:
            f_out.write(zlib.decompress(f_in.read()))
        st.info("Decompressed FAISS file using zlib.")
    except Exception as e:
        st.warning(f"zlib decompression failed for FAISS: {e}. Trying gzip...")
        with gzip.open(f"{folder_path}my_faiss.faiss.gz", "rb") as f_in, open(f"{folder_path}my_faiss.faiss", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        st.info("Decompressed FAISS file using gzip.")

    # Decompress PKL
    try:
        with open(f"{folder_path}my_faiss.pkl.gz", "rb") as f_in, open(f"{folder_path}my_faiss.pkl", "wb") as f_out:
            f_out.write(zlib.decompress(f_in.read()))
        st.info("Decompressed PKL file using zlib.")
    except Exception as e:
        st.warning(f"zlib decompression failed for PKL: {e}. Trying gzip...")
        with gzip.open(f"{folder_path}my_faiss.pkl.gz", "rb") as f_in, open(f"{folder_path}my_faiss.pkl", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        st.info("Decompressed PKL file using gzip.")

def get_llm():
    return BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            "maxTokenCount": 3072,
            "temperature": 0.2,
            "topP": 0.9,
            "stopSequences": []
        }
    )

# -----------------------------
# Image utilities
# -----------------------------
def list_images_for_pages(pages, s3_prefix="pdf_images/"):
    if not pages:
        return []
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix)
    except Exception as e:
        st.warning(f"⚠️ Failed to list images from S3: {e}")
        return []
    keys = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            for p in pages:
                token = f"page{p}_"
                if token in key:
                    keys.append(key)
                    break
    return keys

# -----------------------------
# RAG chain
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
- Keep wording faithful to the PDF. Prefer short quotes from the context when possible.
- Summarize minimally and clearly; do not invent beyond the provided text.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def rag_chain_fn(inputs):
        question = inputs["question"].strip()
        if not question:
            return "OUT_OF_CONTEXT"

        docs = retriever.invoke(question)
        if not docs or len(docs) == 0:
            return "OUT_OF_CONTEXT"

        labeled_chunks = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                labeled_chunks.append(f"Chunk {i+1}:\n{content}")
        context = "\n\n".join(labeled_chunks).strip()

        if not context:
            return "OUT_OF_CONTEXT"

        formatted_prompt = PROMPT.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt).strip()

        if answer == "OUT_OF_CONTEXT" or "OUT_OF_CONTEXT" in answer:
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

    try:
        load_index()
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index from S3: {e}")
        return

    try:
        faiss_index = FAISS.load_local(
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            index_name="my_faiss",
            allow_dangerous_deserialization=True
        )
        st.success("✅ PDF Index is ready! You can now ask questions.")
    except Exception as e:
        st.error(f"❌ Failed to load local FAISS index: {e}")
        return

    st.write("---")

    # Initialize session state keys
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "docs_preview" not in st.session_state:
        st.session_state.docs_preview = []
    if "images" not in st.session_state:
        st.session_state.images = []

    # Input + Reset
    question = st.text_input("Please ask your question about the PDF document:", value=st.session_state.question)
    show_chunks = st.checkbox("Show retrieved chunks")

    col1, col2 = st.columns(2)
    with col1:
        ask_button = st.button("Ask Question")
    with col2:
        reset_button = st.button("Reset")

    # Reset clears everything
    if reset_button:
        st.session_state.question = ""
        st.session_state.response = ""
        st.session_state.docs_preview = []
        st.session_state.images = []
        st.success("🔄 Cleared previous question and results.")

    # Ask Question
    if ask_button:
        if question.strip():
            st.session_state.question = question.strip()
            with st.spinner("Analyzing document and generating answer..."):
                llm = get_llm()
                rag_chain = build_rag_chain(llm, faiss_index)

                docs_preview = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(question.strip())
                st.session_state.docs_preview = docs_preview

                if show_chunks:
                    debug_text = "\n\n---\n\n".join([f"Chunk {i+1}:\n{d.page_content[:1000]}" for i, d in enumerate(docs_preview)])
                    st.expander("🔍 Retrieved Chunks").write(debug_text if debug_text else "No chunks retrieved.")

                response = rag_chain({"question": question})
                st.session_state.response = response

                if response.strip() == "OUT_OF_CONTEXT":
                    st.warning("I cannot answer this question as it is not covered in the uploaded PDF document.")
                else:
                    st.success("Answer:")
                    st.write(response)

                # Collect relevant page numbers
                relevant_pages = set()
                for d in docs_preview:
                    if "page_number" in d.metadata and isinstance(d.metadata["page_number"], int):
                        relevant_pages.add(d.metadata["page_number"])
                    elif "page" in d.metadata and isinstance(d.metadata["page"], int):
                        relevant_pages.add(d.metadata["page"] + 1)

                image_keys = list_images_for_pages(relevant_pages)
                st.session_state.images = image_keys

                if image_keys:
                    st.info("📷 Relevant images from the PDF:")
                    for key in image_keys:
                        try:
                            url = s3_client.generate_presigned_url(
                                "get_object",
                                Params={"Bucket": BUCKET_NAME, "Key": key},
                                ExpiresIn=3600
                            )
                            st.image(url, caption=key)
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load image {key}: {e}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
