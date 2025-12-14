'''
#-------------------This is the basic code-----------------------------------------
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

## Split the pages/texts into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## Create Vector Store
def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## Upload to S3. 
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

## Main method 
def main():
    st.write("This is admin site for chat with pdf demo.")
    uploaded_file = st.file_uploader("Choose a file","pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs Length: {len(splitted_docs)}")
        st.write("=====================")
        st.write(splitted_docs[0])
        st.write("=====================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id,splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully.")
        else:
            st.write("Error!! Please check logs.")

if __name__ == "__main__":
    main()
'''
# ------------------This is my advance code---------------------------------
import os
import uuid
import tempfile
import threading
import gzip
import shutil
import zlib
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import streamlit as st
from boto3.s3.transfer import TransferConfig

# -----------------------------
# AWS setup
# -----------------------------
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")  # ensure this is set in environment

# Bedrock + LangChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


# -----------------------------
# Helpers
# -----------------------------
def get_unique_id() -> str:
    return str(uuid.uuid4())

def load_pdf_lazily(file_path):
    """Load pages lazily for better memory management."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def adaptive_chunk_size(text: str) -> int:
    """Adaptive chunking based on page text density."""
    length = len(text)
    if length < 200:
        return 1200
    elif length < 1000:
        return 800
    else:
        return 600

def split_text(pages, chunk_overlap=200):
    """Split documents adaptively; skip non-text pages, preserve page_number."""
    docs = []
    for page in pages:
        content = page.page_content.strip()
        if len(content) < 50:  # skip blank/image-only pages
            continue
        chunk_size = adaptive_chunk_size(content)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        page_docs = splitter.split_documents([page])
        for doc in page_docs:
            if "page" in doc.metadata:
                doc.metadata["page_number"] = int(doc.metadata["page"]) + 1
        docs.extend(page_docs)
    return docs

def warmup_bedrock():
    """Warm-up Bedrock client to avoid cold start latency."""
    try:
        bedrock_embeddings.embed_query("warmup")
    except Exception:
        pass

# -----------------------------
# Embedding with throttling backoff
# -----------------------------

def safe_embed(batch, retries=10):
    """Embed a batch with exponential backoff on throttling."""
    for i in range(retries):
        try:
            return FAISS.from_documents(batch, bedrock_embeddings)
        except ValueError as e:
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                wait = 2 ** i + random.random()
                time.sleep(wait)
            else:
                raise
    # If we exhausted retries, bubble up
    raise RuntimeError("Max retries exceeded for embedding batch")


def batch_embed_parallel(documents, batch_size=100, max_workers=2, progress=None):
    """
    Embed chunks in batches with adjustable concurrency and backoff.
    If throttling persists, automatically fall back to sequential mode.
    """
    if len(documents) == 0:
        raise ValueError("No documents to embed.")
    warmup_bedrock()

    total = len(documents)
    batches = [documents[i:i+batch_size] for i in range(0, total, batch_size)]
    vectorstore = None

    try:
        # Try parallel first
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(safe_embed, batch): len(batch) for batch in batches}
            processed = 0
            for future in as_completed(futures):
                batch_vs = future.result()
                if vectorstore is None:
                    vectorstore = batch_vs
                else:
                    vectorstore.merge_from(batch_vs)
                processed += futures[future]
                pct = int((processed / total) * 40)
                if progress:
                    progress.progress(pct, text=f"Embedded {processed}/{total} chunks (parallel)…")
                time.sleep(0.2)
        return vectorstore

    except RuntimeError as e:
        if "Max retries exceeded" in str(e):
            # Fallback: sequential embedding
            if progress:
                progress.progress(20, text="⚠️ Throttling detected, switching to sequential embedding…")
            vectorstore = None
            processed = 0
            for batch in batches:
                batch_vs = safe_embed(batch, retries=10)
                if vectorstore is None:
                    vectorstore = batch_vs
                else:
                    vectorstore.merge_from(batch_vs)
                processed += len(batch)
                pct = int((processed / total) * 40)
                if progress:
                    progress.progress(pct, text=f"Embedded {processed}/{total} chunks (sequential)…")
                time.sleep(1.0)  # slower pacing to avoid throttling
            return vectorstore
        else:
            raise

# -----------------------------
# Compression
# -----------------------------
def compress_index_files(faiss_file, pkl_file):
    """Compress FAISS and PKL; prefer zlib for speed, fallback to gzip."""
    faiss_gz = faiss_file + ".gz"
    pkl_gz = pkl_file + ".gz"
    try:
        with open(faiss_file, "rb") as f_in, open(faiss_gz, "wb") as f_out:
            f_out.write(zlib.compress(f_in.read()))
        with open(pkl_file, "rb") as f_in, open(pkl_gz, "wb") as f_out:
            f_out.write(zlib.compress(f_in.read()))
    except Exception:
        with open(faiss_file, "rb") as f_in, gzip.open(faiss_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        with open(pkl_file, "rb") as f_in, gzip.open(pkl_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return faiss_gz, pkl_gz

# -----------------------------
# Upload + Images (background)
# -----------------------------

def upload_to_s3_background(faiss_gz, pkl_gz, bucket_name, transfer_config):
    def _upload():
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(
                    s3_client.upload_file,
                    Filename=faiss_gz,
                    Bucket=bucket_name,
                    Key="my_faiss.faiss.gz",
                    Config=transfer_config,
                )
                f2 = executor.submit(
                    s3_client.upload_file,
                    Filename=pkl_gz,
                    Bucket=bucket_name,
                    Key="my_faiss.pkl.gz",
                    Config=transfer_config,
                )
                f1.result()
                f2.result()
            st.session_state.upload_status = "success"
        except Exception as e:
            st.session_state.upload_status = f"error: {e}"

    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()
    return thread

def extract_and_upload_images_async(pdf_path, s3_prefix="pdf_images/", status_area=None):
    """Extract/upload images concurrently to reduce wall-clock time."""
    def _extract():
        import fitz
        try:
            resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix)
            if "Contents" in resp:
                for obj in resp["Contents"]:
                    s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
            if status_area:
                status_area.write("🧹 Cleared old images in S3 prefix.")
        except Exception:
            pass

        doc = fitz.open(pdf_path)
        uploaded = 0
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:
                    img_bytes = pix.tobytes("png")
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                pix = None
                key = f"{s3_prefix}page{page_num+1}_img{img_index+1}.png"
                s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=img_bytes)
                uploaded += 1
        if status_area:
            status_area.success(f"🖼️ Uploaded {uploaded} images to S3.")

    thread = threading.Thread(target=_extract, daemon=True)
    thread.start()
    return thread


# -----------------------------
# Streamlit App – Two-step workflow
# -----------------------------
def main():
    st.title("📄 Chat with PDF – Admin (Resilient Two-step)")
    st.caption("Step 1: Save & load pages lazily. Step 2: Split, embed with backoff, build FAISS, compress & upload, extract images concurrently.")

    # Session state keys
    if "request_id" not in st.session_state:
        st.session_state.request_id = None
    if "tmp_pdf_path" not in st.session_state:
        st.session_state.tmp_pdf_path = None
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None
    if "image_status" not in st.session_state:
        st.session_state.image_status = None

    # -----------------------------
    # Step 1: Upload + Save + Lazy Load
    # -----------------------------
    st.subheader("Step 1 — Upload and load")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Max size: 100MB")

    col1, col2 = st.columns(2)
    with col1:
        load_button = st.button("Save & Load PDF", type="primary")
    with col2:
        reset_button = st.button("Reset session")

    if reset_button:
        try:
            if st.session_state.tmp_pdf_path and os.path.exists(st.session_state.tmp_pdf_path):
                os.unlink(st.session_state.tmp_pdf_path)
        except Exception:
            pass
        st.session_state.request_id = None
        st.session_state.tmp_pdf_path = None
        st.session_state.upload_status = None
        st.session_state.image_status = None
        st.success("🔄 Session reset.")

    if load_button:
        if uploaded_file is None:
            st.error("Please upload a PDF file first.")
        elif not BUCKET_NAME:
            st.error("BUCKET_NAME is not set in environment.")
        else:
            request_id = get_unique_id()
            st.session_state.request_id = request_id
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.tmp_pdf_path = tmp_file.name

            with st.status("📖 Loading PDF pages...", expanded=True) as status:
                pages = load_pdf_lazily(st.session_state.tmp_pdf_path)
                status.update(label=f"✅ Loaded {len(pages)} pages", state="complete")
            st.success(f"Saved temp file: {st.session_state.tmp_pdf_path}")
            st.info(f"Request ID: {st.session_state.request_id}")

    # -----------------------------
    # Step 2: Process (split, embed, build FAISS, compress, upload, images, cleanup)
    # -----------------------------
    st.subheader("Step 2 — Process and upload")
    process_button = st.button("Split, Embed, Build & Upload", type="secondary")

    if process_button:
        if not st.session_state.tmp_pdf_path or not st.session_state.request_id:
            st.error("Please complete Step 1 first (Save & Load PDF).")
            return

        progress = st.progress(0, text="Starting processing…")
        status_area = st.empty()

        try:
            # Reload pages lazily
            progress.progress(5, text="Reloading pages lazily…")
            pages = load_pdf_lazily(st.session_state.tmp_pdf_path)
            st.write(f"Total pages: {len(pages)}")

            # Split into adaptive chunks
            progress.progress(15, text="Splitting into adaptive chunks…")
            docs = split_text(pages, chunk_overlap=200)
            st.write(f"Total chunks (after skipping low-text pages): {len(docs)}")

            # Embedding with backoff
            progress.progress(20, text="Embedding in batches with backoff…")
            vectorstore = batch_embed_parallel(
                documents=docs,
                batch_size=50,   # safer batch size
                max_workers=1,   # sequential to avoid throttling
                progress=progress
            )

            # Save locally
            progress.progress(45, text="Saving FAISS index locally…")
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_name = f"{st.session_state.request_id}.bin"
                vectorstore.save_local(index_name=file_name, folder_path=tmp_dir)
                faiss_file = os.path.join(tmp_dir, f"{file_name}.faiss")
                pkl_file = os.path.join(tmp_dir, f"{file_name}.pkl")

                # Compress before upload
                progress.progress(55, text="Compressing index files (zlib/gzip)…")
                faiss_gz, pkl_gz = compress_index_files(faiss_file, pkl_file)

                # Multipart config
                transfer_config = TransferConfig(
                    multipart_threshold=5 * 1024 * 1024,
                    max_concurrency=10,
                    multipart_chunksize=5 * 1024 * 1024,
                    use_threads=True,
                )

                # Upload in background
                progress.progress(65, text="Starting background upload to S3…")
                upload_thread = upload_to_s3_background(
                    faiss_gz, pkl_gz, BUCKET_NAME, transfer_config
                )

                # Extract and upload images concurrently
                progress.progress(85, text="Extracting and uploading images in parallel…")
                images_thread = extract_and_upload_images_async(
                    st.session_state.tmp_pdf_path, s3_prefix="pdf_images/"
                )

                progress.progress(95, text="Finishing up…")
                st.success("🎉 Processing started! Upload and image extraction continue in background.")

                # Wait for threads before cleanup
                upload_thread.join()
                images_thread.join()

                # Show results
                if st.session_state.upload_status == "success":
                    status_area.success("✅ Uploads completed.")
                elif st.session_state.upload_status and "error" in st.session_state.upload_status:
                    status_area.error(st.session_state.upload_status)

                if st.session_state.image_status and "success" in st.session_state.image_status:
                    status_area.success(f"🖼️ {st.session_state.image_status}")
                elif st.session_state.image_status and "error" in st.session_state.image_status:
                    status_area.error(st.session_state.image_status)

            # Cleanup temp file
            try:
                if st.session_state.tmp_pdf_path and os.path.exists(st.session_state.tmp_pdf_path):
                    os.unlink(st.session_state.tmp_pdf_path)
                st.session_state.tmp_pdf_path = None
            except Exception:
                pass

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
