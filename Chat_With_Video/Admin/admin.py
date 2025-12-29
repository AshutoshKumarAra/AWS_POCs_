# Simple Streamlit app to upload video files to AWS S3
'''
import streamlit as st
import boto3
import os

# AWS S3 Config
BUCKET_NAME = os.getenv("BUCKET_NAME", "bedrock-chat-with-video")
REGION = os.getenv("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=REGION)

st.title("Upload Video to S3")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("Upload to S3"):
        s3.upload_fileobj(uploaded_file, BUCKET_NAME, uploaded_file.name)
        st.success(f"Uploaded `{uploaded_file.name}` to S3 bucket `{BUCKET_NAME}`")
'''
# Chat with Video RAG Application.
import os
import uuid
import time
import json
import tempfile
import threading
import gzip
import shutil
import zlib
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

# LangChain / Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -----------------------------
# AWS setup
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "bedrock-chat-with-video")

s3_client = boto3.client("s3", region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Multipart upload config (5 MB chunks, 10 threads)
transfer_config = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,
    multipart_chunksize=5 * 1024 * 1024,
    max_concurrency=10,
    use_threads=True
)

# -----------------------------
# Helpers
# -----------------------------
def get_unique_id() -> str:
    return str(uuid.uuid4())

def warmup_bedrock():
    try:
        bedrock_embeddings.embed_query("warmup")
    except Exception:
        pass

def adaptive_chunk_size(text: str) -> int:
    length = len(text)
    if length < 200:
        return 1200
    elif length < 1000:
        return 800
    else:
        return 600

def split_transcript_text(transcript_text: str, chunk_overlap=200):
    base_doc = Document(page_content=transcript_text, metadata={"source": "transcript"})
    chunk_size = adaptive_chunk_size(transcript_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents([base_doc])
    for i, d in enumerate(docs, start=1):
        d.metadata["segment_number"] = i
    return docs

# -----------------------------
# Embedding with throttling backoff
# -----------------------------
def safe_embed(batch, retries=10):
    for i in range(retries):
        try:
            return FAISS.from_documents(batch, bedrock_embeddings)
        except ValueError as e:
            msg = str(e)
            if "ThrottlingException" in msg or "Too many requests" in msg:
                wait = 2 ** i + random.random()
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for embedding batch")

def batch_embed_parallel(documents, batch_size=50, max_workers=1, progress=None):
    if len(documents) == 0:
        raise ValueError("No documents to embed.")
    warmup_bedrock()

    total = len(documents)
    batches = [documents[i:i+batch_size] for i in range(0, total, batch_size)]
    vectorstore = None

    try:
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
                    progress.progress(pct, text=f"Embedded {processed}/{total} segments…")
                time.sleep(0.2)
        return vectorstore
    except RuntimeError as e:
        if "Max retries exceeded" in str(e):
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
                    progress.progress(pct, text=f"Embedded {processed}/{total} segments (sequential)…")
                time.sleep(1.0)
            return vectorstore
        else:
            raise

# -----------------------------
# Compression
# -----------------------------
def compress_index_files(faiss_file, pkl_file):
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
# S3 uploads (background)
# -----------------------------
def upload_index_to_s3_background(faiss_gz, pkl_gz, bucket_name, key_prefix, transfer_config):
    def _upload():
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(
                    s3_client.upload_file,
                    Filename=faiss_gz,
                    Bucket=bucket_name,
                    Key=f"{key_prefix}.faiss.gz",
                    Config=transfer_config,
                )
                f2 = executor.submit(
                    s3_client.upload_file,
                    Filename=pkl_gz,
                    Bucket=bucket_name,
                    Key=f"{key_prefix}.pkl.gz",
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

# -----------------------------
# Transcribe helpers
# -----------------------------
VIDEO_EXT_TO_FORMAT = {
    "mp4": "mp4",
    "mov": "mov",
    "avi": "avi",
}

def upload_video_to_s3(tmp_path: str, bucket: str, key_prefix: str) -> str:
    key = f"{key_prefix}/{Path(tmp_path).name}"
    # Use multipart upload for large files
    s3_client.upload_file(tmp_path, bucket, key, Config=transfer_config)
    return f"s3://{bucket}/{key}"

def start_transcribe_job(media_uri: str, job_name: str, media_format: str, language_code: str = "en-US"):
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat=media_format,
        LanguageCode=language_code,
        OutputBucketName=BUCKET_NAME,
    )

def wait_for_transcribe(job_name: str, poll_seconds: int = 5):
    while True:
        resp = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            return resp
        time.sleep(poll_seconds)

def fetch_transcript_text(resp) -> str:
    tj = resp["TranscriptionJob"]
    if tj["TranscriptionJobStatus"] != "COMPLETED":
        raise RuntimeError(f"Transcribe failed: {tj.get('FailureReason', 'unknown reason')}")
    uri = tj["Transcript"]["TranscriptFileUri"]
    if uri.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(uri) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data["results"]["transcripts"][0]["transcript"]
    else:
        parts = uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data["results"]["transcripts"][0]["transcript"]

# -----------------------------
# Streamlit App – Two-step workflow
# -----------------------------
# -----------------------------
# Streamlit App – Two-step workflow
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with Video – Admin", page_icon="🎬", layout="centered")
    st.title("🎬 Chat with Video – Admin (Resilient Two-step)")
    st.caption("Step 1: Upload video to S3 and transcribe. Step 2: Chunk, embed with backoff, build FAISS, compress & upload.")

    if "request_id" not in st.session_state:
        st.session_state.request_id = None
    if "tmp_video_path" not in st.session_state:
        st.session_state.tmp_video_path = None
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = None
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None

    # -----------------------------
    # Step 1: Upload + Transcribe
    # -----------------------------
    st.subheader("Step 1 — Upload video and transcribe to text")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mpeg4"],
        help="Supports files up to ~1GB"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        do_upload_btn = st.button("Save & Upload to S3", type="primary")
    with col2:
        transcribe_btn = st.button("Start Transcribe")
    with col3:
        reset_btn = st.button("Reset session")

    if reset_btn:
        try:
            if st.session_state.tmp_video_path and os.path.exists(st.session_state.tmp_video_path):
                os.unlink(st.session_state.tmp_video_path)
        except Exception:
            pass
        st.session_state.request_id = None
        st.session_state.tmp_video_path = None
        st.session_state.transcript_text = None
        st.session_state.upload_status = None
        st.success("🔄 Session reset.")

    if do_upload_btn:
        if uploaded_file is None:
            st.error("Please upload a video file first.")
        elif not BUCKET_NAME:
            st.error("BUCKET_NAME is not set in environment.")
        else:
            request_id = get_unique_id()
            st.session_state.request_id = request_id
            # Stream directly to S3 without buffering entire file
            key = f"videos/{request_id}/{uploaded_file.name}"
            with st.spinner("⬆️ Uploading video to S3…"):
                s3_client.upload_fileobj(
                    Fileobj=uploaded_file,
                    Bucket=BUCKET_NAME,
                    Key=key,
                    Config=transfer_config
                )
            st.success(f"✅ Uploaded to S3: {s3_uri}")
            st.info(f"Request ID: {st.session_state.request_id}")

    if transcribe_btn:
        if not st.session_state.tmp_video_path or not st.session_state.request_id:
            st.error("Please complete upload first.")
        else:
            ext = Path(st.session_state.tmp_video_path).suffix.replace(".", "").lower()
            media_format = VIDEO_EXT_TO_FORMAT.get(ext, "mp4")
            job_name = f"vidrag-{st.session_state.request_id}"
            s3_uri = f"s3://{BUCKET_NAME}/videos/{st.session_state.request_id}/{Path(st.session_state.tmp_video_path).name}"

            with st.status("🗣️ Starting AWS Transcribe job…", expanded=True) as status:
                try:
                    start_transcribe_job(s3_uri, job_name, media_format)
                    status.update(label="⏳ Waiting for transcription to complete…")
                    resp = wait_for_transcribe(job_name)
                    transcript_text = fetch_transcript_text(resp)
                    st.session_state.transcript_text = transcript_text
                    status.update(label=f"✅ Transcription ready ({len(transcript_text)} chars).", state="complete")
                    st.success("Transcript fetched successfully.")
                except ClientError as e:
                    st.error(f"Transcribe error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    if st.session_state.transcript_text:
        with st.expander("Preview transcript", expanded=False):
            st.text_area("Transcript", st.session_state.transcript_text[:10000], height=200)

    # -----------------------------
    # Step 2: Process transcript → FAISS → compress → upload
    # -----------------------------
    st.subheader("Step 2 — Build embeddings and upload FAISS index")
    process_btn = st.button("Chunk, Embed, Build & Upload", type="secondary")

    if process_btn:
        if not st.session_state.transcript_text or not st.session_state.request_id:
            st.error("Please complete Step 1 (Transcription) first.")
            return

        progress = st.progress(0, text="Starting processing…")
        try:
            progress.progress(10, text="Splitting transcript into segments…")
            docs = split_transcript_text(st.session_state.transcript_text, chunk_overlap=200)
            st.write(f"Total segments: {len(docs)}")

            progress.progress(20, text="Embedding segments with backoff…")
            vectorstore = batch_embed_parallel(
                documents=docs,
                batch_size=50,
                max_workers=1,
                progress=progress
            )

            progress.progress(50, text="Saving FAISS index locally…")
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_name = f"{st.session_state.request_id}.bin"
                vectorstore.save_local(index_name=file_name, folder_path=tmp_dir)

                faiss_file = os.path.join(tmp_dir, f"{file_name}.faiss")
                pkl_file = os.path.join(tmp_dir, f"{file_name}.pkl")

                progress.progress(60, text="Compressing index files (zlib/gzip)…")
                faiss_gz, pkl_gz = compress_index_files(faiss_file, pkl_file)

                key_prefix = f"faiss/video/{st.session_state.request_id}"
                progress.progress(75, text="Uploading FAISS index to S3 in background…")
                upload_thread = upload_index_to_s3_background(
                    faiss_gz, pkl_gz, BUCKET_NAME, key_prefix, transfer_config
                )
                upload_thread.join()

                if st.session_state.upload_status == "success":
                    progress.progress(100, text="Done.")
                    st.success(f"✅ Uploaded FAISS index to s3://{BUCKET_NAME}/{key_prefix}.{{faiss.gz|pkl.gz}}")
                else:
                    st.error(st.session_state.upload_status or "Upload status unknown.")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()