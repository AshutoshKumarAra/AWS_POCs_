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
