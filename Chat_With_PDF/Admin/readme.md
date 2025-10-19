### Algorithm for Adimin Directory. 

1. Create Amazon S3 Bucket. 
2. Create requirements.txt file. 
3. Create Python App. 
4. Design Dockerfile and Build Docker Image. 
5. Access the appliation from browser. 
6. Upload a PDF. 
7. Confirm the FAISS vector index files are uploaded to S3. 

### Gitbash Commands to build the Docker Image and run the Docker Image.
1. Build = docker build -t pdf-reader-admin .
2. Run = docker run -e BUCKET_NAME=bedrok-chat-with-pdf -v ~/.aws:/root/.aws -p 8083:8083 -it pdf-reader-admin