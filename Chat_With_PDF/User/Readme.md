### Algorithm for User Directory. 

1. Create requirements.txt
2. Create Python App. 
3. Design Dockerfile and build Docker Image. 
4. Access the application from the browser. 
5. Ask a question. 
6. Check the response. 

### Gitbash Commands to build the Docker Image and run the Docker Image. 

1. Build = docker build -t pdf-read-client .
2. Run = docker run -e BUCKET_NAME=bedrok-chat-with-pdf -v ~/.aws:/root/.aws -p 8084:8084 -it pdf-read-client