# Chat with PDF - Generative AI Application. 

## Libraries/Modules/Frameworks used. 
1. Amazon Bedrock, Langchain, Streamlit, S3 Bucket, Docker, Titan Models, Python SDK. 
2. Streamlit is an open-source Python library designed to simplify the creation and sharing of custom web applications, particularly for machine learning, data science, and general data analysis. It allows users to transform Python scripts into interactive web apps with minimal code, eliminating the need for extensive knowledge of front-end technologies like HTML, CSS, or JavaScript. 
3. Langchain Framework is the framework designed to simplify the creation and sharing of custom web applications powered by Large Language Models (LLMs). It provides tools and components that allow developers to build, integrate and deploy AI systems, capable of reasoning, retrieval, and dynamic interaction with external data sources. 
4. RecursiveCharacterTextSplitter is needed because when the statement or a paragraph ends it might be possible that the next paragraph also has some related context therefore we use this module.
5. FAISS is Facebook AI Similarlity Search which helps storing the vector embeddings in the Vector Store such that when user asks the question AI model conevrts the question into vector embedding and search the similar vector embedding into it's vector store. This is called Similarity Search. 

## Models used:
1. Amazon Titan Embedding G1 - Text
2. Anthropic Claude 2.1

## Architecture
![image info](./Bedrock-ChatWithPdf.png)

### ADMIN Application:
   - Build Admin Web application where AdminUser can upload the pdf.
   - The PDF text is split into chunks
   - Using the Amazon Titan Embedding Model, create the vector representation of the chunks
   - Using FAISS, save the vector index locally
   - Upload the index to Amazon S3 bucket (You can use other vector stores like OpenSearch, Pinecone, PgVector etc., but for this demo, I chose cost effective S3)
    
### USER Application:
  - Build User Web application where users can query / chat with the pdf.
  - At the application start, download the index files from S3 to build local FAISS index (vector store)
  - Langchain's RetrievalQA, does the following:
     - Convert the User's query to vector embedding using Amazon Titan Embedding Model (Make sure to use the same model that was used for creating the chunk's embedding on the Admin side)
    - Do similarity search to the FAISS index and retrieve 5 relevant documents pertaining to the user query to build the context
    - Using Prompt template, provide the question and context to the Large Language Model. We are using Claude model from Anthropic.
   -  Display the LLM's response to the user.

## Developer Instruction. 
1. You have to host your Admin Site to upload the PDF in S3 Bucket. 
2. You have to host your User Site to chat with PDF.
   
## Contributor Expectation. 
1. In case of any bug or any enhancement points. Please mailto: ashutosh1.kumar.ara@gmail.com
