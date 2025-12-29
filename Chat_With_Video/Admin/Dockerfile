# Use Python 3.11 base image
#FROM python:3.11

# Expose Streamlit port
#EXPOSE 8083

# Set working directory
#WORKDIR /app

# Copy requirements and install
#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
#COPY . ./

# Run Streamlit app
#ENTRYPOINT ["streamlit", "run", "admin.py", "--server.port=8083", "--server.address=0.0.0.0"]

# Use Python 3.11 base image
FROM python:3.11

# Expose Streamlit port
EXPOSE 8083

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . ./

# Add Streamlit config to allow large uploads (1 GB)
RUN mkdir -p /root/.streamlit
RUN echo "[server]\nmaxUploadSize = 1024" > /root/.streamlit/config.toml

# Run Streamlit app with custom config
ENTRYPOINT ["streamlit", "run", "admin.py", "--server.port=8083", "--server.address=0.0.0.0", "--server.maxUploadSize=1024"]