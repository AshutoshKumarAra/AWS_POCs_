# Here I will be writing steps on how to connect and consume AWS Services programmatically via Github Codespace. 

## Install and test the aws cli 
 1. Check by tying command 'aws' whether aws command is present or not. Gitub Codespace works on Linux Containers therefore Linux 
    command will be used. 
 2. Download the installer = curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
 3. Unzip the file = unzip awscliv2.zip
 4. Install the unzip file = sudo ./aws/install
 5. aws --version
 ## Connect Github Codespace with IAM User to access AWS.
 1. aws configure
 2. Enter access key ID.
 3. Enter secret access key. 
 4. Enter region name. 
 5. Enter output format.
 ## Test the connection. 
 1. aws sts get-caller-identity