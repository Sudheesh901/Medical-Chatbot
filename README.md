# Medical-Chatbot
This project is an AI-powered medical chatbot designed to provide users with instant, reliable medical information and symptom guidance. It leverages NLP and large language models to analyze user queries, suggest possible conditions, offer preventive care tips, and recommend when to consult a professional healthcare provider.


# How to run?

### STEPS?

clone the repository

```bash
git clone https://github.com/Sudheesh901/Medical-Chatbot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -m mdeibot python=3.10 -y
```

```bash
conda activate medibot
```

# STEP 02 - Install requirements

```bash
pip install -r requirements.txt
```

Create a .env file in the rool directory and add API Keys of OpenAI and Pinecone as follows:

```bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
#run the following command to store embeddings to the pinecone vector datbase
python store_index.py
```

```bash
#run the following command to test in web app (flask)
python app.py
```
Now
```
Open up loacalhost:
```
### Techstack
    .Python
    .Langchain
    .Flask
    .GPT
    .Pinecone
    .AWS

# AWS-CICD-Deployment-with-Github-Actions

## 1.Login to AWS console

## 2. Create IAM user for deployment
```bash
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

## 3. Create ECR repo to store/save docker image

```bash
- ECR repo: 782198887741.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:
```bash
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

## 6. Configure EC2 as self-hosted runner:

```bash
setting>actions>runner>new self hosted runner> choose os> then run command one by one
```

## 7. Setup github secrets

```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
PINECONE_API_KEY
OPENAI_API_KEY
```