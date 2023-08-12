FROM python:3.10.9

WORKDIR /polito_llama2
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY sentiment_classification/SFT-paral ./sentiment_classification/SFT-paral

# ARG HUGGINGFACE_TOKEN
# RUN huggingface-cli login --token $HUGGINGFACE_TOKEN
