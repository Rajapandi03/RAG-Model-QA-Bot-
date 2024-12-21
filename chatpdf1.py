!pip install pinecone-client openai langchain-community PyPDF2

import openai
from pinecone import Pinecone, Index
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pinecone_api_key = "pcsk_5CjrXE_***********" #OPENAI API KEY
environment = "us-east-1"
index_name = "ragchatbot"

pinecone_instance = Pinecone(api_key=pinecone_api_key)

if index_name not in pinecone_instance.list_indexes().names():
    pinecone_instance.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=environment
        )
    )
else:
    print(f"Index '{index_name}' already exists. Using the existing index.")

index = pinecone_instance.Index(index_name)

openai.api_key = "sk-proj-************************************"# OPEN AI API KEY

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except PdfReadError:
            print(f"Error processing {pdf_file}: EOF marker not found")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def create_pinecone_index(chunks):
    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        index.upsert(vectors=[(f"chunk-{i}", embedding, {"text": chunk})])

def query_pinecone_index(query):
    query_embedding = create_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return results

def ask_question_from_context(query, context):
    prompt = f"""
    Use the following context to answer the question. If the answer is not available in the context, respond with:
    "The answer is not available in the context."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=200
    )
    return response.choices[0].text.strip()

from google.colab import files
uploaded_files = files.upload()
pdf_files = list(uploaded_files.keys())

raw_text = get_pdf_text(pdf_files)
text_chunks = get_text_chunks(raw_text)

create_pinecone_index(text_chunks)

query = "What is the key business strategy mentioned in the document?"
results = query_pinecone_index(query)

context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
answer = ask_question_from_context(query, context)

print(f"Q: {query}")
print(f"A: {answer}")
