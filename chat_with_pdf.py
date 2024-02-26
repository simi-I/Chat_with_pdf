import openai
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



openai_api_key = os.getenv('OPENAI_API_KEY')

def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#r@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load Document if file is uploaded
    if uploaded_file is not None:
        documents = process_pdf(uploaded_file)
        
        #Split Documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = text_splitter.create_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        persist_directory = 'docs/chroma/'
        
        vectordb = Chroma.from_documents(
            documents = docs,
            embedding = embeddings,
            persist_directory = persist_directory
        )
        
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        
        template = """Use the following pieces of context to answer the question at the end,  If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        
        return qa({'query': query_text})

    
# Page title
st.title('Chat with PDF document')

result = []

with st.form('myform', clear_on_submit=True):
    # File Upload
    uploaded_file = st.file_uploader('Upload a Pdf File', type='pdf')
    
    question = st.text_input(
    "Ask something about the Document",
    placeholder="Can you give me a short summary?")
    
    submitted = st.form_submit_button('Submit')
    
    if submitted:
        with st.spinner('Getting Answer...'):
            response = generate_response(uploaded_file, openai.api_key, question)
            result.append(response)

if len(result):
    st.info(result)    

            
