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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain


def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#Split Documents into chunks
def get_text_chunk(documents):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1500, chunk_overlap=150)
    document_content = text_splitter.split_text(documents)
    docs = text_splitter.create_documents(document_content)
    return docs
        
def get_vectordb(text_chunk, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    persist_directory = 'docs/chroma/' 
    vectordb = Chroma.from_documents(
        documents = text_chunk,
        embedding = embeddings,
        persist_directory = persist_directory
    )
    return vectordb
    
def get_pdf_summary(docs):
    
    # Define LLM Chain
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model_name="gpt-3.5-turbo")
    summary_chain = load_summarize_chain(llm, chain_type="refine")
    
    summary = summary_chain.run(docs)
    return summary
    
def get_conversation_chain(vectordb, openai_api_key):
    template = """Use the following pieces of context to answer the question at the end,  If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
        {context}
        Question: {question}
        Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectordb.as_retriever(search_kwargs={'k': 7}),
        memory = memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}        
    )
    return conversation_chain
    
def handle_userinput(user_question):
    
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    response_container = st.container()
    
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
                

def main():
    
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    st.set_page_config(page_title="Chat with PDF Files")
    
    st.title('Chat with PDF')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "summarize" not in st.session_state:
        st.session_state.summarize = None
        
    
    uploaded_file = st.file_uploader('Upload a Pdf File', type='pdf')
    upload = st.button("Upload File")
    
    if upload:
        with st.spinner('Processing file...'):
            files_text = process_pdf(uploaded_file)
            text_chunks = get_text_chunk(files_text)
            vectorstore = get_vectordb(text_chunks, openai_api_key)
            # st.session_state.summarize = get_pdf_summary(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore,  openai_api_key)
            
            st.session_state.processComplete = True
    
    if st.session_state.processComplete == True:
        # st.text(st.session_state.summarize)
        user_question = st.chat_input("Ask your file a question")
        if user_question:
            handle_userinput(user_question)
        

if __name__ == '__main__':
    main()
