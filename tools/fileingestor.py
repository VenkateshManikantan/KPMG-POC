import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from tools.llama2 import Loadllm
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

DB_FAISS_PATH = 'vectorstore/db_faiss'
DB_FAISS_PATH2 = 'vectorstore/db_faiss2'



class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        data = text_splitter.split_documents(documents)
        #data = loader.load()

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create a FAISS vector store and save embeddings
        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        # Load the language model
        llm = Loadllm.load_llm()

        # Create a conversational chain
        

        # Function for conversational chat
        def conversational_chat(query):
            data1 = db.similarity_search(query)
            db2 = FAISS.from_texts(data1[0].page_content, embeddings)
            db3 = FAISS.from_texts(data1[1].page_content, embeddings)
            db4 = FAISS.from_texts(data1[2].page_content, embeddings)
            
            
            db2.merge_from(db3)
            db2.merge_from(db4)
            
            
            st.caption(data1[0].page_content)
            st.caption(data1[1].page_content)
            st.caption(data1[2].page_content)
            
            
            db2.save_local(DB_FAISS_PATH2)
            answer = chain(query,st.session_state['history'])
            return answer
        def chain(query,state):
            db2 = FAISS.load_local(DB_FAISS_PATH2, embeddings)
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db2.as_retriever())
            result = chain({"question": query, "chat_history": state})
            st.session_state['history'].append((query, result["answer"]))
            st.session_state['history'] = []
            return result["answer"]


        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + self.uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
       
        # User input form
        

        # Display chat history
        
                    

        prompt = st.chat_input("Say something")
        

        if prompt:
                output = conversational_chat(prompt)
                st.session_state['past'].append(prompt)
                st.session_state['generated'].append(output)
                st.header(prompt)
                st.write(output)