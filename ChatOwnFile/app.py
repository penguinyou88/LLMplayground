import streamlit as st
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.document_loaders import ImageCaptionLoader
from langchain.docstore.document import Document
from langchain.memory import VectorStoreRetrieverMemory
from langchain import schema

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
# import pytube
# import openai

# load environment variable
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('hf_token')

# Chat UI title
st.set_page_config(layout="wide", page_title="DataChatBot")
st.header("Chat with your Data")
st.markdown ("""
    - :information_source: This tool can be used to chat with different data files you have, currently supporting: CSV/PDF/DOCX/TXT/JPG/PNG
    - For CSV files, once you upload the file, you can select what column(s) to use as the content columns, and the rest will be used as meta data
    """)

# File uploader in the sidebar on the left
with st.sidebar:
    # select model types
    model_list = ['llama2']
    model_selected = st.selectbox("LLM", model_list,index=0)

# Initialize ChatOpenAI model
llm = Ollama(model=model_selected)

# Sidebar section for uploading files
with st.sidebar:
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=None)
    st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

# Check if files are uploaded or YouTube URL is provided
if uploaded_files: 
    # Print the number of files uploaded or YouTube URL provided to the console
    st.sidebar.write(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded files
        documents = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Check if the file is an image
                if file_path.endswith((".png", ".jpg")):
                    # Use ImageCaptionLoader to load the image file
                    image_loader = ImageCaptionLoader(path_images=[file_path])

                    # Load image captions
                    image_documents = image_loader.load()

                    # Append the Langchain documents to the documents list
                    documents.extend(image_documents)
                    
                elif file_path.endswith((".pdf", ".docx", ".txt")):
                    # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                    loader = UnstructuredFileLoader(file_path)
                    loaded_documents = loader.load()

                    # Extend the main documents list with the loaded documents
                    documents.extend(loaded_documents)

                elif file_path.endswith('csv'):
                    # read the data file
                    df = pd.read_csv(file_path)

                    # select content and meta data columns
                    col_names = df.columns.tolist()
                    obj_cols = [y for x,y in zip(df.dtypes,col_names) if x == object or x== str]
                    content_cols = st.sidebar.multiselect('Select Content Columns',obj_cols,help="Unselected columns will be used as meta data for the documents",default=obj_cols[0])
                    meta_cols = [x for x in col_names if x not in content_cols]

                    # concatenate all content columns
                    df[content_cols] = df[content_cols].astype('string')
                    df[content_cols]=df[content_cols].fillna('')
                    content = df[content_cols].agg('.'.join, axis=1)

                    # put all meta data in a dict
                    meta_dict = df[meta_cols].to_dict(orient='index')

                    for i in range(df.shape[0]):
                        new_doc = Document(
                            page_content=content[i],
                            metadata=meta_dict[i],
                        )
                        documents.append(new_doc)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever(), memory_key="chat_history", return_docs=False, return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(),memory=memory, return_source_documents=True, get_chat_history=lambda h : h)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    system_message = "You are a precise, Q&A system that only answers the question based on the given data file. You are not allowed to make up or create content. If you do not know the answer, please just respond I do not know."
    with st.chat_message('System',avatar="🤖"):
        st.markdown(system_message)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("User Questions"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Query the assistant using the latest chat history
        history = [
            f"{message['role']}: {message['content']}" 
            for message in st.session_state.messages
        ]
    
        # add system message
        prompt.messages.insert(0,schema.SystemMessage(content=system_message))
        result = qa({
            "question": prompt, 
            "chat_history": history
        })
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your files first!")