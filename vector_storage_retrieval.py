
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
from parsing_pdf import create_formatted_text_from_pdfs
import streamlit as st
import time

# Save vector stores for each PDF separately
def save_vector_stores(pdf_files, vectors_directory,output_folder, embedding_model="text-embedding-ada-002"):
    
    '''
        This function processes a list of PDF files to create and save vector stores using embeddings.
    '''
    
    embeddings = OpenAIEmbeddings(engine=embedding_model)
    
    if not os.path.exists(vectors_directory):
        os.makedirs(vectors_directory)

    all_documents = []

    for pdf_path in pdf_files:
        # Create documents from each PDF file
        documents = create_formatted_text_from_pdfs(pdf_path,output_folder)
        all_documents.append(documents)

        retries = 3
        for attempt in range(retries):
            try:
                # Create FAISS vector store from documents
                vectorstore = FAISS.from_documents(documents, embeddings)
                # Save vector store with a unique name
                base_name = os.path.basename(pdf_path).replace('.pdf', '')
                vectorstore.save_local(os.path.join(vectors_directory, f"faiss_index_{base_name}"))

                break

            except openai.error.RateLimitError as e:
                if attempt < retries -1:
                    st.warning(f"Rate limit exceeded: {e}. Waiting for 1 minute before retrying....")
                    time.sleep(60)

                else:
                    st.error("Rate limit exceeded multiple times. Unable to save vector store.")
                    raise e

    return all_documents


# Function to load FAISS vector stores from a given directory
def load_vector_stores(vectors_directory, embedding_model="text-embedding-ada-002"):
    
    '''
    This function loads vector stores from a specified directory using embeddings.
    '''
    
    vectorstores = []
    embeddings = OpenAIEmbeddings(engine=embedding_model)

    vectors_directory = f"{vectors_directory}/vectors"

    for file_name in os.listdir(vectors_directory):
        if file_name.startswith("faiss_index_"):
            index_path = os.path.join(vectors_directory, file_name)
            vectorstore = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
            vectorstores.append(vectorstore)

    return vectorstores


# Function to retrieve relevant content using ensemble retrievers
def retrieve_relevant_content(all_documents,vectorstores,query):
    
    '''
        This function retrieves relevant content from a list of documents and vector stores based on a query.
    '''
    
    retrievers = []

    # Combine retrievers for each namespace
    for i,vectorstore in enumerate(vectorstores):
        
        # try:
        
        retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        keyword_retriever = BM25Retriever.from_documents(all_documents[i])
        keyword_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever], weights=[0.4, 0.6])
        retrievers.append(ensemble_retriever)

    all_relevant_content = ""
    time.sleep(30)
    for retriever in retrievers:
        
        docs_rel = retriever.get_relevant_documents(query)

        # Check and print relevant documents if found
        if docs_rel:
            all_relevant_content+="**Answer found in this PDF**: \n"
            all_relevant_content += "\n".join([doc.page_content for doc in docs_rel])
            for doc in docs_rel:
                print(doc.page_content) # This will include page numbers
        else:
            print("No relevant information")

    return all_relevant_content

