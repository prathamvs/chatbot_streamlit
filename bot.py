import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer
from vector_storage_retrieval import retrieve_relevant_content, save_vector_stores, load_vector_stores
from chat_llm import conversational_chat
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import shutil
import pandas as pd
import openai
from chat_images import chat_response
from pathlib import Path

absolute_path = Path.cwd()

load_dotenv()

os.environ['CURL_CA_BUNDLE'] = ''

st.set_page_config(page_title="ChatPDF", layout="wide")

absolute_path = Path.cwd()

cert_path = os.path.join(absolute_path , 'pki-it-root.crt')
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = os.environ.get('OPENAI_API_KEY')
os.environ["REQUESTS_CA_BUNDLE"]  = cert_path

# openai.Model = "gpt-35-turbo"


os.environ["OPENAI_API_KEY"] = openai.api_key

# Function to clean up the temporary directory
def clean_temp_dir(temp_dir):
    '''
        Function for cleaning the temporary directory
    '''
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("cleaned file")


def process_csv_excel(uploaded_file):
    '''
        This function processes uploaded CSV or Excel files, converting Excel files to CSV format if necessary.
    '''
    
    """Process CSV/Excel files."""
    if uploaded_file.name.endswith('.csv'):
        # File is already a CSV
        return uploaded_file
    else:
        # Convert Excel to CSV
        excel_data = pd.read_excel(uploaded_file)
        csv_path = os.path.join(st.session_state.temp_dir, "converted_file.csv")
        excel_data.to_csv(csv_path, index=False)
        return csv_path


def main():
        # Check if the session is initialized
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.pdf_parsed = False
        st.session_state.messages = []
        st.session_state.all_documents = None
        st.session_state.uploaded_files = []
        st.session_state.file_paths = []
        st.session_state.temp_dir = None

    if "csv_excel_memory" not in st.session_state:
        st.session_state.csv_excel_memory = []

    # Sidebar logic to upload and parse multiple PDFs, CSVs, or Excel
    with st.sidebar:
        image = Image.open(str(absolute_path) + "/images/logo.png")
        st.sidebar.image(image, use_column_width=True)
        st.title("Upload Files")
        
        # Allow single CSV/Excel, multiple PDFs, or Images
        uploaded_files = st.file_uploader("Upload PDF, CSV, Excel", type=["pdf", "csv", "xlsx","jpg","png"], accept_multiple_files=True)

        if uploaded_files:
            # If new files are uploaded, reset the session
            if not st.session_state.initialized or set([file.name for file in uploaded_files]) != set(st.session_state.uploaded_files):
                # Reset session state
                st.session_state.clear()
                st.session_state.initialized = True
                st.session_state.pdf_parsed = False
                st.session_state.messages = []
                st.session_state.uploaded_files = [file.name for file in uploaded_files]
                st.session_state.all_documents = None
                st.session_state.temp_dir = tempfile.mkdtemp()

                st.session_state.file_paths = []
                
                # Handle PDFs, CSV/Excel, and Images differently
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith('.pdf'):
                        # Save PDF file
                        output_folder = f"{st.session_state.temp_dir}/pdf_file"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                        file_path = os.path.join(output_folder, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.file_paths.append(file_path)

                    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
                        # Process CSV or Excel
                        processed_file = process_csv_excel(uploaded_file)
                        if isinstance(processed_file, str): # It's a CSV file path
                            st.session_state.file_paths.append(processed_file)
                        else: # It's a file-like object (CSV)
                            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(processed_file.getbuffer())
                            st.session_state.file_paths.append(file_path)

                    elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
                        # Save Image file
                        output_folder = f"{st.session_state.temp_dir}/image_files"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                        file_path = os.path.join(output_folder, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.file_paths.append(file_path)

                if any(file.endswith('.pdf') for file in st.session_state.file_paths):
                    # Run the parser function and retrieve relevant content for PDFs
                    st.session_state.all_documents = save_vector_stores(st.session_state.file_paths, f"{st.session_state.temp_dir}/vectors", f"{st.session_state.temp_dir}/output_folder")
                    st.session_state.pdf_parsed = True
                    st.success("Parsing and storing vectors completed!")
                elif any(file.endswith(('.jpg', '.jpeg', '.png','.PNG')) for file in st.session_state.file_paths):
                    file_name = st.session_state.uploaded_files[0]
                    input_file = f"{st.session_state.temp_dir}/image_files/{file_name}"
                    if not os.path.exists(input_file):
                        os.makedirs(input_file)


                    st.success("Image files uploaded and processed!")

                else:
                    st.success("CSV/Excel files uploaded and processed!")
            else:
                st.info("Files already parsed.")

    # Title section
    st.markdown(
        "<h3 style='color:white; background-color:#3DCD58; text-align:center; width:100%; display:block; padding: 12px; border-radius: 5px;font-size:21px;margin-bottom: 25px;margin-top: -25px;'>AI Generative Chatbot</h3>",
        unsafe_allow_html=True,
    )

    # Layout: Two columns, one for PDF/CSV/Excel/Image, one for Chat
    col1, col2 = st.columns([2, 2])

    # Column 1: File Viewer or CSV/Excel/Image Processing
    with col1:
        st.markdown("<h3 style='color:#32CD32;'>File Viewer</h3>", unsafe_allow_html=True)
        with st.container(height=630):
            if st.session_state.file_paths:
                for file_path in st.session_state.file_paths:
                    if file_path.endswith(".pdf"):
                        # Display PDF files
                        with open(file_path, "rb") as f:
                            binary_data = f.read()
                        pdf_viewer(input=binary_data, width=2000)
                        st.divider()
                    elif file_path.endswith(('.csv', '.xlsx')):
                        # Display CSV/Excel content
                        df = pd.read_csv(file_path)
                        st.write(df)
                    elif file_path.endswith(('.jpg', '.jpeg', '.png','.PNG')):
                        # Display Image
                        # st.write(file_path)
                        image = Image.open(file_path)
                        st.image(image,caption='')
                        st.divider()
            else:
                st.write("Please upload files to view them here.")

    # Column 2: Chatbot
    with col2:
        st.markdown("<h3 style='color:#32CD32;'>Chatbot</h3>", unsafe_allow_html=True)
        with st.container(height=630):
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            conversation_container = st.container()

            if user_input := st.chat_input("Ask a question based on the documents!!"):
                st.session_state.messages.append({"role": "user", "content": user_input})

                with conversation_container:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    
                    response = ""
                    if any(file.endswith(".pdf") for file in st.session_state.file_paths):
                        # PDF conversation flow
                        vectors = load_vector_stores(f"{st.session_state.temp_dir}")
                        all_relevant_content = retrieve_relevant_content(st.session_state.all_documents, vectors, user_input)
                        output_folder = f"{st.session_state.temp_dir}/data_images"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        input_folder = f"{st.session_state.temp_dir}/pdf_file"
                        response = conversational_chat(all_relevant_content, user_input, input_folder, output_folder)

                    elif any(file.endswith(('.csv', '.xlsx')) for file in st.session_state.file_paths):
                        # # CSV/Excel conversation flow
                        csv_file = st.session_state.file_paths[0] # Assuming a single CSV/Excel file for simplicity

                        # Combine the last 3 interactions into a single prompt
                        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.csv_excel_memory[-3:]])

                        # Add the current user input to the conversation history
                        full_input = conversation_history + "\nUser: " + user_input

                        # Create the CSV agent (without memory support)
                        agent = create_csv_agent(ChatOpenAI(engine="gpt-4", temperature=0), csv_file, verbose=True, allow_dangerous_code=True)

                        # Get response from the agent using the full conversation context
                        response = agent.run(full_input)

                        # Append the new user input and response to the csv_excel_memory list for future context
                        st.session_state.csv_excel_memory.append({"role": "user", "content": user_input})
                        st.session_state.csv_excel_memory.append({"role": "assistant", "content": response})

                        # Keep only the last 3 exchanges in memory
                        if len(st.session_state.csv_excel_memory) > 6: # 3 user + 3 assistant messages
                            st.session_state.csv_excel_memory = st.session_state.csv_excel_memory[-6:] # Retain the last 3 exchanges from each role

                    elif any(file.endswith(('.jpg', '.jpeg', '.png','PNG')) for file in st.session_state.file_paths):
                        # Image conversation flow (You can add your specific code here to process image-based input)
                        file_name = st.session_state.uploaded_files[0]

                        input_file = f"{st.session_state.temp_dir}/image_files/{file_name}"


                        response = chat_response(input_file,user_input)


                    with st.chat_message("assistant"):
                        st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
