# chatbot_streamlit
Chatbot using streamlit
### Description

This chatbot answers user queries based on the files they upload. The supported file types are:

1. PDFs

2. CSV/Excel files

3. Images

The chatbot processes the uploaded files, extracts the necessary information, and uses advanced embeddings and AI models to provide accurate answers.


### Features

Streamlit Interface: Users can upload files through an intuitive web interface.

File Visualization: Uploaded files are displayed on the frontend for reference.

Document Querying: Users can ask questions about the uploaded files, and the chatbot provides meaningful answers based on the context.



---

### Prerequisites

1. Ensure you have Python 3.9 installed.
3. Install the required libraries by running the following command:
#### pip install -r requirements.txt


### Usage

### To run the chatbot:

1. Navigate to the project directory.

2. Use the following command to start the Streamlit app:
#### streamlit run bot.py

Once started, you can upload files and begin querying them.

### Files and Their Purpose

### 1. bot.py:

The main file to run the Streamlit app.
Handles the Streamlit interface and CSV/Excel query functionality using LangChain agents.

### 2. parsing_pdf.py:

Extracts text from PDFs and saves the data in a temporary directory.

Converts the extracted text into document format for further processing.

### 3. vector_storage_retrieval.py:

Converts text documents into vectors using the OpenAI text-embedding-ada-002 model.

Stores the vectors in a FAISS vector database.

Retrieves answers by performing keyword-based and semantic searches based on user queries.

### 4. chat_images.py:

Processes images and answers queries related to the uploaded images.

### 5. chat_llm.py:

Rephrases answers and enhances them based on the query and related context.

Utilizes the GPT-4 model for this purpose.



### 6. requirements.txt:

Contains the list of all necessary dependencies for the project.


### Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.


2. Make your changes.


3. Submit a pull request for review.
