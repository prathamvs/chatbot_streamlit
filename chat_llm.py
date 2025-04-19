import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import time
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from langchain.schema import Document
import fitz # PyMuPDf
import json
import cv2

custom_prompt_template = """
       Use the following pieces of information to answer the user's question.
        
        1. If the question is asked based on what is the title of the PDF/what is the PDF about/name of the PDF/what is the PDF mainly about then please only give the description of what PDF contains and what information does the PDF actually provides don't provide page number
        

        2. for answering every question:
        2.1 Analyze Both Text and Tables, When presented with a question, the chatbot will examine both the textual context and any relevant tables.
        Page 68 
        *****
        69
        Only qualified electrical maintenance personnel should install, operate, 
        service or maintain this equipment. 

            
        ***********
        Page 68 
        So here the page number is 68 Please follow this
        2.4 If there is no answer present in the catlogue then don't give answers just say don't have information about it 
        For example 
        how can we change N31 to N33:- There is no answer mentioned so by replacing KTA5000ZC31 with KTA5000ZC33 is incorrect answer so since the question's answer is not present then don't give wrong answer and the page number is provided before and after asteris likewise

        2.5 The answer might be in different section so consider all the sections analyze the sections and then provide answer along with respective page number such as phase conductor answer is on Page 150 not 180 so handle the page numbers carefully which can be above or below the context/table
        2.6 Please don't give any incomplete answer complete the sentences but dont't provide incomplete sentences please use some logic of yours here
        2.7 If Page number is not found then don't give page number 

        3. for answering questions related to current ratings, please carefully examine the table and provide all relevant information, including any multiple values or ratings that may be listed.
        For example 
        3.1. If tables have information likewise | 1000 | 1732 | 866 | then the value for 2500 is 4000 & 5000 & please give right info. as for 2500A the values are in front of it not above or below please follow the instruction carefully
        for example:-
        | Busbar | 1 Source | 2 Sources |
        |--------|----------|-----------|
        | 800    | 1386     | 693       |
        | 1000   | 1732     | 866       |
        
        
        Note:-  This is just example context will start after this instruction
        
        3.2. The structure of table might not be seen for example it may look like  
            e.g.
            Utilization
            'Medical Party
            P1 installation 0.45 0.23
            P2 installation 0.72 0.82
            ' but which is actually a table and has answers likewise

            '
            |P1 Installation|Medical|0.45|
            |P1 Installation|Party|0.23|
            |P2 Installation|Medical|0.72|
            |P2 Installation|Party|0.82|

            which means P1 installation for medical is 0.45 and for party is 0.23
        
        4. for analyzing correct page numbers:
        4.1 Check the page numbers correctly because some page numbers are mentioned after paragraph
            for example:-
            Page 68 
            *****
            Only qualified electrical maintenance personnel should install, operate, 
            service or maintain this equipment. 

              
            ***********
            Page 68 
            Where in this context the page number is Page 68 which is provided after the context
        
        5. Type of Question
        5.1 For some of the question please use some keyword searching for example For which type of conductor the section is provided in catalogue? because it's answer is 4 + PE 
        5.2 Some questions might be incomplete so please autocomplete those questions by using logic and then give answer
        
        6. So, Atlast while giving answer do mention page number and Name of the PDf now there will be multiple context so you can use seperate by finding "Answer found in this PDF" text for example

        **Answer found in this PDF**:
        Title: **Canalis KT - Busbar Trunking System - Installation Manual - 11/2018**
        Name:**PDF2**
        Supports and Run Components:- Page 22
        22 QGH3492101-01 11/2018General Rules for Installing Supports:- Page 22
        Safety Instructions:- Page 22
    
        **Answer found in this PDF**:
        Title: **CANALIS KTA 800-5000 DEBU021EN 2023-V3**
        Name:**CANALIS KTA 800-5000 DEBU021EN 2023-V3**
        -1/2:- Page 34
        **L1  L2  L3N  PEDD205853 DB430444 DD205854**:- Page 34
        DD202434DD202435-m DD202436-mDD205855:- Page 34
        32:- Page 34
        **3L + N + PER3L + PE**:- Page 34
        **3L + N + PE**:- Page 34

        So, above you can see after every line of **Answer found in this PDF**: the Name and Title of the uploaded multiple PDFs does get change so please update the Name & Title as per the question asked
        
        But while answering only context and Name of the PDF is to be mentioned not Title of the PDF

        Helpful answer:
"""

categorize_prompt="""
You are a helpful assistant which has to categorize the question and give output of whether the question is related to images/diagrams or just text & tables. Just give the output as Diagrams or Text/Tables and along with that give the title of the PDF and page number only in the dictionary format mentioning "Name", "Page" & "Category" in this format only i.e. key and value both should be in double quoutes. Please don't give output as strings or JSON; it should be compulsory in dictionary form. And keep in mind the page number will always be an integer

Categorization part: Your main aim is categorization, so first analyze the question and then decide whether it is related to diagrams or just text. Mention the correct name as in context; the proper name is given. For example, Name: PDF, so here PDF will be the name of the PDF but don't mention the title in asterisks.

Note: Please don't mention the answer in this “```”; just mention it as a dictionary. And Only give first json as output

"""

def conversational_chat(all_relevant_content,query,input_folder,output_folder):
    
    '''
        This function facilitates a conversational chat using relevant content, a query, 
        and specified input and output folders.
    '''
    

    llm = ChatOpenAI(engine="gpt-4o") ## Defining llm model 

    if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

    ### Setting Human & System message
    system_msg_template = SystemMessagePromptTemplate.from_template(template=custom_prompt_template)

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    ### Creating Propmpt template
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    ### Conversational chat
    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
    # print(all_relevant_content)

    # time.sleep(30)
    retries = 3
    for attempt in range(retries):
        try:
            response = conversation.predict(input=f"Context:\n {all_relevant_content} \n\n Query:\n{query} with context & only name and Page number of the PDF if answer found in both PDF then only mention context & Name of both PDF don't mention Title of the PDF else asked, if title of the PDF is asked then only mention title and noting else")
            break

        except openai.error.RateLimitError as e:
            if attempt < retries - 1:
                st.warning(f"Rate limit exceeded: Waiting for 30 seconds before retrying....")
                # time.sleep(30)

            else:
                st.error("Rate limit exceeded multiple times. Unable to save vector store.")
                raise e

    
    return response
    

