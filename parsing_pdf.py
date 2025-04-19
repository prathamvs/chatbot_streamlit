
from PyPDF2 import PdfReader
from langchain.schema import Document
import os

# Function to format text with page numbers and proper headings
def format_text_with_page_numbers(lines, page_number):
    """
        This function formats a list of text lines by adding page numbers and proper headings.
    """
    
    formatted_text = ""
    section_heading = ""
    

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line.isupper() or stripped_line.endswith(":"):
            section_heading = stripped_line
            formatted_text += f"**{section_heading}**:- Page {page_number}\n"
        else:
            formatted_text += f"{stripped_line}:- Page {page_number}\n"
    
    return formatted_text


# Function to extract and format text from PDFs
def create_formatted_text_from_pdfs(pdf_path, output_folder):
    '''
        This function extracts text from a PDF file, formats it with page numbers and headings, 
        and saves the formatted text to a specified output folder.
    '''
    
    base_name = os.path.basename(pdf_path).replace('.pdf', '')
    output_file_path = os.path.join(output_folder, f"{base_name}.txt")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    documents = []
    reader = PdfReader(pdf_path)
            # Attempt to extract the title from PDF metadata
    title = reader.metadata.get('/Title', None)
    name = base_name

    # If no title, use the filename as the title
    if not title:
        title = base_name


    
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        

            # Store the extracted content
        extracted_text = f"Title of the PDF: **{title}**\n Name of te PDF:**{name}**"
        text_file.write(extracted_text)

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                lines = text.splitlines()
                formatted_text = format_text_with_page_numbers(lines, page_number + 1)

                chunked_documents = formatted_text.split('\n\n')

                for chunk in chunked_documents:
                    # Collect documents (chunks) for embedding
                    document_content = f"Title of the PDF: **{title}**\nName of the PDF: **{name}**\n\n{chunk}"
                    documents.append(Document(page_content=document_content,metadata={"page_number":page_number})) # Split into chunks by double newlines

                # Write formatted content to the text file
                text_file.write(f"******************\nPage {page_number + 1}\n\n")
                text_file.write(formatted_text + "\n")
                text_file.write(f"******************\nPage {page_number + 1}\n\n")

    return documents

