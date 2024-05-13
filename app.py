import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain_astradb import AstraDBVectorStore
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from flask import Flask, render_template, request, jsonify, redirect
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired
from flask_cors import CORS, cross_origin
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


app = Flask(__name__) 
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
class MYForm(FlaskForm):
    question = StringField(label='question',validators=[DataRequired()])
    submit = SubmitField(label=('ask'))
from dotenv import load_dotenv
load_dotenv()
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
database_id=os.getenv("ASTRA_DB_ID")
namespace = os.getenv("ASTRA_DB_NAMESPACE")
openai_api_key = os.getenv("OPENAI_API_KEY")
collection_name = os.getenv("ASTRA_DB_COLLECTION")
dimension = os.getenv("VECTOR_DIMENSION")
openai_api_key=os.getenv("OPENAI_API_KEY")
input_data = os.getenv("SCRAPED_FILE")
model = os.getenv("VECTOR_MODEL")

# Directory path containing PDF files
directory_path = r"C:\Users\HP\Desktop\Think AI\Chatbot Demo\documents"

# List to store extracted text from all PDF files
raw_text = ""
# docs = []
# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Construct the full path of the PDF file
        filepath = os.path.join(directory_path, filename)
        
        # Read the PDF file and extract text
        with open(filepath, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
            # doc = Document(page_content=text)
            # docs.append(doc)
            # Append extracted text to the string
            raw_text += text

vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="test",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
vstore.add_texts(texts[:])

class Memory:
    def __init__(self):
        self.memory = {}

    def store(self, key, value):
        self.memory[key] = value

    def retrieve(self, key):
        return self.memory.get(key, None)

    def clear_memory(self):
        self.memory = {}
memory = Memory()
session_id = 101
@app.route('/chatnew', methods=['GET', 'POST'])
@cross_origin()
def chatnew():
    if request.method == 'POST':
        question= request.json['question']
        check = request.json['session_id']
        if(session_id != check):
            memory = Memory()
       
        template = """Given excerpts from a document and a question, provide a relevant answer without introducing irrelevant 
        information. Prompt for clarification if needed. Maintain question {context}. Handle tricky 
        insurance-related queries carefully. Remember to respond with details from the relevant company if multiple documents 
        are involved. List all variations if a detail appears with different values.
        remeber this while answering questions:
        {chat_history}
        Human: {human_input}
        Chatbot:"""
        docs = vstore.similarity_search(question)
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        chain = load_qa_chain(
            OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
        )
        return jsonify(chain({"input_documents": docs, "human_input": question}, return_only_outputs=True)['output_text'])
    else:
        # Handle GET request
        return render_template('chatnew.html')

if __name__ == '__main__':
    app.run()
    
