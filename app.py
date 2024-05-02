import asyncio
from flask import Flask, request, render_template, jsonify
from pydantic import BaseModel
import threading
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from flask import send_from_directory

import re
app = Flask(__name__)

processing_status = "idle"
gQuestion = None
gResult = None

class GetQuestionAndFactsResponse(BaseModel):
    question: str
    facts: list[str]
    status: str

def process_question_and_documents(question, documents):
    global processing_status
    global gQuestion
    global gResult

    processing_status = "processing"
    gQuestion = question

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        knowledgeBase = loop.run_until_complete(get_vectorstore_from_url(documents))
        retriever = loop.run_until_complete(knowledgeBase.as_retriever())
        conversation_rag_chain = get_conversational_rag_chain(retriever)

        responses = conversation_rag_chain.invoke({"input": question})
        response = responses['answer']
        items = response.split('+')
        items = [item.strip() for item in items if item.strip()]

        gResult = items
        processing_status = "done"
    except Exception as e:
        processing_status = "error"
        print(f"Error processing question and documents: {e}")

async def load_document(url: str):
    loader = WebBaseLoader(url)
    return await loader.load()

async def get_vectorstore_from_url(urls):
    tasks = [load_document(url) for url in urls]
    documents = await asyncio.gather(*tasks)
    
    # Initialize the vector store
    vector_store = None
    for document in documents:
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        if vector_store is None:
            vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
        else:
            vector_store.add_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store

def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "The context is about a team call log. Act like a supervisor and notice all the facts. Answer the user question in concise point. Each point should start with 'The team has' and separated by using '+'. Use only positive sentences. :\n\n{context}"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_question_and_documents_app', methods=['post'])
def submit_question_and_documents_app():
    question = request.form.get('question')
    documents = request.form.getlist('documents')

    # Run the processing in a separate thread
    threading.Thread(target=process_question_and_documents, args=(question, documents)).start()
    return render_template('response.html')

@app.route('/get_question_and_facts_app', methods=['GET'])
def get_question_and_facts_app():
    global processing_status
    global gQuestion
    global gResult
    return render_template('response.html', response=gResult)

#APIS

@app.route('/submit_question_and_documents', methods=['post'])
def submit_question_and_documents():
    
    question = request.form.get('question')
    documents = request.form.getlist('documents')

    # Run the processing in a separate thread
    threading.Thread(target=process_question_and_documents, args=(question, documents)).start()

    return jsonify({'message': 'Question and documents submitted successfully'}), 200

@app.route('/get_question_and_facts', methods=['GET'])
def get_question_and_facts():
    global processing_status
    global gQuestion
    global gResult

    if gResult is None:
        gResult = [] # Or any other appropriate default value

    response = GetQuestionAndFactsResponse(question=gQuestion, facts=gResult, status=processing_status)
    return jsonify(response.dict()), 200

@app.route('/test/<path:path>')
def send_test(path):
    return send_from_directory('test', path)


if __name__ == '__main__':
    app.run(debug=True)