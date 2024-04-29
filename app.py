from flask import Flask, request, render_template, jsonify
from pydantic import BaseModel

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

processing_status = "not-started"
gQuestion = ""
gResult = []

class GetQuestionAndFactsResponse(BaseModel):
    question: str
    facts: list[str]
    status: str

def get_vectorstore_from_url(urls):
    # Load multiple URLs concurrently
    loaders = [WebBaseLoader(url) for url in urls]
    
    alldocuments = []
    for loader in loaders:
        # Load the document
        document = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        # Append chunks to the list of all documents
        alldocuments.extend(document_chunks)
    
    # Create a vectorstore from the chunks
    vector_store = FAISS.from_documents(alldocuments, OpenAIEmbeddings())
    
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever

def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "The context is about a team call log. Act like a supervisor and notice all the decision made. Answer the user question in points by using '-' to separate point, and also by saying 'The team has' in each point:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
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
    global processing_status
    global gQuestion
    global gResult

    processing_status = "processing"
    question = request.form.get('question')
    documents = request.form.getlist('documents')

    document = documents[0].split()

    gQuestion = question

    try:
        knowledgeBase = get_vectorstore_from_url(document)
    except Exception as e:
        return render_template('index.html', fail="Invalid url, or can not parse the url")

    retriever_chain = get_context_retriever_chain(knowledgeBase)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    chat_history = []

    responses = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    
    response = responses['answer']
    items = response.split('-')

    items = [item.strip() for item in items if item.strip()]

    gResult = items
    processing_status = "done"
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
    global processing_status
    global gQuestion
    global gResult

    processing_status = "processing"
    question = request.form.get('question')
    documents = request.form.getlist('documents')

    gQuestion = question

    try:
        knowledgeBase = get_vectorstore_from_url(documents)
    except Exception as e:
        return jsonify({"status": "error", "message": "Invalid url, or can not parse the url"}), 500

    retriever_chain = get_context_retriever_chain(knowledgeBase)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    chat_history = []

    responses = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    
    response = responses['answer']
    items = response.split('-')

    items = [item.strip() for item in items if item.strip()]

    gResult = items
    processing_status = "done"


    return jsonify({'message': 'Question and documents submitted successfully'}), 200

@app.route('/get_question_and_facts', methods=['GET'])
def get_question_and_facts():
    global processing_status
    global gQuestion
    global gResult

    response = GetQuestionAndFactsResponse(question=gQuestion, facts=gResult, status=processing_status)
    return jsonify(response.dict()), 200

@app.route('/test/<path:path>')
def send_test(path):
    return send_from_directory('test', path)


if __name__ == '__main__':
    app.run(debug=True)