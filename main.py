# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Update these imports
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone
from openai import OpenAI
import os
import json
from typing import List, Dict
import datetime

app = FastAPI()

# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ricco.ai", "https://www.ricco.ai", "https://riccoai.onrender.com"],  # Add the new domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatBot:
    def __init__(self):
        # Initialize with minimal settings
        self.embeddings = None
        self.vectorstore = None
        self.memory_client = None
        self.conversations = {}  # Store only last few messages
        
    def initialize_pinecone(self):
        # Only initialize when needed
        if not self.vectorstore:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("ricco-ai-chatbot")
            
            # Initialize embeddings with minimal settings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
                model_kwargs={'device': 'cpu'}
            )

            # Initialize vectorstore with minimal settings
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=""
            )
        
    def load_documents(self, directory: str):
        try:
            documents = []
            # Process files in smaller batches
            batch_size = 100
            for file in os.listdir(directory):
                if file.endswith('.txt'):
                    loader = TextLoader(f"{directory}/{file}")
                    documents.extend(loader.load())
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(f"{directory}/{file}")
                    documents.extend(loader.load())
            
            # More memory-efficient text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced from 1000
                chunk_overlap=50,  # Reduced from 200
                length_function=len,
                is_separator_regex=False
            )
            texts = text_splitter.split_documents(documents)
            
            # Add documents in smaller batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                
        except Exception as e:
            print(f"Error loading documents: {str(e)}")

    async def get_chat_history(self, session_id: str):
        try:
            # Only keep last 10 messages
            return self.conversations.get(session_id, [])[-10:]
        except Exception as e:
            print(f"Error retrieving chat history: {str(e)}")
            return []

chatbot = ChatBot()

chatbot.load_documents("docs")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await websocket.accept()
        
        while True:
            message = await websocket.receive_text()
            response = await chatbot.process_message(message, session_id)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass