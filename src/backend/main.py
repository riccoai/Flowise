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
    allow_origins=["https://ricco.ai", "https://www.ricco.ai"],  # Update with your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatBot:
    def __init__(self):
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("ricco-ai-chatbot")

        description = index.describe_index_stats()
        print(f"Index info: {description}")
    
        # Initialize NVIDIA API client
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )

        # Initialize embeddings with explicit configuration
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8
            }
        )
    
        # Initialize vectorstore
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",
            namespace=""
        )
        
        self.memory_client = None
        self.make_webhook_url = os.getenv("MAKE_WEBHOOK_URL")
        
        # Chat configuration
        self.chat_config = {
            "model": "meta/llama-3.1-405b-instruct",
            "temperature": 0.7,
            "top_p": 0.7,
            "max_tokens": 150,
            "stream": True
        }
        
        # Store conversation history
        self.conversations: Dict[str, List[Dict]] = {}

    async def get_llm_response(self, prompt: str, session_id: str) -> str:
        # Get conversation history
        conversation = self.conversations.get(session_id, [])
        
        # Add current prompt to messages
        messages = [
            *[{"role": msg["role"], "content": msg["content"]} for msg in conversation],
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.chat_config["model"],
                messages=messages,
                temperature=self.chat_config["temperature"],
                top_p=self.chat_config["top_p"],
                max_tokens=self.chat_config["max_tokens"],
                stream=self.chat_config["stream"]
            )
            
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            # Update conversation history
            conversation.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
            self.conversations[session_id] = conversation[-10:]  # Keep last 10 messages
            
            return response
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"

    async def search_documents(self, query: str, session_id: str) -> str:
        docs = self.vectorstore.similarity_search(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are Ai, a friendly AI assistant for ricco.AI, an AI consultancy company. 
    
    Context: {context}
    Question: {query}

    Instructions: 
    - Be engaging and show genuine interest in the visitor's needs
    - After 1-2 exchanges, suggest a consultation if the user shows interest in AI services
    - Highlight ricco.AI's expertise in AI consulting and implementation
    - Suggest a consultation when user shows interest
    - Use phrases like "I'd be happy to arrange a consultation to discuss this in detail" or "Our experts can guide you through this in a consultation"
    - Keep responses brief but persuasive 
    - Maximum 2-3 sentences
    - Be direct and get to the point quickly
    - If they mention any business challenges or AI interests, emphasize how a consultation could help them
    - Be natural and conversational, not pushy

    Example responses:
    - "That's a great question! Let's discuss your specific needs with one of our experts? I can help schedule a consultation."
    - "I see. I think you'd benefit from a quick chat with our AI consultants. They can provide detailed insights about [specific aspect]."

    Current conversation context: {self.conversations.get(session_id, [])}"""
        
        return await self.get_llm_response(prompt, session_id)

    async def handle_scheduling(self, user_info: dict = None) -> str:
       
        # Send scheduling request to Make.com webhook
        async with httpx.AsyncClient() as client:
            try:
                payload = {
                    "action": "schedule_meeting",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
                response = await client.post(
                    self.make_webhook_url,
                    json=payload,
                    timeout=10.0
                )
            
                print(f"Make.com response: {response.text}")  # Debug line
            
                if response.status_code == 200:
                    booking_url = "https://calendly.com/d/cqvb-cvn-6gc/15-minute-meeting"
                    return f"Here's your scheduling link! 🗓️ <a href='{booking_url}' target='_blank' style='color: #0066cc; text-decoration: underline; font-weight: bold;'>Click here to book your consultation</a>"
                else:
                    print(f"Webhook error: Status {response.status_code}, Response: {response.text}")
                    return "I'm having trouble connecting to the scheduling system. Please try again later."
            
            except Exception as e:
                print(f"Make.com webhook error: {str(e)}")
                return "Sorry, there was an error with the scheduling system. Please try again later."

    def load_documents(self, directory: str):
        documents = []
        for file in os.listdir(directory):
            if file.endswith('.txt'):
                loader = TextLoader(f"{directory}/{file}")
                documents.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(f"{directory}/{file}")
                documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Store in Pinecone
        self.vectorstore.add_documents(texts)

    async def save_chat_history(self, session_id: str, message: dict):
        try:
            if not self.memory_client:
                self.memory_client = UpstashRedisChatMessageHistory(
                    url=os.getenv("UPSTASH_REDIS_URL"),
                    token=os.getenv("UPSTASH_REDIS_TOKEN"),
                    session_id=session_id
                )
            # Convert dict to ChatMessage format
            from langchain_core.messages import HumanMessage, AIMessage
            if message["role"] == "user":
                chat_message = HumanMessage(content=message["content"])
            else:
                chat_message = AIMessage(content=message["content"])
            self.memory_client.add_message(chat_message)
        except Exception as e:
            print(f"Error saving to Upstash: {str(e)}")

    async def get_chat_history(self, session_id: str):
        try:
            key = f"chat_history:{session_id}"
            return self.memory_client.messages
        except Exception as e:
            print(f"Error retrieving from Upstash: {str(e)}")
            return []

    async def process_message(self, message: str, session_id: str) -> str:
    # Save incoming message to history
        await self.save_chat_history(session_id, {
            "role": "user",
            "content": message
        })

         # Check if it's a scheduling request
        if any(word in message.lower() for word in ["schedule", "meeting", "consultation", "book", "appointment"]):
            response = await self.handle_scheduling()
        else:
            response = await self.search_documents(message, session_id)

        # Save bot response to history
        await self.save_chat_history(session_id, {
            "role": "assistant",
            "content": response
        })

        return response

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