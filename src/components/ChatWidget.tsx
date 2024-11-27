// ChatWidget.tsx
import React, { useState, useEffect, useRef } from 'react';
import './ChatWidget.css';

interface Message {
    type: 'user' | 'bot';
    content: string;
}

interface ChatWidgetProps {
    isOpen: boolean;
    setIsOpen: (isOpen: boolean) => void;
}

const ChatWidget: React.FC<ChatWidgetProps> = ({ isOpen, setIsOpen }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState<string>('');
    const [ws, setWs] = useState<WebSocket | null>(null);
    const [sessionId, setSessionId] = useState<string>('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Generate a random session ID
        setSessionId(Math.random().toString(36).substring(7));
    }, []);

    useEffect(() => {
        // Add this initial greeting message
        setMessages([{ 
            type: 'bot', 
            content: "👋 Welcome to ricco.AI! I'm Ai. How can I help you today?" 
        }]);
    }, []); 

    useEffect(() => {
        if (sessionId && isOpen && !ws) {
            // Try both secure and non-secure connections
            const websocket = new WebSocket(`wss://ricco-ai-lihf.onrender.com/ws/${sessionId}`);
            
            websocket.onopen = () => {
                console.log('Connected to chat server');
            };
    
            websocket.onmessage = (event) => {
                setMessages(prev => [...prev, { type: 'bot', content: event.data }]);
            };
    
            websocket.onclose = () => {
                console.log('Disconnected from chat server');
                setWs(null);
                // Don't show error message on normal closure
                if (isOpen) {
                    setMessages(prev => [...prev, { 
                        type: 'bot', 
                        content: "Connection lost. Please refresh the page to reconnect." 
                    }]);
                }
            };
    
            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                // Only show error message if it's not already shown
                if (!messages.some(m => m.content.includes("error connecting"))) {
                    setMessages(prev => [...prev.filter(m => !m.content.includes("error connecting")), { 
                        type: 'bot', 
                        content: "Sorry, there was an error connecting to the chat server. Please try again later." 
                    }]);
                }
            };
    
            setWs(websocket);
        }
    
        return () => {
            if (ws) {
                ws.close();
            }
        };
    }, [sessionId, isOpen]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const sendMessage = () => {
        if (input.trim() && ws) {
            try {
                ws.send(input);
                setMessages(prev => [...prev, { type: 'user', content: input }]);
                setInput('');
            } catch (error) {
                console.error('Error sending message:', error);
                setMessages(prev => [...prev, { 
                    type: 'bot', 
                    content: "Sorry, there was an error sending your message." 
                }]);
            }
        }
    };

    return (
        <div className="chat-widget">
            {!isOpen ? (
                <button 
                    className="chat-bubble" 
                    onClick={() => setIsOpen(true)}
                    aria-label="Open chat"
                >
                    💬
                </button>
            ) : (
                <div className="chat-window">
                    <div className="chat-header">
                        <h3>Chat with us</h3>
                        <button 
                            onClick={() => setIsOpen(false)}
                            aria-label="Close chat"
                        >×</button>
                    </div>
                    <div className="chat-messages">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`message ${msg.type}`}
                                dangerouslySetInnerHTML={{ __html: msg.content }}>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                    <div className="chat-input">
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                            placeholder="Type your message..."
                            aria-label="Chat input"
                        />
                        <button 
                            onClick={sendMessage}
                            aria-label="Send message"
                        >
                            Send
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChatWidget;