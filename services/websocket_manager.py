import json
import asyncio
from typing import Dict, Set, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, WebSocket] = {}
        # Store user locations for live tracking
        self.user_locations: Dict[str, Dict] = {}
        # Store navigation sessions
        self.navigation_sessions: Dict[str, Dict] = {}
        # Store building subscriptions
        self.building_subscriptions: Dict[str, Set[str]] = {}  # building_id -> set of user_ids
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected via WebSocket")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to live navigation service",
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)
    
    def disconnect(self, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        
        if user_id in self.user_locations:
            del self.user_locations[user_id]
        
        if user_id in self.navigation_sessions:
            del self.navigation_sessions[user_id]
        
        # Remove from building subscriptions
        for building_id, users in self.building_subscriptions.items():
            users.discard(user_id)
        
        logger.info(f"User {user_id} disconnected")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send a message to a specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast_to_building(self, message: dict, building_id: str, exclude_user: Optional[str] = None):
        """Broadcast a message to all users in a building"""
        if building_id in self.building_subscriptions:
            for user_id in self.building_subscriptions[buildi