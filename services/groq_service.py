"""
Groq AI Service for generating labels and landmark descriptions
"""
from groq import Groq
import os
from typing import Optional

class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = "llama-3.3-70b-versatile"  # Fast and accurate model
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Groq client"""
        if self._client is None:
            try:
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                print("AI features will use fallback templates")
                self._client = None
        return self._client
    
    def generate_landmark_description(
        self, 
        node_type: str, 
        label: str, 
        context: Optional[str] = None
    ) -> str:
        """
        Generate a concise landmark description for indoor navigation
        
        Args:
            node_type: Type of node (room, entrance, elevator, etc.)
            label: The label/name of the location
            context: Optional additional context
        
        Returns:
            A 20-30 word description for navigation
        """
        # If client is not available, use fallback immediately
        if self.client is None:
            return self._generate_fallback_description(node_type, label)
        
        prompt = f"""Generate a concise landmark description for indoor navigation in exactly 20-30 words.

Location Type: {node_type}
Location Name: {label}
{f'Additional Context: {context}' if context else ''}

The description should help someone navigate to this location. Include visual landmarks, nearby features, or distinctive characteristics. Be specific and practical.

Description:"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise, practical landmark descriptions for indoor navigation. Keep descriptions between 20-30 words, focusing on visual cues and distinctive features."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=100,
                top_p=1,
                stream=False,
            )
            
            description = completion.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            print(f"Error generating description: {e}")
            # Fallback to template-based description
            return self._generate_fallback_description(node_type, label)
    
    def _generate_fallback_description(self, node_type: str, label: str) -> str:
        """Fallback template-based descriptions if API fails"""
        templates = {
            "room": f"Located at {label}. Look for the room number on the door.",
            "entrance": f"Main entrance at {label}. Large doorway with signage.",
            "exit": f"Exit point at {label}. Emergency exit signs visible.",
            "elevator": f"Elevator at {label}. Look for elevator doors and call buttons.",
            "stairs": f"Staircase at {label}. Look for stairwell door with level indicators.",
            "bathroom": f"Restroom at {label}. Look for restroom signage.",
            "cafe": f"Café at {label}. Look for seating area and food service.",
            "office": f"Office at {label}. Look for office door with nameplate.",
            "waypoint": f"Navigation point at {label}. Corridor intersection or landmark.",
        }
        return templates.get(node_type, f"Located at {label}. Follow directional signs.")
    
    def suggest_label(
        self, 
        node_type: str, 
        number: Optional[str] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Generate a smart label based on node type and optional details
        
        Args:
            node_type: Type of node
            number: Optional room/floor number
            name: Optional custom name
        
        Returns:
            Suggested label
        """
        if name:
            return name
        
        templates = {
            "room": f"Room {number}" if number else "Room",
            "entrance": f"Entrance {number}" if number else "Main Entrance",
            "exit": f"Exit {number}" if number else "Emergency Exit",
            "elevator": f"Elevator {number}" if number else "Elevator",
            "stairs": f"Stairs {number}" if number else "Staircase",
            "bathroom": f"Restroom {number}" if number else "Restroom",
            "cafe": f"Café {name or number}" if (name or number) else "Café",
            "office": f"Office {number}" if number else "Office",
            "waypoint": f"Point {number}" if number else "Waypoint",
        }
        
        return templates.get(node_type, f"{node_type.title()} {number if number else ''}")


# Singleton instance
groq_service = GroqService()
