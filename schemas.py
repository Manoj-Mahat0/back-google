from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic_core import core_schema
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: str(x)
        ))

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

class UserBase(BaseModel):
    username: str
    email: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

class User(UserBase):
    id: PyObjectId
    is_admin: bool
    role: str  # "user" or "admin"
    profile_picture: Optional[str] = None
    phone: Optional[str] = None  # Phone number for OTPless
    auth_method: str = "otpless"  # "otpless", "social_google", "social_facebook"
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class BuildingBase(BaseModel):
    name: str
    description: Optional[str] = None
    address: str
    latitude: float  # Center point X coordinate
    longitude: float  # Center point Y coordinate
    boundary_points: Optional[List[Dict[str, float]]] = None  # Polygon boundary points
    version: int = 1

class BuildingCreate(BuildingBase):
    pass

class Building(BuildingBase):
    id: PyObjectId
    created_by: PyObjectId
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class FloorBase(BaseModel):
    floor_number: int  # Z coordinate
    name: str
    height: float = 3.0

class FloorCreate(FloorBase):
    building_id: PyObjectId

class Floor(FloorBase):
    id: PyObjectId
    building_id: PyObjectId
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class RoomBase(BaseModel):
    name: str
    room_type: str
    coordinates: Dict[str, float]  # {"lat": 0, "lng": 0, "floor": 0, "width": 5, "length": 4}

class RoomCreate(RoomBase):
    floor_id: PyObjectId

class Room(RoomBase):
    id: PyObjectId
    floor_id: PyObjectId
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class WaypointBase(BaseModel):
    latitude: float  # X coordinate
    longitude: float  # Y coordinate
    floor_number: int  # Z coordinate
    waypoint_type: str
    name: str
    room_name: Optional[str] = None  # Room identifier/number
    notes: Optional[str] = None  # Additional notes
    images: Optional[List[str]] = None  # Image file paths/URLs

class WaypointCreate(WaypointBase):
    floor_id: PyObjectId

class Waypoint(WaypointBase):
    id: PyObjectId
    floor_id: PyObjectId
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class NavigationRequest(BaseModel):
    building_id: PyObjectId
    start_latitude: float  # X coordinate
    start_longitude: float  # Y coordinate
    start_floor: int  # Z coordinate
    destination_room_name: str

class NavigationResponse(BaseModel):
    path: List[Dict[str, float]]
    distance: float
    estimated_time: int
    instructions: List[str]

class CoordinateGPS(BaseModel):
    latitude: float  # X coordinate
    longitude: float  # Y coordinate
    floor_number: int  # Z coordinate (floor number)
    room_name: Optional[str] = None  # Room identifier/number
    notes: Optional[str] = None  # Additional notes

class BuildingStructure(BaseModel):
    coordinates: List[CoordinateGPS]
    rooms: List[RoomCreate]
    waypoints: List[WaypointCreate]

class BuildingStructureResponse(BaseModel):
    building_id: str
    rooms: List[Room]
    waypoints: List[Waypoint]

class GraphNodeResponse(BaseModel):
    id: str
    label: Optional[str] = None
    cloud_anchor_id: Optional[str] = None
    x: float
    y: float
    z: float
    is_anchor: bool
    anchor_id: Optional[str] = None
    neighbors: List[str]

class NavigationGraphRequest(BaseModel):
    nodes: List[GraphNodeResponse]
