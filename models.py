from beanie import Document, Indexed
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

class User(Document):
    username: Indexed(str, unique=True)
    email: Indexed(str, unique=True)
    hashed_password: Optional[str] = None  # Not used for OTPless/social auth
    is_admin: bool = False
    role: str = "user"  # "user" or "admin"
    profile_picture: Optional[str] = None  # URL or path to profile picture
    phone: Optional[str] = None  # Phone number for OTPless authentication
    auth_method: str = "otpless"  # "otpless", "social_google", "social_facebook", etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "users"

class Building(Document):
    name: str
    description: Optional[str] = None
    address: str
    latitude: float  # Center point latitude (X coordinate)
    longitude: float  # Center point longitude (Y coordinate)
    boundary_points: Optional[List[Dict[str, float]]] = None  # List of {"lat": x, "lng": y} for polygon boundary
    created_by: ObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "buildings"

class Floor(Document):
    building_id: ObjectId
    floor_number: int  # This will be used as Z coordinate
    name: str
    height: float = 3.0  # Floor height in meters
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "floors"

class Room(Document):
    floor_id: ObjectId
    name: str
    room_type: str  # office, bathroom, elevator, etc.
    coordinates: Dict[str, float]  # {"lat": 0, "lng": 0, "floor": 0, "width": 5, "length": 4}
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "rooms"

class Waypoint(Document):
    floor_id: ObjectId
    latitude: float  # X coordinate (latitude)
    longitude: float  # Y coordinate (longitude)
    floor_number: int  # Z coordinate (floor number)
    waypoint_type: str  # entrance, exit, junction, destination
    name: str
    room_name: Optional[str] = None  # Room identifier/number
    notes: Optional[str] = None  # Additional notes
    images: Optional[List[str]] = None  # Image file paths/URLs
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "waypoints"

class NavigationPath(Document):
    start_waypoint_id: ObjectId
    end_waypoint_id: ObjectId
    path_data: List[Dict[str, float]]  # Array of coordinates for the path
    distance: float
    estimated_time: int  # in seconds
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "navigation_paths"

class ARMarker(Document):
    floor_id: ObjectId
    latitude: float  # X coordinate
    longitude: float  # Y coordinate
    floor_number: int  # Z coordinate (floor number)
    marker_type: str  # qr_code, image, beacon
    marker_data: str  # QR code content or image path
    description: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "ar_markers"

    id: str
    label: Optional[str] = None
    cloud_anchor_id: Optional[str] = None
    x: float  # Relative AR coordinate X (meters)
    y: float  # Relative AR coordinate Y (meters)
    z: float  # Relative AR coordinate Z (meters)
    is_anchor: bool = False
    anchor_id: Optional[str] = None  # If is_anchor is True, this is the QR code data
    neighbors: List[str] = [] # List of Neighbor Node IDs

class GraphNode(BaseModel):
    id: str
    label: Optional[str] = None
    x: float  # X coordinate (latitude)
    y: float  # Y coordinate (longitude)
    z: float  # Z coordinate (floor number)
    node_type: str  # waypoint, room, entrance, exit, etc.
    neighbors: List[str] = []  # List of connected node IDs
    category: Optional[str] = None  # Category for intent-based navigation
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class BuildingGraph(Document):
    building_id: ObjectId
    nodes: List[GraphNode]
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "building_graphs"


# Indoor Navigation Graph with Step-based Edges
class IndoorEdge(BaseModel):
    """Edge connecting two indoor nodes with step-based distance"""
    to_node_id: str
    steps: int  # Number of walking steps
    direction: str  # N, S, E, W, NE, NW, SE, SW, UP, DOWN
    is_accessible: bool = True  # Wheelchair accessible
    crowd_level: int = 0  # 0-5 (0=empty, 5=very crowded)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class IndoorGraphNode(BaseModel):
    """Indoor navigation node with step-based edges"""
    id: str
    label: str
    latitude: float
    longitude: float
    floor_number: int
    image_url: Optional[str] = None
    node_type: str = "waypoint"  # waypoint, room, entrance, exit, elevator, stairs, bathroom
    edges: List[IndoorEdge] = []
    qr_code: Optional[str] = None  # QR code data for this node
    is_emergency_exit: bool = False  # Is this an emergency exit
    is_accessible: bool = True  # Wheelchair accessible node
    landmark_description: Optional[str] = None  # Visual landmark description
    category: Optional[str] = None  # Category for intent-based navigation (food, shopping, services, etc.)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class IndoorGraph(Document):
    """Indoor navigation graph for a building - uses step-based navigation"""
    building_id: ObjectId
    nodes: List[IndoorGraphNode] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Precomputed shortest paths (Floyd-Warshall result)
    shortest_paths: Optional[dict] = None
    accessible_paths: Optional[dict] = None  # Wheelchair accessible paths
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "indoor_graphs"


# ============================================
# A/B TESTING MODELS
# ============================================

class RouteVariant(BaseModel):
    """A variant in an A/B test experiment"""
    id: str
    name: str
    description: Optional[str] = None
    algorithm: str = "floyd_warshall"  # floyd_warshall, dijkstra, a_star
    prefer_accessible: bool = False
    avoid_crowds: bool = False
    prefer_landmarks: bool = False  # Prefer routes with more landmarks
    minimize_floor_changes: bool = False
    weight_multipliers: Dict[str, float] = {}  # Custom edge weight multipliers
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ABTestExperiment(Document):
    """A/B test experiment for route optimization"""
    building_id: ObjectId
    name: str
    description: Optional[str] = None
    variants: List[RouteVariant] = []
    allocation: Dict[str, float] = {}  # variant_id -> percentage (0.0-1.0)
    status: str = "draft"  # draft, active, paused, completed
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    winner_variant_id: Optional[str] = None
    created_by: ObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "ab_test_experiments"

class UserTestAssignment(Document):
    """User assignment to A/B test variant"""
    user_id: ObjectId
    experiment_id: ObjectId
    variant_id: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "user_test_assignments"

class ABTestResult(Document):
    """Result of a single route navigation in an A/B test"""
    experiment_id: ObjectId
    variant_id: str
    user_id: ObjectId
    building_id: ObjectId
    from_node_id: str
    to_node_id: str
    route_completed: bool = False
    completion_time_seconds: Optional[int] = None
    expected_time_seconds: Optional[int] = None
    total_steps: Optional[int] = None
    user_satisfaction: Optional[int] = None  # 1-5 rating
    recalculations: int = 0  # Number of route recalculations
    wrong_turns: int = 0  # Number of wrong turns detected
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "ab_test_results"


# ============================================
# PERSISTENT STORAGE MODELS (replacing in-memory)
# ============================================

class UserFavorite(Document):
    """User's favorite destination"""
    user_id: ObjectId
    building_id: ObjectId
    node_id: str
    label: str
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "user_favorites"

class UserRecentRoute(Document):
    """User's recent navigation route"""
    user_id: ObjectId
    building_id: ObjectId
    from_node_id: str
    to_node_id: str
    from_label: str
    to_label: str
    total_steps: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "user_recent_routes"

class SharedLocation(Document):
    """Shared location link"""
    share_id: str  # Short unique ID for URL
    building_id: ObjectId
    node_id: str
    message: Optional[str] = None
    created_by: ObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "shared_locations"

class GraphVersion(Document):
    """Version history for indoor graphs"""
    building_id: ObjectId
    version: int
    nodes_snapshot: List[dict]  # Snapshot of nodes at this version
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[ObjectId] = None
    change_description: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "graph_versions"

class MagneticCalibration(Document):
    """Magnetic field calibration data for a building"""
    building_id: ObjectId
    floor_number: int
    declination: float  # Magnetic declination in degrees
    inclination: float  # Magnetic inclination in degrees
    field_strength: Optional[float] = None  # Field strength in microtesla
    calibration_points: List[Dict[str, Any]] = []  # Calibration measurements
    calibrated_at: datetime = Field(default_factory=datetime.utcnow)
    calibrated_by: Optional[ObjectId] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "magnetic_calibrations"

class CrowdReport(Document):
    """Crowdsourced crowd density report"""
    building_id: ObjectId
    from_node_id: str
    to_node_id: str
    crowd_level: int  # 0-5
    user_id: ObjectId
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "crowd_reports"
        # Add TTL index for auto-expiration (15 minutes)
        indexes = [
            {"keys": [("timestamp", 1)], "expireAfterSeconds": 900}
        ]

class BlockedPathReport(Document):
    """Report of a blocked path"""
    building_id: ObjectId
    from_node_id: str
    to_node_id: str
    reason: str = "blocked"  # blocked, closed, construction, etc.
    user_id: ObjectId
    confirmed: bool = False
    reports_count: int = 1
    reporters: List[ObjectId] = []
    reported_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "blocked_path_reports"
        # Auto-expire after 1 hour
        indexes = [
            {"keys": [("reported_at", 1)], "expireAfterSeconds": 3600}
        ]

class LandmarkChangeReport(Document):
    """Report of landmark change at a node"""
    building_id: ObjectId
    node_id: str
    user_id: ObjectId
    image_hash: Optional[str] = None
    landmark_visible: bool = True
    flagged: bool = False
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "landmark_change_reports"

class RouteAnomaly(Document):
    """Detected route anomaly"""
    building_id: ObjectId
    from_node_id: str
    to_node_id: str
    travel_times: List[int] = []  # Recent travel times in seconds
    avg_travel_time: float = 0.0
    anomaly_detected: bool = False
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "route_anomalies"
