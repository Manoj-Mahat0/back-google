"""
Indoor Navigation Graph API
- Nodes with step-based edges for indoor navigation
- Supports Floyd-Warshall and Johnson's algorithms
- Persisted to MongoDB

FEATURES:
1. QR Code at each node
2. Landmark photos
3. Turn-by-turn animation data
4. ETA calculation
5. Accessibility routes (wheelchair-friendly)
6. Dead reckoning fallback
7. Magnetic field calibration
8. Step length personalization
9. Crowd density data
10. Emergency exit routes
11. Graph validation
12. Bulk import/export
13. Version history
14. A/B testing routes
15. Favorite destinations
16. Recent routes
17. Share location
18. Haptic feedback patterns
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime, timedelta
import math
import json
import uuid
import hashlib
import statistics

from models import Building, User, IndoorGraph, IndoorGraphNode, IndoorEdge, ARMarker
from auth_utils import get_admin_user, get_current_user

router = APIRouter()

# ============================================
# CROWDSOURCED INTELLIGENCE SYSTEM
# ============================================

# Real-time congestion data: { edge_key: [{ timestamp, crowd_level, user_id }] }
_live_congestion_reports: Dict[str, List[dict]] = {}

# Travel time reports: { edge_key: [{ timestamp, actual_steps, expected_steps, user_id }] }
_travel_time_reports: Dict[str, List[dict]] = {}

# Landmark change reports: { node_id: [{ timestamp, image_hash, user_id, flagged }] }
_landmark_change_reports: Dict[str, List[dict]] = {}

# Route anomalies: { route_key: { avg_time, reports_count, anomaly_detected } }
_route_anomalies: Dict[str, dict] = {}

# Blocked/closed paths reported by users
_blocked_paths: Dict[str, dict] = {}  # { edge_key: { reported_at, reports_count, confirmed } }

# Congestion decay time (reports older than this are ignored)
CONGESTION_DECAY_MINUTES = 15

# Minimum reports needed to confirm a blocked path
MIN_BLOCKED_REPORTS = 3

# Anomaly threshold (if actual time > expected * threshold, flag as anomaly)
ANOMALY_THRESHOLD = 1.5

# Default walking speed (meters per second)
DEFAULT_WALKING_SPEED = 1.2
DEFAULT_STEP_LENGTH = 0.7  # meters

# ============================================
# SCHEMAS
# ============================================

class EdgeDataRequest(BaseModel):
    """Edge connecting two nodes"""
    to_node_id: str
    steps: int  # Number of steps to reach
    direction: str  # N, S, E, W, NE, NW, SE, SW, UP, DOWN
    is_accessible: bool = True  # Wheelchair accessible
    crowd_level: int = 0  # 0-5 (0=empty, 5=very crowded)

class IndoorNodeRequest(BaseModel):
    """Indoor navigation node"""
    id: str
    label: str
    latitude: float
    longitude: float
    floor_number: int
    image_url: Optional[str] = None
    node_type: str = "waypoint"
    edges: List[EdgeDataRequest] = []
    qr_code: Optional[str] = None  # QR code data for this node
    is_emergency_exit: bool = False  # Is this an emergency exit
    is_accessible: bool = True  # Wheelchair accessible node
    landmark_description: Optional[str] = None  # Visual landmark description
    category: Optional[str] = None  # Category for intent-based navigation

class IndoorGraphRequest(BaseModel):
    """Request to save indoor graph"""
    nodes: List[IndoorNodeRequest]

class UserPreferences(BaseModel):
    """User navigation preferences"""
    step_length: float = DEFAULT_STEP_LENGTH  # User's step length in meters
    walking_speed: float = DEFAULT_WALKING_SPEED  # meters per second
    prefer_accessible: bool = False  # Prefer wheelchair accessible routes
    avoid_crowds: bool = False  # Avoid crowded areas
    height_cm: Optional[int] = None  # User height for step length calculation

class FavoriteDestination(BaseModel):
    """User's favorite destination"""
    node_id: str
    label: str
    building_id: str
    added_at: datetime = Field(default_factory=datetime.utcnow)

class RecentRoute(BaseModel):
    """Recently navigated route"""
    from_node_id: str
    to_node_id: str
    building_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_steps: int

class ShareLocationRequest(BaseModel):
    """Request to share location"""
    node_id: str
    building_id: str
    message: Optional[str] = None
    expires_in_minutes: int = 60

class GraphValidationResult(BaseModel):
    """Result of graph validation"""
    is_valid: bool
    disconnected_nodes: List[str]
    dead_ends: List[str]
    missing_reverse_edges: List[Dict[str, str]]
    inaccessible_nodes: List[str]
    warnings: List[str]

class HapticPattern(BaseModel):
    """Haptic feedback pattern for navigation"""
    pattern_type: str  # "turn_left", "turn_right", "arrived", "warning"
    vibration_pattern: List[int]  # Duration in ms [vibrate, pause, vibrate, ...]
    intensity: float = 1.0  # 0.0 to 1.0


# ============================================
# CROWDSOURCED INTELLIGENCE SCHEMAS
# ============================================

class TravelReport(BaseModel):
    """Silent background report from user during navigation"""
    building_id: str
    from_node_id: str
    to_node_id: str
    actual_steps: int
    expected_steps: int
    travel_time_seconds: int
    crowd_level_observed: int = 0  # 0-5 based on how slow they moved
    direction: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class LandmarkReport(BaseModel):
    """Report when user captures/views landmark at a node"""
    building_id: str
    node_id: str
    image_hash: Optional[str] = None  # Hash of captured image
    landmark_visible: bool = True  # Could user see the expected landmark?
    notes: Optional[str] = None

class BlockedPathReport(BaseModel):
    """Report when user encounters a blocked path"""
    building_id: str
    from_node_id: str
    to_node_id: str
    reason: str = "blocked"  # blocked, closed, construction, etc.

class RouteConditionResponse(BaseModel):
    """Real-time route conditions for a path"""
    edge_key: str
    current_crowd_level: float
    is_blocked: bool
    estimated_delay_seconds: int
    last_updated: datetime
    reports_count: int


# ============================================
# HAPTIC PATTERNS
# ============================================

HAPTIC_PATTERNS = {
    "turn_left": HapticPattern(
        pattern_type="turn_left",
        vibration_pattern=[100, 50, 100],  # Short-pause-short
        intensity=0.8
    ),
    "turn_right": HapticPattern(
        pattern_type="turn_right",
        vibration_pattern=[100, 50, 100, 50, 100],  # Short-pause-short-pause-short
        intensity=0.8
    ),
    "go_straight": HapticPattern(
        pattern_type="go_straight",
        vibration_pattern=[200],  # Single long
        intensity=0.5
    ),
    "arrived": HapticPattern(
        pattern_type="arrived",
        vibration_pattern=[100, 100, 100, 100, 300],  # Celebration pattern
        intensity=1.0
    ),
    "warning": HapticPattern(
        pattern_type="warning",
        vibration_pattern=[50, 50, 50, 50, 50, 50],  # Rapid pulses
        intensity=1.0
    ),
    "floor_change": HapticPattern(
        pattern_type="floor_change",
        vibration_pattern=[300, 100, 300],  # Long-pause-long
        intensity=0.9
    ),
    "recalculating": HapticPattern(
        pattern_type="recalculating",
        vibration_pattern=[100, 200, 100, 200, 100],
        intensity=0.7
    )
}


# ============================================
# FLOYD-WARSHALL ALGORITHM
# ============================================

def floyd_warshall(nodes: List[dict], accessible_only: bool = False, avoid_crowds: bool = False) -> Dict[str, Dict[str, dict]]:
    """
    Calculate shortest paths between all pairs of nodes.
    Returns: { from_node_id: { to_node_id: { distance, path, steps, directions } } }
    
    Args:
        nodes: List of node dictionaries
        accessible_only: If True, only use wheelchair accessible edges
        avoid_crowds: If True, penalize crowded edges
    """
    INF = float('inf')
    
    # Create node ID to index mapping
    node_ids = [n['id'] for n in nodes]
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Initialize distance and next matrices
    dist = [[INF] * n for _ in range(n)]
    steps = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    directions = [[None] * n for _ in range(n)]
    
    # Set diagonal to 0
    for i in range(n):
        dist[i][i] = 0
        steps[i][i] = 0
    
    # Fill in direct edges
    for node in nodes:
        i = id_to_idx[node['id']]
        for edge in node.get('edges', []):
            if edge['to_node_id'] in id_to_idx:
                # Skip non-accessible edges if accessible_only
                if accessible_only and not edge.get('is_accessible', True):
                    continue
                
                j = id_to_idx[edge['to_node_id']]
                edge_cost = edge['steps']
                
                # Add penalty for crowded edges
                if avoid_crowds:
                    crowd_level = edge.get('crowd_level', 0)
                    edge_cost += crowd_level * 5  # 5 steps penalty per crowd level
                
                dist[i][j] = edge_cost
                steps[i][j] = edge['steps']
                next_node[i][j] = j
                directions[i][j] = edge['direction']
    
    # Floyd-Warshall main loop
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    steps[i][j] = steps[i][k] + steps[k][j]
                    next_node[i][j] = next_node[i][k]
                    directions[i][j] = directions[i][k]
    
    # Build result dictionary
    result = {}
    for i, from_id in enumerate(node_ids):
        result[from_id] = {}
        for j, to_id in enumerate(node_ids):
            if dist[i][j] < INF:
                # Reconstruct path
                path = []
                path_directions = []
                if next_node[i][j] is not None:
                    curr = i
                    while curr != j:
                        path.append(node_ids[curr])
                        if directions[curr][next_node[curr][j]] is not None:
                            path_directions.append(directions[curr][next_node[curr][j]])
                        curr = next_node[curr][j]
                    path.append(node_ids[j])
                
                result[from_id][to_id] = {
                    'total_steps': int(steps[i][j]) if steps[i][j] < INF else -1,
                    'path': path,
                    'directions': path_directions,
                    'reachable': True
                }
            else:
                result[from_id][to_id] = {
                    'total_steps': -1,
                    'path': [],
                    'directions': [],
                    'reachable': False
                }
    
    return result


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_eta(total_steps: int, step_length: float = DEFAULT_STEP_LENGTH, 
                  walking_speed: float = DEFAULT_WALKING_SPEED) -> dict:
    """Calculate ETA based on steps and user preferences"""
    distance_meters = total_steps * step_length
    time_seconds = distance_meters / walking_speed
    
    return {
        "distance_meters": round(distance_meters, 1),
        "time_seconds": int(time_seconds),
        "time_formatted": _format_time(int(time_seconds))
    }

def _format_time(seconds: int) -> str:
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins} min {secs} sec" if secs > 0 else f"{mins} min"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"

def calculate_step_length_from_height(height_cm: int) -> float:
    """Calculate step length based on user height (empirical formula)"""
    # Average step length is about 41% of height
    return (height_cm / 100) * 0.41

def generate_qr_code_data(building_id: str, node_id: str) -> str:
    """Generate unique QR code data for a node"""
    return f"indoor-nav://{building_id}/{node_id}"

def generate_share_link(building_id: str, node_id: str, share_id: str) -> str:
    """Generate shareable location link"""
    return f"https://nav.app/share/{share_id}"

def get_haptic_for_direction_change(from_dir: str, to_dir: str) -> HapticPattern:
    """Get haptic pattern for direction change"""
    # Direction angles
    dir_angles = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
        'UP': -1, 'DOWN': -2
    }
    
    if to_dir in ['UP', 'DOWN']:
        return HAPTIC_PATTERNS['floor_change']
    
    from_angle = dir_angles.get(from_dir, 0)
    to_angle = dir_angles.get(to_dir, 0)
    
    # Calculate turn angle
    diff = (to_angle - from_angle + 360) % 360
    
    if diff < 45 or diff > 315:
        return HAPTIC_PATTERNS['go_straight']
    elif diff <= 180:
        return HAPTIC_PATTERNS['turn_right']
    else:
        return HAPTIC_PATTERNS['turn_left']

def validate_graph(nodes: List[dict]) -> GraphValidationResult:
    """Validate indoor navigation graph for issues"""
    disconnected = []
    dead_ends = []
    missing_reverse = []
    inaccessible = []
    warnings = []
    
    node_ids = {n['id'] for n in nodes}
    node_map = {n['id']: n for n in nodes}
    
    # Check each node
    for node in nodes:
        edges = node.get('edges', [])
        
        # Check for disconnected nodes (no edges)
        if not edges:
            disconnected.append(node['id'])
        
        # Check for dead ends (only one connection)
        if len(edges) == 1:
            dead_ends.append(node['id'])
        
        # Check for missing reverse edges
        for edge in edges:
            to_id = edge['to_node_id']
            if to_id not in node_ids:
                warnings.append(f"Node {node['id']} has edge to non-existent node {to_id}")
                continue
            
            to_node = node_map[to_id]
            has_reverse = any(e['to_node_id'] == node['id'] for e in to_node.get('edges', []))
            if not has_reverse:
                missing_reverse.append({
                    'from': node['id'],
                    'to': to_id
                })
        
        # Check accessibility
        if not node.get('is_accessible', True):
            inaccessible.append(node['id'])
    
    # Check graph connectivity using BFS
    if nodes:
        visited = set()
        queue = [nodes[0]['id']]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in node_map:
                for edge in node_map[current].get('edges', []):
                    if edge['to_node_id'] not in visited:
                        queue.append(edge['to_node_id'])
        
        unreachable = node_ids - visited
        if unreachable:
            warnings.append(f"Graph has {len(unreachable)} unreachable nodes from starting point")
    
    is_valid = len(disconnected) == 0 and len(missing_reverse) == 0 and len(warnings) == 0
    
    return GraphValidationResult(
        is_valid=is_valid,
        disconnected_nodes=disconnected,
        dead_ends=dead_ends,
        missing_reverse_edges=missing_reverse,
        inaccessible_nodes=inaccessible,
        warnings=warnings
    )


# ============================================
# CROWDSOURCED INTELLIGENCE FUNCTIONS
# ============================================

def _get_edge_key(building_id: str, from_node: str, to_node: str) -> str:
    """Generate unique key for an edge"""
    return f"{building_id}:{from_node}:{to_node}"

def _get_route_key(building_id: str, from_node: str, to_node: str) -> str:
    """Generate unique key for a route"""
    return f"{building_id}:{from_node}->{to_node}"

def _cleanup_old_reports():
    """Remove reports older than decay time"""
    cutoff = datetime.utcnow() - timedelta(minutes=CONGESTION_DECAY_MINUTES)
    
    # Clean congestion reports
    for edge_key in list(_live_congestion_reports.keys()):
        _live_congestion_reports[edge_key] = [
            r for r in _live_congestion_reports[edge_key]
            if datetime.fromisoformat(r['timestamp']) > cutoff
        ]
        if not _live_congestion_reports[edge_key]:
            del _live_congestion_reports[edge_key]

def get_live_crowd_level(building_id: str, from_node: str, to_node: str) -> float:
    """Get current crowd level for an edge based on recent reports"""
    _cleanup_old_reports()
    
    edge_key = _get_edge_key(building_id, from_node, to_node)
    reports = _live_congestion_reports.get(edge_key, [])
    
    if not reports:
        return 0.0
    
    # Weight recent reports more heavily
    now = datetime.utcnow()
    weighted_sum = 0
    weight_total = 0
    
    for report in reports:
        report_time = datetime.fromisoformat(report['timestamp'])
        age_minutes = (now - report_time).total_seconds() / 60
        weight = max(0, 1 - (age_minutes / CONGESTION_DECAY_MINUTES))
        weighted_sum += report['crowd_level'] * weight
        weight_total += weight
    
    return weighted_sum / weight_total if weight_total > 0 else 0.0

def is_path_blocked(building_id: str, from_node: str, to_node: str) -> bool:
    """Check if a path is reported as blocked"""
    edge_key = _get_edge_key(building_id, from_node, to_node)
    blocked_info = _blocked_paths.get(edge_key)
    
    if not blocked_info:
        return False
    
    # Check if block report is recent (within 1 hour)
    reported_at = datetime.fromisoformat(blocked_info['reported_at'])
    if datetime.utcnow() - reported_at > timedelta(hours=1):
        del _blocked_paths[edge_key]
        return False
    
    return blocked_info.get('confirmed', False) or blocked_info.get('reports_count', 0) >= MIN_BLOCKED_REPORTS

def calculate_dynamic_crowd_penalty(building_id: str, from_node: str, to_node: str, base_steps: int) -> int:
    """Calculate additional steps penalty based on live crowd data"""
    crowd_level = get_live_crowd_level(building_id, from_node, to_node)
    
    if is_path_blocked(building_id, from_node, to_node):
        return 99999  # Effectively infinite - avoid this path
    
    # Each crowd level adds 10% delay
    penalty_multiplier = 1 + (crowd_level * 0.1)
    return int(base_steps * penalty_multiplier)

def detect_route_anomaly(building_id: str, from_node: str, to_node: str, actual_time: int, expected_time: int) -> bool:
    """Detect if a route segment has anomalies based on travel time"""
    route_key = _get_route_key(building_id, from_node, to_node)
    
    if route_key not in _route_anomalies:
        _route_anomalies[route_key] = {
            'times': [],
            'anomaly_detected': False
        }
    
    _route_anomalies[route_key]['times'].append(actual_time)
    
    # Keep only last 20 reports
    _route_anomalies[route_key]['times'] = _route_anomalies[route_key]['times'][-20:]
    
    times = _route_anomalies[route_key]['times']
    if len(times) >= 5:
        avg_time = statistics.mean(times)
        if actual_time > expected_time * ANOMALY_THRESHOLD:
            _route_anomalies[route_key]['anomaly_detected'] = True
            return True
    
    return False

def get_smart_route_with_live_data(
    graph: 'IndoorGraph',
    from_node: str,
    to_node: str,
    building_id: str,
    accessible_only: bool = False
) -> Dict[str, Any]:
    """Calculate route considering live crowd data and blocked paths"""
    nodes_data = []
    
    for node in graph.nodes:
        node_dict = node.dict()
        # Adjust edge weights based on live data
        adjusted_edges = []
        for edge in node_dict.get('edges', []):
            edge_copy = edge.copy()
            
            # Check if blocked
            if is_path_blocked(building_id, node.id, edge['to_node_id']):
                edge_copy['steps'] = 99999  # Avoid blocked paths
            else:
                # Add crowd penalty
                edge_copy['steps'] = calculate_dynamic_crowd_penalty(
                    building_id, node.id, edge['to_node_id'], edge['steps']
                )
            
            adjusted_edges.append(edge_copy)
        
        node_dict['edges'] = adjusted_edges
        nodes_data.append(node_dict)
    
    # Run Floyd-Warshall with adjusted weights
    return floyd_warshall(nodes_data, accessible_only=accessible_only)

def suggest_alternate_route(
    graph: 'IndoorGraph',
    original_path: List[str],
    blocked_edge_from: str,
    blocked_edge_to: str,
    building_id: str
) -> Optional[Dict[str, Any]]:
    """Suggest an alternate route avoiding a blocked edge"""
    if len(original_path) < 2:
        return None
    
    from_node = original_path[0]
    to_node = original_path[-1]
    
    # Temporarily mark the edge as blocked
    edge_key = _get_edge_key(building_id, blocked_edge_from, blocked_edge_to)
    _blocked_paths[edge_key] = {
        'reported_at': datetime.utcnow().isoformat(),
        'reports_count': MIN_BLOCKED_REPORTS,
        'confirmed': True
    }
    
    # Calculate new route
    new_paths = get_smart_route_with_live_data(graph, from_node, to_node, building_id)
    
    if from_node in new_paths and to_node in new_paths[from_node]:
        return new_paths[from_node][to_node]
    
    return None


# ============================================
# IN-MEMORY STORES (would be MongoDB in production)
# ============================================

# User favorites: { user_id: [FavoriteDestination] }
_user_favorites: Dict[str, List[dict]] = {}

# User recent routes: { user_id: [RecentRoute] }
_user_recent_routes: Dict[str, List[dict]] = {}

# Shared locations: { share_id: { node_id, building_id, message, expires_at } }
_shared_locations: Dict[str, dict] = {}

# Graph version history: { building_id: [{ version, nodes, timestamp }] }
_graph_versions: Dict[str, List[dict]] = {}

# Magnetic calibration data: { building_id: { floor: { declination, inclination } } }
_magnetic_calibration: Dict[str, dict] = {}


# ============================================
# API ENDPOINTS
# ============================================

@router.post("/buildings/{building_id}/indoor-graph")
async def save_indoor_graph(
    building_id: str,
    graph_data: IndoorGraphRequest,
    current_user: User = Depends(get_admin_user)
):
    """Save indoor navigation graph for a building with version history"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Convert request to model format
    nodes = []
    for node in graph_data.nodes:
        edges = [IndoorEdge(
            to_node_id=e.to_node_id,
            steps=e.steps,
            direction=e.direction,
            is_accessible=getattr(e, 'is_accessible', True),
            crowd_level=getattr(e, 'crowd_level', 0)
        ) for e in node.edges]
        
        # Generate QR code if not provided
        qr_code = node.qr_code or generate_qr_code_data(building_id, node.id)
        
        nodes.append(IndoorGraphNode(
            id=node.id,
            label=node.label,
            latitude=node.latitude,
            longitude=node.longitude,
            floor_number=node.floor_number,
            image_url=node.image_url,
            node_type=node.node_type,
            edges=edges,
            qr_code=qr_code,
            is_emergency_exit=getattr(node, 'is_emergency_exit', False),
            is_accessible=getattr(node, 'is_accessible', True),
            landmark_description=getattr(node, 'landmark_description', None),
            category=getattr(node, 'category', None)
        ))
    
    # Check if graph exists
    existing = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    # Save version history
    if building_id not in _graph_versions:
        _graph_versions[building_id] = []
    
    if existing:
        # Save current version to history
        _graph_versions[building_id].append({
            'version': len(_graph_versions[building_id]) + 1,
            'nodes': [n.dict() for n in existing.nodes],
            'timestamp': existing.updated_at.isoformat() if existing.updated_at else datetime.utcnow().isoformat()
        })
        # Keep only last 10 versions
        _graph_versions[building_id] = _graph_versions[building_id][-10:]
        
        existing.nodes = nodes
        existing.updated_at = datetime.utcnow()
        # Precompute shortest paths (normal and accessible)
        nodes_dict = [n.dict() for n in nodes]
        existing.shortest_paths = floyd_warshall(nodes_dict)
        existing.accessible_paths = floyd_warshall(nodes_dict, accessible_only=True)
        await existing.save()
    else:
        nodes_dict = [n.dict() for n in nodes]
        graph = IndoorGraph(
            building_id=ObjectId(building_id),
            nodes=nodes,
            shortest_paths=floyd_warshall(nodes_dict),
            accessible_paths=floyd_warshall(nodes_dict, accessible_only=True)
        )
        await graph.insert()
    
    # Validate graph
    validation = validate_graph([n.dict() for n in nodes])
    
    return {
        "message": "Indoor graph saved successfully",
        "building_id": building_id,
        "nodes_count": len(nodes),
        "edges_count": sum(len(n.edges) for n in nodes),
        "shortest_paths_computed": True,
        "validation": validation.dict(),
        "version": len(_graph_versions.get(building_id, [])) + 1
    }


@router.get("/buildings/{building_id}/indoor-graph")
async def get_indoor_graph(building_id: str):
    """Get indoor navigation graph for a building"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        return {
            "building_id": building_id,
            "nodes": [],
            "message": "No indoor graph found for this building"
        }
    
    # Convert nodes to dict format with all new fields
    nodes_data = []
    for node in graph.nodes:
        node_dict = {
            "id": node.id,
            "label": node.label,
            "latitude": node.latitude,
            "longitude": node.longitude,
            "floor_number": node.floor_number,
            "image_url": node.image_url,
            "node_type": node.node_type,
            "edges": [{"to_node_id": e.to_node_id, "steps": e.steps, "direction": e.direction,
                      "is_accessible": getattr(e, 'is_accessible', True),
                      "crowd_level": getattr(e, 'crowd_level', 0)} for e in node.edges],
            "qr_code": getattr(node, 'qr_code', None) or generate_qr_code_data(building_id, node.id),
            "is_emergency_exit": getattr(node, 'is_emergency_exit', False),
            "is_accessible": getattr(node, 'is_accessible', True),
            "landmark_description": getattr(node, 'landmark_description', None)
        }
        nodes_data.append(node_dict)
    
    return {
        "building_id": building_id,
        "nodes": nodes_data,
        "nodes_count": len(nodes_data),
        "edges_count": sum(len(n.get('edges', [])) for n in nodes_data)
    }


@router.get("/buildings/{building_id}/indoor-graph/route")
async def get_route(
    building_id: str,
    from_node: str,
    to_node: str,
    accessible: bool = False,
    avoid_crowds: bool = False,
    step_length: float = DEFAULT_STEP_LENGTH,
    walking_speed: float = DEFAULT_WALKING_SPEED
):
    """Get shortest route between two nodes with ETA and haptic patterns"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Get or compute shortest paths based on preferences
    nodes_data = [n.dict() for n in graph.nodes]
    
    if accessible or avoid_crowds:
        # Compute custom paths
        all_pairs = floyd_warshall(nodes_data, accessible_only=accessible, avoid_crowds=avoid_crowds)
    elif accessible and hasattr(graph, 'accessible_paths') and graph.accessible_paths:
        all_pairs = graph.accessible_paths
    elif not graph.shortest_paths:
        graph.shortest_paths = floyd_warshall(nodes_data)
        await graph.save()
        all_pairs = graph.shortest_paths
    else:
        all_pairs = graph.shortest_paths
    
    if from_node not in all_pairs:
        raise HTTPException(status_code=404, detail=f"Node '{from_node}' not found")
    
    if to_node not in all_pairs[from_node]:
        raise HTTPException(status_code=404, detail=f"Node '{to_node}' not found")
    
    route_info = all_pairs[from_node][to_node]
    
    if not route_info['reachable']:
        raise HTTPException(status_code=404, detail="No route found between these nodes")
    
    # Get detailed step-by-step instructions with haptic patterns
    instructions = []
    path = route_info['path']
    directions = route_info['directions']
    
    nodes_dict = {n.id: n for n in graph.nodes}
    
    prev_direction = None
    for i, node_id in enumerate(path):
        node = nodes_dict.get(node_id)
        if not node:
            continue
            
        instruction = {
            "step": i + 1,
            "node_id": node_id,
            "label": node.label,
            "floor": node.floor_number,
            "image_url": node.image_url,
            "node_type": node.node_type,
            "qr_code": getattr(node, 'qr_code', None),
            "landmark": getattr(node, 'landmark_description', None),
            "latitude": node.latitude,
            "longitude": node.longitude,
        }
        
        if i < len(directions):
            # Get edge info for steps
            next_node_id = path[i + 1] if i + 1 < len(path) else None
            edge_steps = 0
            
            for edge in node.edges:
                if edge.to_node_id == next_node_id:
                    edge_steps = edge.steps
                    break
            
            current_direction = directions[i]
            instruction["direction"] = current_direction
            instruction["steps_to_next"] = edge_steps
            instruction["instruction"] = f"Walk {edge_steps} steps {_direction_to_text(current_direction)}"
            
            # Add haptic pattern
            if prev_direction:
                haptic = get_haptic_for_direction_change(prev_direction, current_direction)
            else:
                haptic = HAPTIC_PATTERNS['go_straight']
            
            instruction["haptic"] = haptic.dict()
            prev_direction = current_direction
            
            # Add turn-by-turn animation data
            instruction["animation"] = {
                "type": "arrow",
                "direction": current_direction,
                "angle": _direction_to_angle(current_direction)
            }
        else:
            instruction["instruction"] = "🎉 You have arrived!"
            instruction["haptic"] = HAPTIC_PATTERNS['arrived'].dict()
            instruction["animation"] = {"type": "celebration"}
        
        instructions.append(instruction)
    
    # Calculate ETA
    eta = calculate_eta(route_info['total_steps'], step_length, walking_speed)
    
    return {
        "from": from_node,
        "to": to_node,
        "total_steps": route_info['total_steps'],
        "reachable": route_info['reachable'],
        "path": path,
        "directions": directions,
        "instructions": instructions,
        "eta": eta,
        "accessible_route": accessible,
        "avoided_crowds": avoid_crowds
    }


def _direction_to_angle(direction: str) -> int:
    """Convert direction to angle in degrees"""
    angles = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
        'UP': 0, 'DOWN': 180
    }
    return angles.get(direction, 0)


@router.delete("/buildings/{building_id}/indoor-graph")
async def delete_indoor_graph(
    building_id: str,
    current_user: User = Depends(get_admin_user)
):
    """Delete indoor graph for a building"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    result = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not result:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    await result.delete()
    
    return {"message": "Indoor graph deleted successfully"}


@router.get("/buildings/{building_id}/indoor-graph/stats")
async def get_graph_stats(building_id: str):
    """Get statistics about the indoor graph"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        return {"building_id": building_id, "exists": False}
    
    # Calculate stats
    total_nodes = len(graph.nodes)
    total_edges = sum(len(n.edges) for n in graph.nodes)
    floors = set(n.floor_number for n in graph.nodes)
    node_types = {}
    emergency_exits = []
    accessible_nodes = 0
    
    for n in graph.nodes:
        node_types[n.node_type] = node_types.get(n.node_type, 0) + 1
        if getattr(n, 'is_emergency_exit', False):
            emergency_exits.append(n.id)
        if getattr(n, 'is_accessible', True):
            accessible_nodes += 1
    
    # Check connectivity
    connected_nodes = set()
    for n in graph.nodes:
        if n.edges:
            connected_nodes.add(n.id)
            for e in n.edges:
                connected_nodes.add(e.to_node_id)
    
    unconnected = [n.id for n in graph.nodes if n.id not in connected_nodes]
    
    # Validate graph
    validation = validate_graph([n.dict() for n in graph.nodes])
    
    return {
        "building_id": building_id,
        "exists": True,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "floors": sorted(list(floors)),
        "node_types": node_types,
        "unconnected_nodes": unconnected,
        "is_fully_connected": len(unconnected) == 0,
        "emergency_exits_count": len(emergency_exits),
        "accessible_nodes_count": accessible_nodes,
        "validation": validation.dict(),
        "created_at": graph.created_at.isoformat() if graph.created_at else None,
        "updated_at": graph.updated_at.isoformat() if graph.updated_at else None,
    }


# ============================================
# EMERGENCY EXIT ROUTES
# ============================================

@router.get("/buildings/{building_id}/indoor-graph/emergency-exit")
async def get_nearest_emergency_exit(
    building_id: str,
    from_node: str,
    accessible: bool = False
):
    """Find nearest emergency exit from current location"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Find all emergency exits
    emergency_exits = [n for n in graph.nodes if getattr(n, 'is_emergency_exit', False)]
    
    if not emergency_exits:
        raise HTTPException(status_code=404, detail="No emergency exits defined in this building")
    
    # Get shortest paths
    nodes_data = [n.dict() for n in graph.nodes]
    all_pairs = floyd_warshall(nodes_data, accessible_only=accessible)
    
    if from_node not in all_pairs:
        raise HTTPException(status_code=404, detail=f"Node '{from_node}' not found")
    
    # Find nearest exit
    nearest_exit = None
    min_steps = float('inf')
    
    for exit_node in emergency_exits:
        if exit_node.id in all_pairs[from_node]:
            route = all_pairs[from_node][exit_node.id]
            if route['reachable'] and route['total_steps'] < min_steps:
                min_steps = route['total_steps']
                nearest_exit = exit_node
    
    if not nearest_exit:
        raise HTTPException(status_code=404, detail="No reachable emergency exit found")
    
    route_info = all_pairs[from_node][nearest_exit.id]
    
    return {
        "emergency": True,
        "from": from_node,
        "to": nearest_exit.id,
        "exit_label": nearest_exit.label,
        "exit_floor": nearest_exit.floor_number,
        "total_steps": route_info['total_steps'],
        "path": route_info['path'],
        "directions": route_info['directions'],
        "eta": calculate_eta(route_info['total_steps']),
        "all_exits": [{"id": e.id, "label": e.label, "floor": e.floor_number} for e in emergency_exits]
    }


# ============================================
# QR CODE LOOKUP
# ============================================

@router.get("/buildings/{building_id}/qr-markers/export")
async def export_qr_markers(building_id: str, format: str = "json"):
    """Export all QR markers for a building"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    # Get building info
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Get indoor graph
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Get AR markers (QR codes)
    markers = await ARMarker.find(ARMarker.floor_id.in_([floor.id for floor in building.floors])).to_list()
    qr_markers = [marker for marker in markers if marker.marker_type == "qr_code"]
    
    # Prepare export data
    export_data = {
        "building_id": str(building.id),
        "building_name": building.name,
        "exported_at": datetime.utcnow().isoformat(),
        "total_markers": len(qr_markers),
        "qr_markers": []
    }
    
    # Add QR marker data with node information
    for marker in qr_markers:
        # Find corresponding node in graph
        node_info = None
        for node in graph.nodes:
            if getattr(node, 'qr_code', None) == marker.marker_data:
                node_info = {
                    "node_id": node.id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "floor_number": node.floor_number,
                    "is_accessible": getattr(node, 'is_accessible', True),
                    "is_emergency_exit": getattr(node, 'is_emergency_exit', False),
                    "landmark_description": getattr(node, 'landmark_description', None)
                }
                break
        
        marker_data = {
            "id": str(marker.id),
            "building_id": str(building.id),
            "floor_id": str(marker.floor_id),
            "x": marker.latitude,
            "y": marker.longitude,
            "floor_number": marker.floor_number,
            "orientation_degrees": 0.0,  # Default orientation
            "qr_data": marker.marker_data,
            "description": marker.description,
            "node_info": node_info
        }
        export_data["qr_markers"].append(marker_data)
    
    if format.lower() == "csv":
        # Generate CSV format
        csv_content = "ID,Building ID,Floor ID,X,Y,Floor Number,QR Data,Description,Node ID,Node Label,Node Type\n"
        for marker in export_data["qr_markers"]:
            node = marker.get("node_info", {})
            csv_content += f"{marker['id']},{marker['building_id']},{marker['floor_id']},{marker['x']},{marker['y']},{marker['floor_number']},\"{marker['qr_data']}\",\"{marker['description']}\",{node.get('node_id', '')},{node.get('label', '')},{node.get('node_type', '')}\n"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={building.name}_qr_markers.csv"}
        )
    
    return export_data


@router.get("/buildings/{building_id}/indoor-graph/qr-lookup")
async def lookup_node_by_qr(building_id: str, qr_data: str):
    """Look up node by QR code data"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Find node by QR code
    for node in graph.nodes:
        if getattr(node, 'qr_code', None) == qr_data:
            return {
                "found": True,
                "node": {
                    "id": node.id,
                    "label": node.label,
                    "floor_number": node.floor_number,
                    "node_type": node.node_type,
                    "latitude": node.latitude,
                    "longitude": node.longitude,
                    "landmark": getattr(node, 'landmark_description', None)
                }
            }
    
    # Try parsing QR data format: indoor-nav://building_id/node_id
    if qr_data.startswith("indoor-nav://"):
        parts = qr_data.replace("indoor-nav://", "").split("/")
        if len(parts) >= 2:
            node_id = parts[1]
            for node in graph.nodes:
                if node.id == node_id:
                    return {
                        "found": True,
                        "node": {
                            "id": node.id,
                            "label": node.label,
                            "floor_number": node.floor_number,
                            "node_type": node.node_type,
                            "latitude": node.latitude,
                            "longitude": node.longitude
                        }
                    }
    
    return {"found": False, "message": "Node not found for this QR code"}


# ============================================
# FAVORITES & RECENT ROUTES
# ============================================

@router.post("/favorites")
async def add_favorite(
    favorite: FavoriteDestination,
    current_user: User = Depends(get_current_user)
):
    """Add a favorite destination"""
    user_id = str(current_user.id)
    
    if user_id not in _user_favorites:
        _user_favorites[user_id] = []
    
    # Check if already exists
    for fav in _user_favorites[user_id]:
        if fav['node_id'] == favorite.node_id and fav['building_id'] == favorite.building_id:
            return {"message": "Already in favorites", "favorites": _user_favorites[user_id]}
    
    _user_favorites[user_id].append(favorite.dict())
    
    # Keep only last 20 favorites
    _user_favorites[user_id] = _user_favorites[user_id][-20:]
    
    return {"message": "Added to favorites", "favorites": _user_favorites[user_id]}


@router.get("/favorites")
async def get_favorites(current_user: User = Depends(get_current_user)):
    """Get user's favorite destinations"""
    user_id = str(current_user.id)
    return {"favorites": _user_favorites.get(user_id, [])}


@router.delete("/favorites/{node_id}")
async def remove_favorite(
    node_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove a favorite destination"""
    user_id = str(current_user.id)
    
    if user_id in _user_favorites:
        _user_favorites[user_id] = [f for f in _user_favorites[user_id] if f['node_id'] != node_id]
    
    return {"message": "Removed from favorites", "favorites": _user_favorites.get(user_id, [])}


@router.post("/recent-routes")
async def add_recent_route(
    route: RecentRoute,
    current_user: User = Depends(get_current_user)
):
    """Add a recent route"""
    user_id = str(current_user.id)
    
    if user_id not in _user_recent_routes:
        _user_recent_routes[user_id] = []
    
    _user_recent_routes[user_id].append(route.dict())
    
    # Keep only last 10 routes
    _user_recent_routes[user_id] = _user_recent_routes[user_id][-10:]
    
    return {"message": "Route saved", "recent_routes": _user_recent_routes[user_id]}


@router.get("/recent-routes")
async def get_recent_routes(current_user: User = Depends(get_current_user)):
    """Get user's recent routes"""
    user_id = str(current_user.id)
    return {"recent_routes": _user_recent_routes.get(user_id, [])}


# ============================================
# SHARE LOCATION
# ============================================

@router.post("/share-location")
async def share_location(
    request: ShareLocationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate a shareable location link"""
    share_id = str(uuid.uuid4())[:8]
    
    expires_at = datetime.utcnow().timestamp() + (request.expires_in_minutes * 60)
    
    _shared_locations[share_id] = {
        "node_id": request.node_id,
        "building_id": request.building_id,
        "message": request.message,
        "expires_at": expires_at,
        "created_by": str(current_user.id),
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {
        "share_id": share_id,
        "share_link": generate_share_link(request.building_id, request.node_id, share_id),
        "expires_in_minutes": request.expires_in_minutes,
        "message": request.message
    }


@router.get("/share-location/{share_id}")
async def get_shared_location(share_id: str):
    """Get shared location details"""
    if share_id not in _shared_locations:
        raise HTTPException(status_code=404, detail="Share link not found or expired")
    
    location = _shared_locations[share_id]
    
    # Check expiration
    if datetime.utcnow().timestamp() > location['expires_at']:
        del _shared_locations[share_id]
        raise HTTPException(status_code=410, detail="Share link has expired")
    
    # Get node details
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(location['building_id']))
    
    node_info = None
    if graph:
        for node in graph.nodes:
            if node.id == location['node_id']:
                node_info = {
                    "id": node.id,
                    "label": node.label,
                    "floor_number": node.floor_number,
                    "latitude": node.latitude,
                    "longitude": node.longitude
                }
                break
    
    return {
        "building_id": location['building_id'],
        "node_id": location['node_id'],
        "node": node_info,
        "message": location['message'],
        "created_at": location['created_at']
    }


# ============================================
# VERSION HISTORY
# ============================================

@router.get("/buildings/{building_id}/indoor-graph/versions")
async def get_graph_versions(
    building_id: str,
    current_user: User = Depends(get_admin_user)
):
    """Get version history of indoor graph"""
    versions = _graph_versions.get(building_id, [])
    return {
        "building_id": building_id,
        "versions": versions,
        "total_versions": len(versions)
    }


@router.post("/buildings/{building_id}/indoor-graph/rollback/{version}")
async def rollback_graph(
    building_id: str,
    version: int,
    current_user: User = Depends(get_admin_user)
):
    """Rollback to a previous version of the graph"""
    versions = _graph_versions.get(building_id, [])
    
    target_version = None
    for v in versions:
        if v['version'] == version:
            target_version = v
            break
    
    if not target_version:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    
    # Restore the version
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Convert dict nodes back to model
    nodes = []
    for node_data in target_version['nodes']:
        edges = [IndoorEdge(**e) for e in node_data.get('edges', [])]
        node_data['edges'] = edges
        nodes.append(IndoorGraphNode(**node_data))
    
    graph.nodes = nodes
    graph.updated_at = datetime.utcnow()
    graph.shortest_paths = floyd_warshall([n.dict() for n in nodes])
    await graph.save()
    
    return {
        "message": f"Rolled back to version {version}",
        "nodes_count": len(nodes)
    }


# ============================================
# BULK IMPORT/EXPORT
# ============================================

@router.get("/buildings/{building_id}/indoor-graph/export")
async def export_graph(building_id: str):
    """Export indoor graph as JSON"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    export_data = {
        "building_id": building_id,
        "exported_at": datetime.utcnow().isoformat(),
        "nodes": [n.dict() for n in graph.nodes]
    }
    
    return JSONResponse(
        content=export_data,
        headers={"Content-Disposition": f"attachment; filename=indoor_graph_{building_id}.json"}
    )


@router.post("/buildings/{building_id}/indoor-graph/import")
async def import_graph(
    building_id: str,
    import_data: dict,
    current_user: User = Depends(get_admin_user)
):
    """Import indoor graph from JSON"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    nodes_data = import_data.get('nodes', [])
    if not nodes_data:
        raise HTTPException(status_code=400, detail="No nodes in import data")
    
    # Convert to model format
    nodes = []
    for node_data in nodes_data:
        edges = [IndoorEdge(**e) for e in node_data.get('edges', [])]
        node_data['edges'] = edges
        nodes.append(IndoorGraphNode(**node_data))
    
    # Save
    existing = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if existing:
        existing.nodes = nodes
        existing.updated_at = datetime.utcnow()
        existing.shortest_paths = floyd_warshall([n.dict() for n in nodes])
        await existing.save()
    else:
        graph = IndoorGraph(
            building_id=ObjectId(building_id),
            nodes=nodes,
            shortest_paths=floyd_warshall([n.dict() for n in nodes])
        )
        await graph.insert()
    
    return {
        "message": "Graph imported successfully",
        "nodes_count": len(nodes),
        "edges_count": sum(len(n.edges) for n in nodes)
    }


# ============================================
# GRAPH VALIDATION
# ============================================

@router.get("/buildings/{building_id}/indoor-graph/validate")
async def validate_indoor_graph(building_id: str):
    """Validate indoor graph for issues"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    validation = validate_graph([n.dict() for n in graph.nodes])
    
    return {
        "building_id": building_id,
        "validation": validation.dict()
    }


# ============================================
# MAGNETIC CALIBRATION
# ============================================

@router.post("/buildings/{building_id}/magnetic-calibration")
async def save_magnetic_calibration(
    building_id: str,
    calibration_data: dict,
    current_user: User = Depends(get_admin_user)
):
    """Save magnetic field calibration data for a building"""
    _magnetic_calibration[building_id] = calibration_data
    return {"message": "Calibration data saved", "building_id": building_id}


@router.get("/buildings/{building_id}/magnetic-calibration")
async def get_magnetic_calibration(building_id: str):
    """Get magnetic field calibration data"""
    return {
        "building_id": building_id,
        "calibration": _magnetic_calibration.get(building_id, {})
    }


# ============================================
# STEP LENGTH PERSONALIZATION
# ============================================

@router.post("/user/preferences")
async def save_user_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_user)
):
    """Save user navigation preferences"""
    # Calculate step length from height if provided
    if preferences.height_cm:
        preferences.step_length = calculate_step_length_from_height(preferences.height_cm)
    
    # In production, save to user profile in MongoDB
    return {
        "message": "Preferences saved",
        "preferences": preferences.dict()
    }


# ============================================
# DEAD RECKONING
# ============================================

@router.post("/dead-reckoning/calculate")
async def calculate_dead_reckoning(
    last_known_node: str,
    steps_taken: int,
    direction: str,
    building_id: str
):
    """Calculate estimated position using dead reckoning when GPS fails"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Find last known node
    last_node = None
    for node in graph.nodes:
        if node.id == last_known_node:
            last_node = node
            break
    
    if not last_node:
        raise HTTPException(status_code=404, detail="Last known node not found")
    
    # Find possible current nodes based on steps and direction
    possible_nodes = []
    
    for edge in last_node.edges:
        if edge.direction == direction:
            step_diff = abs(edge.steps - steps_taken)
            if step_diff <= 5:  # Within 5 steps tolerance
                for node in graph.nodes:
                    if node.id == edge.to_node_id:
                        possible_nodes.append({
                            "node_id": node.id,
                            "label": node.label,
                            "confidence": max(0, 100 - step_diff * 10),
                            "expected_steps": edge.steps,
                            "actual_steps": steps_taken
                        })
    
    # If no exact match, estimate position
    if not possible_nodes:
        # Calculate estimated lat/lng based on direction and steps
        step_length = DEFAULT_STEP_LENGTH
        distance = steps_taken * step_length
        
        # Direction to bearing
        bearings = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        bearing = bearings.get(direction, 0)
        
        # Simple estimation (not accounting for Earth's curvature)
        lat_change = distance * math.cos(math.radians(bearing)) / 111000
        lng_change = distance * math.sin(math.radians(bearing)) / (111000 * math.cos(math.radians(last_node.latitude)))
        
        return {
            "estimated": True,
            "last_known_node": last_known_node,
            "estimated_latitude": last_node.latitude + lat_change,
            "estimated_longitude": last_node.longitude + lng_change,
            "confidence": 50,
            "message": "Position estimated using dead reckoning"
        }
    
    # Sort by confidence
    possible_nodes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "estimated": False,
        "last_known_node": last_known_node,
        "possible_nodes": possible_nodes,
        "best_match": possible_nodes[0] if possible_nodes else None
    }


# ============================================
# CROWD DENSITY
# ============================================

@router.post("/buildings/{building_id}/indoor-graph/crowd-density")
async def update_crowd_density(
    building_id: str,
    updates: List[dict],  # [{"from_node": "x", "to_node": "y", "crowd_level": 3}]
    current_user: User = Depends(get_current_user)
):
    """Update crowd density on edges (crowdsourced)"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    updated_count = 0
    for update in updates:
        from_node_id = update.get('from_node')
        to_node_id = update.get('to_node')
        crowd_level = update.get('crowd_level', 0)
        
        for node in graph.nodes:
            if node.id == from_node_id:
                for edge in node.edges:
                    if edge.to_node_id == to_node_id:
                        edge.crowd_level = min(5, max(0, crowd_level))
                        updated_count += 1
    
    if updated_count > 0:
        await graph.save()
    
    return {
        "message": f"Updated {updated_count} edges",
        "building_id": building_id
    }


# ============================================
# HAPTIC PATTERNS
# ============================================

@router.get("/haptic-patterns")
async def get_haptic_patterns():
    """Get all available haptic feedback patterns"""
    return {
        "patterns": {k: v.dict() for k, v in HAPTIC_PATTERNS.items()}
    }


# ============================================
# SHORTEST PATHS
# ============================================

@router.get("/buildings/{building_id}/indoor-graph/shortest-paths")
async def get_all_shortest_paths(
    building_id: str,
    accessible: bool = False
):
    """Get all-pairs shortest paths using Floyd-Warshall"""
    
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Return precomputed paths or compute on demand
    if accessible:
        if hasattr(graph, 'accessible_paths') and graph.accessible_paths:
            return {
                "building_id": building_id,
                "algorithm": "Floyd-Warshall (cached, accessible)",
                "all_pairs_distances": graph.accessible_paths
            }
        nodes_data = [n.dict() for n in graph.nodes]
        all_pairs = floyd_warshall(nodes_data, accessible_only=True)
    elif graph.shortest_paths:
        return {
            "building_id": building_id,
            "algorithm": "Floyd-Warshall (cached)",
            "all_pairs_distances": graph.shortest_paths
        }
    else:
        nodes_data = [n.dict() for n in graph.nodes]
        all_pairs = floyd_warshall(nodes_data)
        graph.shortest_paths = all_pairs
        await graph.save()
    
    return {
        "building_id": building_id,
        "algorithm": "Floyd-Warshall",
        "time_complexity": "O(V³)",
        "all_pairs_distances": all_pairs
    }


def _direction_to_text(direction: str) -> str:
    """Convert direction code to human readable text"""
    mapping = {
        'N': 'North (straight ahead)',
        'S': 'South (turn around)',
        'E': 'East (turn right)',
        'W': 'West (turn left)',
        'NE': 'Northeast (slight right)',
        'NW': 'Northwest (slight left)',
        'SE': 'Southeast (back right)',
        'SW': 'Southwest (back left)',
        'UP': 'Up (stairs/elevator)',
        'DOWN': 'Down (stairs/elevator)'
    }
    return mapping.get(direction, direction)


# ============================================
# CROWDSOURCED INTELLIGENCE ENDPOINTS
# ============================================

@router.post("/travel-report")
async def submit_travel_report(
    report: TravelReport,
    current_user: User = Depends(get_current_user)
):
    """
    Silent background report submitted automatically during navigation.
    This updates live crowd data and detects anomalies.
    User doesn't see this - it happens in background.
    """
    edge_key = _get_edge_key(report.building_id, report.from_node_id, report.to_node_id)
    
    # Calculate observed crowd level based on travel time vs expected
    if report.expected_steps > 0:
        time_ratio = report.actual_steps / report.expected_steps
        # If took 50% longer, crowd level ~3, if 100% longer, crowd level ~5
        observed_crowd = min(5, max(0, int((time_ratio - 1) * 10)))
    else:
        observed_crowd = report.crowd_level_observed
    
    # Store congestion report
    if edge_key not in _live_congestion_reports:
        _live_congestion_reports[edge_key] = []
    
    _live_congestion_reports[edge_key].append({
        'timestamp': datetime.utcnow().isoformat(),
        'crowd_level': observed_crowd,
        'user_id': str(current_user.id),
        'actual_steps': report.actual_steps,
        'expected_steps': report.expected_steps
    })
    
    # Keep only last 50 reports per edge
    _live_congestion_reports[edge_key] = _live_congestion_reports[edge_key][-50:]
    
    # Store travel time for anomaly detection
    if edge_key not in _travel_time_reports:
        _travel_time_reports[edge_key] = []
    
    _travel_time_reports[edge_key].append({
        'timestamp': datetime.utcnow().isoformat(),
        'actual_steps': report.actual_steps,
        'expected_steps': report.expected_steps,
        'travel_time': report.travel_time_seconds,
        'user_id': str(current_user.id)
    })
    
    # Keep only last 100 reports
    _travel_time_reports[edge_key] = _travel_time_reports[edge_key][-100:]
    
    # Check for anomaly
    anomaly_detected = detect_route_anomaly(
        report.building_id,
        report.from_node_id,
        report.to_node_id,
        report.travel_time_seconds,
        report.expected_steps  # Using steps as proxy for expected time
    )
    
    return {
        "received": True,
        "edge_key": edge_key,
        "current_crowd_level": get_live_crowd_level(report.building_id, report.from_node_id, report.to_node_id),
        "anomaly_detected": anomaly_detected
    }


@router.post("/landmark-report")
async def submit_landmark_report(
    report: LandmarkReport,
    current_user: User = Depends(get_current_user)
):
    """
    Silent background report when user views/captures landmark.
    If image hash differs significantly from stored, flag for review.
    """
    node_key = f"{report.building_id}:{report.node_id}"
    
    if node_key not in _landmark_change_reports:
        _landmark_change_reports[node_key] = []
    
    # Check if this is a potential change
    flagged = False
    if report.image_hash and _landmark_change_reports[node_key]:
        # Compare with recent hashes
        recent_hashes = [r['image_hash'] for r in _landmark_change_reports[node_key][-10:] if r.get('image_hash')]
        if recent_hashes and report.image_hash not in recent_hashes:
            # New hash detected - might be a change
            flagged = True
    
    if not report.landmark_visible:
        flagged = True  # User couldn't see expected landmark
    
    _landmark_change_reports[node_key].append({
        'timestamp': datetime.utcnow().isoformat(),
        'image_hash': report.image_hash,
        'user_id': str(current_user.id),
        'flagged': flagged,
        'landmark_visible': report.landmark_visible,
        'notes': report.notes
    })
    
    # Keep only last 50 reports
    _landmark_change_reports[node_key] = _landmark_change_reports[node_key][-50:]
    
    # Check if multiple users flagged this
    recent_flags = sum(1 for r in _landmark_change_reports[node_key][-10:] if r.get('flagged'))
    needs_review = recent_flags >= 3
    
    return {
        "received": True,
        "flagged": flagged,
        "needs_admin_review": needs_review
    }


@router.post("/blocked-path-report")
async def submit_blocked_path_report(
    report: BlockedPathReport,
    current_user: User = Depends(get_current_user)
):
    """
    Report when user encounters a blocked path.
    After multiple reports, path is marked as blocked and routes avoid it.
    """
    edge_key = _get_edge_key(report.building_id, report.from_node_id, report.to_node_id)
    
    if edge_key not in _blocked_paths:
        _blocked_paths[edge_key] = {
            'reported_at': datetime.utcnow().isoformat(),
            'reports_count': 0,
            'confirmed': False,
            'reason': report.reason,
            'reporters': []
        }
    
    # Don't count same user twice
    user_id = str(current_user.id)
    if user_id not in _blocked_paths[edge_key].get('reporters', []):
        _blocked_paths[edge_key]['reports_count'] += 1
        _blocked_paths[edge_key]['reporters'] = _blocked_paths[edge_key].get('reporters', []) + [user_id]
        _blocked_paths[edge_key]['reported_at'] = datetime.utcnow().isoformat()
    
    # Auto-confirm if enough reports
    if _blocked_paths[edge_key]['reports_count'] >= MIN_BLOCKED_REPORTS:
        _blocked_paths[edge_key]['confirmed'] = True
    
    return {
        "received": True,
        "edge_key": edge_key,
        "reports_count": _blocked_paths[edge_key]['reports_count'],
        "is_now_blocked": _blocked_paths[edge_key]['confirmed']
    }


@router.get("/buildings/{building_id}/live-conditions")
async def get_live_conditions(building_id: str):
    """
    Get real-time conditions for all edges in a building.
    Used by app to show live crowd levels and blocked paths.
    """
    _cleanup_old_reports()
    
    conditions = []
    
    # Get all edges for this building from congestion reports
    for edge_key, reports in _live_congestion_reports.items():
        if edge_key.startswith(f"{building_id}:"):
            parts = edge_key.split(":")
            if len(parts) >= 3:
                from_node = parts[1]
                to_node = parts[2]
                
                crowd_level = get_live_crowd_level(building_id, from_node, to_node)
                is_blocked = is_path_blocked(building_id, from_node, to_node)
                
                conditions.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "crowd_level": round(crowd_level, 1),
                    "is_blocked": is_blocked,
                    "reports_count": len(reports),
                    "last_report": reports[-1]['timestamp'] if reports else None
                })
    
    # Add blocked paths
    for edge_key, blocked_info in _blocked_paths.items():
        if edge_key.startswith(f"{building_id}:") and blocked_info.get('confirmed'):
            parts = edge_key.split(":")
            if len(parts) >= 3:
                from_node = parts[1]
                to_node = parts[2]
                
                # Check if already in conditions
                existing = next((c for c in conditions if c['from_node'] == from_node and c['to_node'] == to_node), None)
                if existing:
                    existing['is_blocked'] = True
                else:
                    conditions.append({
                        "from_node": from_node,
                        "to_node": to_node,
                        "crowd_level": 5.0,
                        "is_blocked": True,
                        "reports_count": blocked_info.get('reports_count', 0),
                        "reason": blocked_info.get('reason', 'blocked')
                    })
    
    return {
        "building_id": building_id,
        "conditions": conditions,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/buildings/{building_id}/indoor-graph/smart-route")
async def get_smart_route(
    building_id: str,
    from_node: str,
    to_node: str,
    accessible: bool = False,
    use_live_data: bool = True,
    step_length: float = DEFAULT_STEP_LENGTH,
    walking_speed: float = DEFAULT_WALKING_SPEED
):
    """
    Get route with LIVE crowd data and blocked path avoidance.
    This is the smart version that considers real-time conditions.
    """
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
    
    if not graph:
        raise HTTPException(status_code=404, detail="No indoor graph found")
    
    # Get route with live data
    if use_live_data:
        all_pairs = get_smart_route_with_live_data(graph, from_node, to_node, building_id, accessible)
    else:
        nodes_data = [n.dict() for n in graph.nodes]
        all_pairs = floyd_warshall(nodes_data, accessible_only=accessible)
    
    if from_node not in all_pairs:
        raise HTTPException(status_code=404, detail=f"Node '{from_node}' not found")
    
    if to_node not in all_pairs[from_node]:
        raise HTTPException(status_code=404, detail=f"Node '{to_node}' not found")
    
    route_info = all_pairs[from_node][to_node]
    
    if not route_info['reachable']:
        # Try to find alternate route
        raise HTTPException(status_code=404, detail="No route found - all paths may be blocked")
    
    # Build instructions with live conditions
    instructions = []
    path = route_info['path']
    directions = route_info['directions']
    nodes_dict = {n.id: n for n in graph.nodes}
    
    total_delay = 0
    blocked_segments = []
    
    for i, node_id in enumerate(path):
        node = nodes_dict.get(node_id)
        if not node:
            continue
        
        instruction = {
            "step": i + 1,
            "node_id": node_id,
            "label": node.label,
            "floor": node.floor_number,
            "image_url": node.image_url,
            "node_type": node.node_type,
            "qr_code": getattr(node, 'qr_code', None),
            "landmark": getattr(node, 'landmark_description', None),
        }
        
        if i < len(path) - 1:
            next_node_id = path[i + 1]
            
            # Get live conditions for this segment
            live_crowd = get_live_crowd_level(building_id, node_id, next_node_id)
            is_blocked = is_path_blocked(building_id, node_id, next_node_id)
            
            # Get base steps
            edge_steps = 0
            for edge in node.edges:
                if edge.to_node_id == next_node_id:
                    edge_steps = edge.steps
                    break
            
            # Calculate delay
            delay_seconds = int(live_crowd * 2)  # 2 seconds per crowd level
            total_delay += delay_seconds
            
            instruction["direction"] = directions[i] if i < len(directions) else None
            instruction["steps_to_next"] = edge_steps
            instruction["live_crowd_level"] = round(live_crowd, 1)
            instruction["is_blocked"] = is_blocked
            instruction["estimated_delay_seconds"] = delay_seconds
            
            if is_blocked:
                blocked_segments.append({"from": node_id, "to": next_node_id})
                instruction["warning"] = "⚠️ This path may be blocked"
            elif live_crowd >= 3:
                instruction["warning"] = f"🚶 Crowded area (level {int(live_crowd)})"
            
            instruction["instruction"] = f"Walk {edge_steps} steps {_direction_to_text(directions[i] if i < len(directions) else 'N')}"
        else:
            instruction["instruction"] = "🎉 You have arrived!"
        
        instructions.append(instruction)
    
    # Calculate ETA with delays
    base_eta = calculate_eta(route_info['total_steps'], step_length, walking_speed)
    adjusted_time = base_eta['time_seconds'] + total_delay
    
    return {
        "from": from_node,
        "to": to_node,
        "total_steps": route_info['total_steps'],
        "reachable": route_info['reachable'],
        "path": path,
        "directions": directions,
        "instructions": instructions,
        "eta": {
            "base_time_seconds": base_eta['time_seconds'],
            "delay_seconds": total_delay,
            "total_time_seconds": adjusted_time,
            "time_formatted": _format_time(adjusted_time)
        },
        "live_conditions": {
            "blocked_segments": blocked_segments,
            "total_delay_seconds": total_delay,
            "using_live_data": use_live_data
        }
    }


@router.post("/buildings/{building_id}/clear-blocked-path")
async def clear_blocked_path(
    building_id: str,
    from_node: str,
    to_node: str,
    current_user: User = Depends(get_current_user)
):
    """
    Report that a previously blocked path is now clear.
    Multiple reports will unblock the path.
    """
    edge_key = _get_edge_key(building_id, from_node, to_node)
    
    if edge_key in _blocked_paths:
        _blocked_paths[edge_key]['reports_count'] -= 1
        if _blocked_paths[edge_key]['reports_count'] <= 0:
            del _blocked_paths[edge_key]
            return {"message": "Path cleared", "is_blocked": False}
        else:
            _blocked_paths[edge_key]['confirmed'] = _blocked_paths[edge_key]['reports_count'] >= MIN_BLOCKED_REPORTS
    
    return {
        "message": "Report received",
        "is_blocked": is_path_blocked(building_id, from_node, to_node)
    }


@router.get("/buildings/{building_id}/landmark-changes")
async def get_landmark_changes(
    building_id: str,
    current_user: User = Depends(get_admin_user)
):
    """
    Admin endpoint to see nodes where landmarks may have changed.
    Based on user reports.
    """
    changes = []
    
    for node_key, reports in _landmark_change_reports.items():
        if node_key.startswith(f"{building_id}:"):
            node_id = node_key.split(":")[1]
            
            flagged_count = sum(1 for r in reports[-10:] if r.get('flagged'))
            invisible_count = sum(1 for r in reports[-10:] if not r.get('landmark_visible', True))
            
            if flagged_count >= 2 or invisible_count >= 2:
                changes.append({
                    "node_id": node_id,
                    "flagged_reports": flagged_count,
                    "invisible_reports": invisible_count,
                    "total_reports": len(reports),
                    "last_report": reports[-1]['timestamp'] if reports else None,
                    "needs_review": True
                })
    
    return {
        "building_id": building_id,
        "landmark_changes": changes,
        "total_flagged": len(changes)
    }


@router.get("/buildings/{building_id}/route-anomalies")
async def get_route_anomalies(
    building_id: str,
    current_user: User = Depends(get_admin_user)
):
    """
    Admin endpoint to see routes with detected anomalies.
    Helps identify areas that need attention.
    """
    anomalies = []
    
    for route_key, data in _route_anomalies.items():
        if route_key.startswith(f"{building_id}:"):
            if data.get('anomaly_detected'):
                parts = route_key.replace(f"{building_id}:", "").split("->")
                if len(parts) == 2:
                    times = data.get('times', [])
                    anomalies.append({
                        "from_node": parts[0],
                        "to_node": parts[1],
                        "avg_travel_time": statistics.mean(times) if times else 0,
                        "reports_count": len(times),
                        "anomaly_detected": True
                    })
    
    return {
        "building_id": building_id,
        "anomalies": anomalies,
        "total_anomalies": len(anomalies)
    }


# ============================================
# A/B TESTING ROUTES
# ============================================

from models import (
    ABTestExperiment, UserTestAssignment, ABTestResult,
    RouteVariant, UserFavorite, UserRecentRoute, SharedLocation,
    GraphVersion, MagneticCalibration, CrowdReport, 
    BlockedPathReport as BlockedPathModel, LandmarkChangeReport, RouteAnomaly
)

# A/B Test Schemas
class CreateExperimentRequest(BaseModel):
    """Request to create an A/B test experiment"""
    name: str
    description: Optional[str] = None
    variants: List[Dict[str, Any]]  # List of variant configurat