from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from models import User, Building, Floor, Room, Waypoint, ARMarker
from schemas import BuildingCreate
from auth_utils import get_current_user
from bson import ObjectId
from datetime import datetime

router = APIRouter()

@router.get("/buildings/{building_id}/download")
async def download_building_for_offline(
    building_id: str,
    client_version: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    Download complete building data for offline navigation.
    Returns all floors, rooms, waypoints, AR markers, AND navigation graph data.
    """
    try:
        # Get building
        building = await Building.get(building_id)
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
            
        # Check version
        if building.version <= client_version:
            return {"status": "not_modified", "version": building.version, "message": "Building is up to date"}
        
        # Get all floors for this building
        floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
        
        # Get all rooms, waypoints, and markers
        rooms = []
        waypoints = []
        markers = []
        
        for floor in floors:
            # Get rooms for this floor
            floor_rooms = await Room.find(Room.floor_id == floor.id).to_list()
            rooms.extend(floor_rooms)
            
            # Get waypoints for this floor
            floor_waypoints = await Waypoint.find(Waypoint.floor_id == floor.id).to_list()
            waypoints.extend(floor_waypoints)
            
            # Get AR markers for this floor
            floor_markers = await ARMarker.find(ARMarker.floor_id == floor.id).to_list()
            markers.extend(floor_markers)
        
        # Get navigation graph (indoor graph)
        navigation_graph = None
        navigation_nodes = []
        shortest_paths = {}
        
        try:
            from models import IndoorGraph, BuildingGraph
            
            # Try IndoorGraph first
            indoor_graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
            if indoor_graph and indoor_graph.nodes:
                navigation_graph = {
                    "type": "indoor_graph",
                    "nodes": [n.dict() if hasattr(n, 'dict') else n for n in indoor_graph.nodes],
                    "shortest_paths": indoor_graph.shortest_paths if hasattr(indoor_graph, 'shortest_paths') else {},
                    "accessible_paths": indoor_graph.accessible_paths if hasattr(indoor_graph, 'accessible_paths') else {},
                }
                
                # Normalize node format for consistency
                normalized_nodes = []
                for node in navigation_graph["nodes"]:
                    normalized_node = {
                        "id": node.get("id", ""),
                        "label": node.get("label", ""),
                        "name": node.get("label", ""),
                        "building_id": building_id,
                        "floor_id": None,  # IndoorGraph nodes don't have floor_id
                        "x": node.get("latitude", 0),
                        "y": node.get("longitude", 0),
                        "z": node.get("floor_number", 0),
                        "latitude": node.get("latitude", 0),
                        "longitude": node.get("longitude", 0),
                        "floor_number": node.get("floor_number", 0),
                        "type": node.get("node_type", "waypoint"),
                        "node_type": node.get("node_type", "waypoint"),
                        "image_url": node.get("image_url"),
                        "qr_code": node.get("qr_code", f"indoor-nav://{building_id}/{node.get('id', '')}"),
                        "is_accessible": node.get("is_accessible", True),
                        "is_emergency_exit": node.get("is_emergency_exit", False),
                        "landmark_description": node.get("landmark_description"),
                        "category": node.get("category"),
                        "edges": node.get("edges", []),
                        "connected_node_ids": [e.get("to_node_id") for e in node.get("edges", [])],
                        "neighbors": [e.get("to_node_id") for e in node.get("edges", [])],
                        "distances": {e.get("to_node_id"): e.get("steps", 0) * 0.7 for e in node.get("edges", [])},
                    }
                    normalized_nodes.append(normalized_node)
                
                navigation_nodes = normalized_nodes
                shortest_paths = navigation_graph.get("shortest_paths", {})
            
            # Try BuildingGraph as fallback
            if not navigation_graph:
                building_graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
                if building_graph and building_graph.nodes:
                    navigation_graph = {
                        "type": "building_graph",
                        "nodes": [n.dict() if hasattr(n, 'dict') else n for n in building_graph.nodes],
                    }
                    navigation_nodes = navigation_graph["nodes"]
        except Exception as e:
            print(f"⚠️ Could not load navigation graph: {e}")
        
        # Convert waypoints to navigation nodes if no graph exists
        if not navigation_nodes and waypoints:
            print(f"📍 Converting {len(waypoints)} waypoints to navigation nodes")
            for idx, waypoint in enumerate(waypoints):
                # Create navigation node with connections
                connected_nodes = []
                distances = {}
                
                # Connect to nearby waypoints on same floor (within 15 meters)
                for other_idx, other_waypoint in enumerate(waypoints):
                    if other_idx != idx and waypoint.floor_id == other_waypoint.floor_id:
                        # Calculate distance
                        dx = waypoint.latitude - other_waypoint.latitude
                        dy = waypoint.longitude - other_waypoint.longitude
                        distance = (dx**2 + dy**2)**0.5 * 111000  # Convert to meters
                        
                        if distance < 15:  # Within 15 meters
                            connected_nodes.append(str(other_waypoint.id))
                            distances[str(other_waypoint.id)] = distance
                
                navigation_nodes.append({
                    "id": str(waypoint.id),
                    "label": waypoint.name,
                    "building_id": building_id,
                    "floor_id": str(waypoint.floor_id),
                    "x": waypoint.latitude,
                    "y": waypoint.longitude,
                    "z": waypoint.floor_number,
                    "type": waypoint.waypoint_type or "corridor",
                    "node_type": waypoint.waypoint_type or "waypoint",
                    "connected_node_ids": connected_nodes,
                    "neighbors": connected_nodes,
                    "distances": distances,
                    "name": waypoint.name,
                    "image_url": waypoint.images[0] if waypoint.images else None,
                    "qr_code": f"indoor-nav://{building_id}/{waypoint.id}",
                    "is_accessible": True,
                    "category": None,
                })
        
        # Get all available locations (for smart navigation)
        locations = []
        for node in navigation_nodes:
            locations.append({
                "id": node.get("id"),
                "name": node.get("label") or node.get("name"),
                "node_type": node.get("node_type") or node.get("type"),
                "floor_number": int(node.get("z", 0)),
                "latitude": float(node.get("x", 0)),
                "longitude": float(node.get("y", 0)),
                "image_url": node.get("image_url"),
                "neighbors": node.get("neighbors", []),
                "category": node.get("category"),
            })
        
        # Get available categories
        categories = list(set(
            node.get("category") 
            for node in navigation_nodes 
            if node.get("category")
        ))
        
        # Prepare response
        response = {
            "building": {
                "id": str(building.id),
                "name": building.name,
                "description": building.description,
                "address": building.address,
                "latitude": building.latitude,
                "longitude": building.longitude,
                "downloaded_at": datetime.utcnow().isoformat(),
                "version": building.version,
            },
            "floors": [
                {
                    "id": str(floor.id),
                    "building_id": building_id,
                    "floor_number": floor.floor_number,
                    "name": floor.name,
                    "width": 50.0,  # Default width in meters
                    "height": 40.0,  # Default height in meters
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                }
                for floor in floors
            ],
            "rooms": [
                {
                    "id": str(room.id),
                    "floor_id": str(room.floor_id),
                    "building_id": building_id,
                    "name": room.name,
                    "type": room.room_type,
                    "x": room.coordinates.get("lat", 0),
                    "y": room.coordinates.get("lng", 0),
                    "width": room.coordinates.get("width", 5.0),
                    "height": room.coordinates.get("length", 4.0),
                    "entrance_node_id": None,
                }
                for room in rooms
            ],
            "navigation_nodes": navigation_nodes,
            "qr_markers": [
                {
                    "id": str(marker.id),
                    "building_id": building_id,
                    "floor_id": str(marker.floor_id),
                    "x": marker.latitude,
                    "y": marker.longitude,
                    "orientation_degrees": 0.0,
                    "qr_data": marker.marker_data,
                    "description": marker.description,
                }
                for marker in markers
            ],
            "navigation_graph": navigation_graph,
            "shortest_paths": shortest_paths,
            "locations": locations,
            "categories": categories,
            "metadata": {
                "total_floors": len(floors),
                "total_rooms": len(rooms),
                "total_waypoints": len(waypoints),
                "total_markers": len(markers),
                "total_navigation_nodes": len(navigation_nodes),
                "has_navigation_graph": navigation_graph is not None,
                "has_shortest_paths": len(shortest_paths) > 0,
                "available_categories": len(categories),
                "download_timestamp": datetime.utcnow().isoformat(),
            }
        }
        
        return response
        
    except Exception as e:
        print(f"❌ Error downloading building data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading building data: {str(e)}")


@router.get("/buildings")
async def get_available_buildings_for_offline(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of buildings available for offline download.
    Returns basic building info and download size estimate.
    """
    try:
        buildings = await Building.find_all().to_list()
        
        result = []
        for building in buildings:
            # Count resources
            floors = await Floor.find(Floor.building_id == building.id).to_list()
            
            total_rooms = 0
            total_waypoints = 0
            total_markers = 0
            
            # CRITICAL FIX: Check IndoorGraph first for navigation nodes
            try:
                from models import IndoorGraph
                indoor_graph = await IndoorGraph.find_one(IndoorGraph.building_id == building.id)
                if indoor_graph and indoor_graph.nodes:
                    total_waypoints = len(indoor_graph.nodes)
                    print(f"✅ Building {building.name}: Found {total_waypoints} nodes in IndoorGraph")
            except Exception as e:
                print(f"⚠️ Could not check IndoorGraph for {building.name}: {e}")
            
            # Fallback to Waypoint table if no IndoorGraph
            if total_waypoints == 0:
                for floor in floors:
                    rooms = await Room.find(Room.floor_id == floor.id).to_list()
                    waypoints = await Waypoint.find(Waypoint.floor_id == floor.id).to_list()
                    markers = await ARMarker.find(ARMarker.floor_id == floor.id).to_list()
                    
                    total_rooms += len(rooms)
                    total_waypoints += len(waypoints)
                    total_markers += len(markers)
            else:
                # Still count rooms and markers from floors
                for floor in floors:
                    rooms = await Room.find(Room.floor_id == floor.id).to_list()
                    markers = await ARMarker.find(ARMarker.floor_id == floor.id).to_list()
                    
                    total_rooms += len(rooms)
                    total_markers += len(markers)
            
            # Estimate download size (rough estimate in KB)
            estimated_size_kb = (
                len(floors) * 2 +  # Floor data
                total_rooms * 1 +  # Room data
                total_waypoints * 1 +  # Waypoint/Node data
                total_markers * 2  # Marker data
            )
            
            # Ensure minimum size display
            if estimated_size_kb == 0 and (len(floors) > 0 or total_rooms > 0):
                estimated_size_kb = 5  # Minimum 5KB for buildings with data
            
            result.append({
                "id": str(building.id),
                "name": building.name,
                "address": building.address or "No address",
                "description": building.description or "",
                "floors_count": len(floors),
                "rooms_count": total_rooms,
                "waypoints_count": total_waypoints,
                "markers_count": total_markers,
                "estimated_size_kb": estimated_size_kb,
                "version": getattr(building, 'version', 1),
                "is_ready_for_offline": total_waypoints > 0,  # Has navigation data
                "created_at": building.created_at.isoformat() if hasattr(building, 'created_at') else None,
                "latitude": building.latitude if hasattr(building, 'latitude') else None,
                "longitude": building.longitude if hasattr(building, 'longitude') else None,
            })
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching buildings for offline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching buildings: {str(e)}")


@router.post("/buildings/{building_id}/qr-markers")
async def create_qr_marker(
    building_id: str,
    marker_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Create QR marker for offline navigation calibration.
    Admin only.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Create AR marker with QR code data
        marker = ARMarker(
            floor_id=ObjectId(marker_data["floor_id"]),
            latitude=marker_data["x"],
            longitude=marker_data["y"],
            floor_number=marker_data["floor"],
            marker_type="qr_code",
            marker_data=marker_data["qr_data"],
            description=marker_data.get("description", "QR calibration marker"),
        )
        await marker.save()
        
        return {
            "id": str(marker.id),
            "message": "QR marker created successfully",
            "marker": {
                "id": str(marker.id),
                "floor_id": str(marker.floor_id),
                "x": marker.latitude,
                "y": marker.longitude,
                "floor": marker.floor_number,
                "qr_data": marker.marker_data,
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating QR marker: {str(e)}")


@router.delete("/buildings/{building_id}/qr-markers/{marker_id}")
async def delete_qr_marker(
    building_id: str,
    marker_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete QR marker.
    Admin only.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        marker = await ARMarker.get(marker_id)
        if not marker:
            raise HTTPException(status_code=404, detail="Marker not found")
        
        await marker.delete()
        
        return {"message": "QR marker deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting QR marker: {str(e)}")


@router.post("/sync/buildings")
async def sync_offline_building(
    building_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Sync an offline-created building to the server.
    This endpoint allows any authenticated user to sync buildings they created offline.
    """
    try:
        # Validate required fields
        required_fields = ['name', 'address', 'latitude', 'longitude']
        for field in required_fields:
            if field not in building_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create building
        db_building = Building(
            name=building_data['name'],
            description=building_data.get('description', ''),
            address=building_data['address'],
            latitude=building_data['latitude'],
            longitude=building_data['longitude'],
            boundary_points=building_data.get('boundary_points', []),
            created_by=current_user.id,
            created_at=datetime.utcnow(),
        )
        await db_building.insert()
        
        return {
            "success": True,
            "message": "Building synced successfully",
            "id": str(db_building.id),
            "building": {
                "id": str(db_building.id),
                "name": db_building.name,
                "address": db_building.address,
                "latitude": db_building.latitude,
                "longitude": db_building.longitude,
                "created_at": db_building.created_at.isoformat(),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing building: {str(e)}")


@router.post("/sync/batch")
async def sync_batch_offline_data(
    sync_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Batch sync multiple offline items.
    Accepts buildings, coordinates, waypoints, etc.
    """
    results = {
        "buildings_synced": 0,
        "buildings_failed": 0,
        "errors": [],
        "synced_items": []
    }
    
    try:
        # Sync buildings
        buildings = sync_data.get('buildings', [])
        for building_data in buildings:
            try:
                db_building = Building(
                    name=building_data['name'],
                    description=building_data.get('description', ''),
                    address=building_data['address'],
                    latitude=building_data['latitude'],
                    longitude=building_data['longitude'],
                    boundary_points=building_data.get('boundary_points', []),
                    created_by=current_user.id,
                    created_at=datetime.utcnow(),
                )
                await db_building.insert()
                results['buildings_synced'] += 1
                results['synced_items'].append({
                    'type': 'building',
                    'local_id': building_data.get('local_id'),
                    'server_id': str(db_building.id),
                    'name': db_building.name,
                })
            except Exception as e:
                results['buildings_failed'] += 1
                results['errors'].append(f"Building '{building_data.get('name', 'unknown')}': {str(e)}")
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch sync failed: {str(e)}")


@router.get("/buildings/{building_id}/debug")
async def debug_building_data(
    building_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Debug endpoint to check building data completeness.
    Returns detailed information about what data exists.
    """
    try:
        building = await Building.get(building_id)
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
        
        floor_details = []
        total_waypoints = 0
        total_rooms = 0
        total_markers = 0
        
        for floor in floors:
            waypoints = await Waypoint.find(Waypoint.floor_id == floor.id).to_list()
            rooms = await Room.find(Room.floor_id == floor.id).to_list()
            markers = await ARMarker.find(ARMarker.floor_id == floor.id).to_list()
            
            total_waypoints += len(waypoints)
            total_rooms += len(rooms)
            total_markers += len(markers)
            
            floor_details.append({
                "floor_id": str(floor.id),
                "floor_number": floor.floor_number,
                "floor_name": floor.name,
                "waypoints_count": len(waypoints),
                "rooms_count": len(rooms),
                "markers_count": len(markers),
                "waypoints": [
                    {
                        "id": str(wp.id),
                        "name": wp.name,
                        "type": wp.waypoint_type,
                        "lat": wp.latitude,
                        "lng": wp.longitude,
                    }
                    for wp in waypoints[:5]  # Show first 5
                ] if waypoints else [],
            })
        
        return {
            "building": {
                "id": str(building.id),
                "name": building.name,
                "address": building.address,
                "version": getattr(building, 'version', 1),
            },
            "summary": {
                "total_floors": len(floors),
                "total_waypoints": total_waypoints,
                "total_rooms": total_rooms,
                "total_markers": total_markers,
                "is_ready_for_offline": total_waypoints > 0,
            },
            "floors": floor_details,
            "issues": [
                "No waypoints found - building cannot be downloaded for offline use" if total_waypoints == 0 else None,
                "No floors found - building structure is incomplete" if len(floors) == 0 else None,
                "No rooms found - building may not have room data" if total_rooms == 0 else None,
            ],
            "recommendations": [
                "Add waypoints using the coordinate collection page" if total_waypoints == 0 else None,
                "Add at least 2-5 waypoints per floor for navigation" if total_waypoints < len(floors) * 2 else None,
                "Ensure waypoints are properly connected (within 15 meters)" if total_waypoints > 0 else None,
            ],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
