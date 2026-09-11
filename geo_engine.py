from typing import Dict, Any, List, Tuple

class GeolocationEngine:
    """
    Geographic location manager binding camera metadata and satellite lat/lon records.
    """
    def __init__(self, default_lat: float = 37.7749, default_lon: float = -122.4194):
        self.default_lat = default_lat
        self.default_lon = default_lon

    def map_camera_detection(self, camera_lat: float, camera_lon: float,
                             detection_source: str = "Camera 1") -> Dict[str, Any]:
        return {
            "latitude": camera_lat,
            "longitude": camera_lon,
            "source": detection_source,
            "crs": "EPSG:4326 (WGS84)"
        }

    def map_satellite_hotspot(self, lat: float, lon: float, sat_name: str, frp: float) -> Dict[str, Any]:
        return {
            "latitude": lat,
            "longitude": lon,
            "satellite": sat_name,
            "frp": frp,
            "crs": "EPSG:4326 (WGS84)"
        }
