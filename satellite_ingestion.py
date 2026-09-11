import os
import requests
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional

class NASASatelliteIngestion:
    """
    Ingestion interface for NASA MODIS and VIIRS satellite active fire hotspots (via FIRMS API or local CSVs).
    Supports MODE=MOCK for offline testing, strictly labeling mock records.
    """
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or os.getenv("MODE", "MOCK")
        self.api_key = os.getenv("NASA_FIRMS_MAP_KEY", "")

    def get_active_hotspots(self, min_lat: float = -90.0, max_lat: float = 90.0,
                             min_lon: float = -180.0, max_lon: float = 180.0,
                             days: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves active fire hotspots from NASA FIRMS API or fallback local dataset.
        Preserves latitude, longitude, timestamp, confidence, FRP, satellite/platform.
        """
        if self.mode == "MOCK" or not self.api_key:
            return self._get_mock_hotspots(min_lat, max_lat, min_lon, max_lon)

        try:
            # Query real NASA FIRMS API
            url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{self.api_key}/VIIRS_NRT/USA/{days}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 50:
                df = pd.read_csv(pd.compat.StringIO(response.text))
                filtered = df[
                    (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
                ]
                results = []
                for _, row in filtered.iterrows():
                    results.append({
                        "id": f"NASA-{row.get('acq_date', 'N/A')}-{row.get('latitude')}",
                        "latitude": float(row.get('latitude')),
                        "longitude": float(row.get('longitude')),
                        "timestamp": f"{row.get('acq_date', '2026-09-11')} {row.get('acq_time', '12:00')}",
                        "confidence": float(row.get('confidence', 85.0)) if str(row.get('confidence')).replace('.','',1).isdigit() else 80.0,
                        "FRP": float(row.get('frp', 45.2)),
                        "satellite": str(row.get('satellite', 'VIIRS')),
                        "source": "REAL NASA FIRMS DATA",
                        "is_mock": False
                    })
                return results
            else:
                return self._get_mock_hotspots(min_lat, max_lat, min_lon, max_lon)
        except Exception as e:
            print(f"[NASA Ingestion Warning] Real NASA API call failed ({e}). Falling back to local MOCK mode.")
            return self._get_mock_hotspots(min_lat, max_lat, min_lon, max_lon)

    def _get_mock_hotspots(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[Dict[str, Any]]:
        """
        Returns realistic sample hotspot records clearly labeled as MOCK DATA.
        """
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        mock_records = [
            {
                "id": "MOCK-HOTSPOT-001",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "timestamp": now_str,
                "confidence": 92.5,
                "FRP": 118.4,
                "satellite": "VIIRS-NPP",
                "source": "MOCK LOCAL NASA DATA (DEMO ONLY)",
                "is_mock": True
            },
            {
                "id": "MOCK-HOTSPOT-002",
                "latitude": 37.7833,
                "longitude": -122.4167,
                "timestamp": now_str,
                "confidence": 84.0,
                "FRP": 76.2,
                "satellite": "MODIS-Terra",
                "source": "MOCK LOCAL NASA DATA (DEMO ONLY)",
                "is_mock": True
            },
            {
                "id": "MOCK-HOTSPOT-003",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "timestamp": now_str,
                "confidence": 68.0,
                "FRP": 32.1,
                "satellite": "VIIRS-NOAA20",
                "source": "MOCK LOCAL NASA DATA (DEMO ONLY)",
                "is_mock": True
            }
        ]
        
        # Filter mock records to bounding box if required
        results = [
            r for r in mock_records
            if min_lat <= r["latitude"] <= max_lat and min_lon <= r["longitude"] <= max_lon
        ]
        return results if results else mock_records
