from typing import Dict, Any, List, Optional

class SeverityEstimator:
    """
    Estimates fire severity based on visual bounding box area, smoke intensity, temporal growth, and satellite FRP.
    NOTE: RGB bounding box area is a visual 2D proxy and is NOT physically equivalent to ground fire area (hectares/acres).
    """
    def estimate_severity(self, boxes: List[Dict[str, Any]],
                          smoke_confidence: float = 0.0,
                          temporal_growth_rate: float = 0.0,
                          satellite_frp: Optional[float] = None) -> Dict[str, Any]:
        
        fire_boxes = [b for b in boxes if b.get("class") == "fire"]
        total_bbox_area = 0.0
        
        for box in fire_boxes:
            b = box.get("bbox", [0, 0, 0, 0])
            w = max(0, b[2] - b[0])
            h = max(0, b[3] - b[1])
            total_bbox_area += (w * h)

        num_fire_regions = len(fire_boxes)
        
        # Calculate severity index (0.0 to 1.0)
        area_score = min(1.0, total_bbox_area / 100000.0) # normalized pixel area proxy
        growth_score = min(1.0, max(0.0, temporal_growth_rate))
        frp_score = min(1.0, (satellite_frp or 0.0) / 150.0)

        severity_index = (
            0.35 * area_score +
            0.25 * smoke_confidence +
            0.25 * growth_score +
            0.15 * frp_score
        )

        if severity_index >= 0.70:
            severity_level = "CRITICAL"
        elif severity_index >= 0.45:
            severity_level = "HIGH"
        elif severity_index >= 0.20:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        return {
            "severity_level": severity_level,
            "severity_index": round(float(severity_index), 4),
            "num_fire_regions": num_fire_regions,
            "smoke_intensity": round(float(smoke_confidence), 4),
            "temporal_growth_rate": round(float(temporal_growth_rate), 4),
            "satellite_frp_mw": satellite_frp if satellite_frp is not None else 0.0,
            "limitation_warning": "RGB bounding-box pixel area is a visual 2D projection and is NOT physically equivalent to actual ground fire surface area."
        }
