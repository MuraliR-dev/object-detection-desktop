import uuid
from typing import List, Dict

class TemporalTracker:
    """
    Tracks detections across consecutive frames to prevent flickering 
    and filter out spurious false positives.
    """
    def __init__(self, confirmation_frames: int = 3, max_missed_frames: int = 2, iou_thresh: float = 0.3):
        self.confirmation_frames = confirmation_frames
        self.max_missed_frames = max_missed_frames
        self.iou_thresh = iou_thresh
        self.tracks = []  # List of track dictionaries

    def update(self, detections: List[dict]) -> List[dict]:
        # 1. Decay all existing tracks
        for track in self.tracks:
            track['misses'] += 1

        # 2. Match incoming detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        for d_idx, det in enumerate(detections):
            best_iou = self.iou_thresh
            best_track_idx = -1
            
            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_tracks:
                    continue
                
                # Never match different classes!
                if track['class_id'] != det['class_id']:
                    continue
                
                iou = self._box_iou(det, track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = t_idx
            
            if best_track_idx != -1:
                matched_tracks.add(best_track_idx)
                matched_detections.add(d_idx)
                self.tracks[best_track_idx]['misses'] = 0
                self.tracks[best_track_idx]['hits'] += 1
                self.tracks[best_track_idx]['box'] = det
                self.tracks[best_track_idx]['confidence'] = det['confidence']

        # 3. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks.append({
                    'id': str(uuid.uuid4()),
                    'hits': 1,
                    'misses': 0,
                    'box': det,
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'source_model': det['source_model']
                })

        # 4. Filter tracks (remove lost tracks, only keep confirmed tracks)
        self.tracks = [t for t in self.tracks if t['misses'] <= self.max_missed_frames]
        
        confirmed = []
        for t in self.tracks:
            if t['hits'] >= self.confirmation_frames:
                # Output the detection
                out_det = t['box'].copy()
                out_det['track_id'] = t['id']
                confirmed.append(out_det)
                
        return confirmed

    def _box_iou(self, a: dict, b: dict) -> float:
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
