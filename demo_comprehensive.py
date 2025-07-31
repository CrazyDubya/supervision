#!/usr/bin/env python3
"""
Comprehensive Supervision Demo: Smart Traffic Analytics System

This demo showcases the full functionality of the Supervision library by creating
a comprehensive traffic analytics system that demonstrates:

1. Multi-object detection and tracking with real models (YOLO, Roboflow) or mock data
2. Zone-based analytics (entry/exit counting, time spent in zones)
3. Speed estimation using perspective transformation
4. Movement heatmap generation
5. Rich annotations with multiple annotators
6. Video processing pipeline
7. Metrics collection and data export
8. Real-time performance monitoring

Features demonstrated:
- Real object detection with YOLO or Roboflow Inference models
- Mock detection model for testing and demonstration
- ByteTracker for consistent tracking
- PolygonZone and LineZone for spatial analytics
- Speed calculation with perspective transformation
- HeatMapAnnotator for visualization
- Multiple annotation types (Box, Label, Trace, etc.)
- Video processing with VideoSink
- CSV and JSON data export
- FPS monitoring
- Dataset operations

Model Support:
- YOLO models (via ultralytics): Local .pt files, automatic download
- Roboflow Inference: Cloud-hosted models via API
- Mock model: For testing and demonstration purposes

Author: Supervision Demo
"""

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv

# Color palette for visualizations
COLORS = sv.ColorPalette.from_hex([
    "#E6194B",  # Red
    "#3CB44B",  # Green  
    "#FFE119",  # Yellow
    "#3C76D1",  # Blue
    "#F58231",  # Orange
    "#911EB4",  # Purple
    "#42D4F4",  # Cyan
    "#F032E6",  # Magenta
])

@dataclass
class AnalyticsConfig:
    """Configuration for analytics system"""
    # Zone definitions (can be customized for different camera angles)
    entry_zones: List[np.ndarray] = field(default_factory=lambda: [
        np.array([[100, 100], [300, 100], [300, 200], [100, 200]]),  # Entry zone 1
        np.array([[500, 100], [700, 100], [700, 200], [500, 200]]),  # Entry zone 2
    ])
    
    exit_zones: List[np.ndarray] = field(default_factory=lambda: [
        np.array([[100, 400], [300, 400], [300, 500], [100, 500]]),  # Exit zone 1
        np.array([[500, 400], [700, 400], [700, 500], [500, 500]]),  # Exit zone 2
    ])
    
    # Speed estimation setup (perspective transformation)
    speed_source_points: np.ndarray = field(default_factory=lambda: np.array([
        [100, 200], [700, 200], [700, 400], [100, 400]
    ]))
    
    speed_target_points: np.ndarray = field(default_factory=lambda: np.array([
        [0, 0], [100, 0], [100, 200], [0, 200]
    ]))
    
    # Real-world measurements for speed calculation (in meters)
    real_world_width: float = 50.0
    real_world_height: float = 25.0
    
    # Tracking parameters
    track_buffer: int = 30
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    
    # Detection filtering
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.7
    
    # Analytics parameters  
    min_time_in_zone: float = 1.0  # seconds
    speed_smoothing_window: int = 5
    
    # Visualization
    line_thickness: int = 2
    text_scale: float = 0.5
    annotation_text_padding: int = 10


@dataclass
class DetectionMetrics:
    """Metrics for a single detection/track"""
    track_id: int
    class_id: int
    confidence: float
    first_seen: float
    last_seen: float
    total_detections: int = 0
    zone_entries: Dict[int, float] = field(default_factory=dict)
    zone_exits: Dict[int, float] = field(default_factory=dict)
    speeds: List[float] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def average_speed(self) -> float:
        return sum(self.speeds) / len(self.speeds) if self.speeds else 0.0
    
    @property
    def lifetime(self) -> float:
        return self.last_seen - self.first_seen


class ViewTransformer:
    """Handles perspective transformation for speed estimation"""
    
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray):
        self.source = source_points.astype(np.float32)
        self.target = target_points.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.source, self.target)
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from source to target perspective"""
        if points.size == 0:
            return points
            
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
        return transformed_points.reshape(-1, 2)


class SpeedEstimator:
    """Estimates object speeds using perspective transformation"""
    
    def __init__(self, transformer: ViewTransformer, real_width: float, real_height: float):
        self.transformer = transformer
        self.real_width = real_width  # meters
        self.real_height = real_height  # meters
        self.position_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        
    def update(self, detections: sv.Detections, timestamp: float) -> Dict[int, float]:
        """Update speed estimates for tracked objects"""
        speeds = {}
        
        if detections.tracker_id is None:
            return speeds
            
        centroids = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        transformed_centroids = self.transformer.transform_points(centroids)
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is not None:
                x, y = transformed_centroids[i]
                self.position_history[track_id].append((x, y, timestamp))
                
                # Calculate speed if we have enough history
                if len(self.position_history[track_id]) >= 2:
                    speeds[track_id] = self._calculate_speed(track_id)
                    
        return speeds
    
    def _calculate_speed(self, track_id: int) -> float:
        """Calculate speed for a specific track"""
        history = self.position_history[track_id]
        if len(history) < 2:
            return 0.0
            
        # Use recent positions for speed calculation
        (x1, y1, t1) = history[-2]
        (x2, y2, t2) = history[-1]
        
        # Convert pixel distance to real-world distance
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        real_distance = pixel_distance * (self.real_width / self.pixels_per_unit)
        
        # Calculate speed in m/s
        time_diff = t2 - t1
        if time_diff > 0:
            speed_ms = real_distance / time_diff
            return speed_ms * 3.6  # Convert to km/h
        
        return 0.0


class ZoneAnalyzer:
    """Analyzes object behavior in defined zones"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.entry_zones = [sv.PolygonZone(polygon=zone) for zone in config.entry_zones]
        self.exit_zones = [sv.PolygonZone(polygon=zone) for zone in config.exit_zones]
        self.zone_timers: Dict[int, Dict[int, float]] = defaultdict(dict)
        
    def update(self, detections: sv.Detections, timestamp: float) -> Dict[str, any]:
        """Update zone analytics"""
        results = {
            'entries': [],
            'exits': [],
            'current_occupancy': []
        }
        
        if detections.tracker_id is None:
            return results
            
        # Check entry zones
        for zone_id, zone in enumerate(self.entry_zones):
            mask = zone.trigger(detections)
            triggered_tracks = detections.tracker_id[mask]
            
            for track_id in triggered_tracks:
                if track_id not in self.zone_timers[zone_id]:
                    self.zone_timers[zone_id][track_id] = timestamp
                    results['entries'].append({
                        'zone_id': zone_id,
                        'track_id': track_id,
                        'timestamp': timestamp
                    })
        
        # Check exit zones  
        for zone_id, zone in enumerate(self.exit_zones):
            mask = zone.trigger(detections)
            triggered_tracks = detections.tracker_id[mask]
            
            for track_id in triggered_tracks:
                if track_id in self.zone_timers.get(zone_id, {}):
                    entry_time = self.zone_timers[zone_id][track_id]
                    time_in_zone = timestamp - entry_time
                    
                    results['exits'].append({
                        'zone_id': zone_id,
                        'track_id': track_id,
                        'timestamp': timestamp,
                        'time_in_zone': time_in_zone
                    })
                    
                    del self.zone_timers[zone_id][track_id]
        
        # Current occupancy
        for zone_id in range(len(self.entry_zones)):
            current_count = len(self.zone_timers.get(zone_id, {}))
            results['current_occupancy'].append({
                'zone_id': zone_id,
                'count': current_count
            })
            
        return results


class SmartTrafficAnalytics:
    """Main analytics system combining all features"""
    
    def __init__(self, config: AnalyticsConfig, model=None):
        self.config = config
        self.metrics: Dict[int, DetectionMetrics] = {}
        self.model = model  # Store model reference for class names
        
        # Initialize components
        self.transformer = ViewTransformer(
            config.speed_source_points, 
            config.speed_target_points
        )
        self.speed_estimator = SpeedEstimator(
            self.transformer, 
            config.real_world_width, 
            config.real_world_height
        )
        self.zone_analyzer = ZoneAnalyzer(config)
        
        # Initialize tracker
        self.tracker = sv.ByteTrack(
            frame_rate=config.track_buffer,
            track_thresh=config.track_thresh,
            match_thresh=config.match_thresh
        )
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.heatmap_annotator = sv.HeatMapAnnotator()
        
        # Zone annotators
        self.entry_zone_annotators = [
            sv.PolygonZoneAnnotator(zone=zone, color=COLORS.colors[i % len(COLORS.colors)])
            for i, zone in enumerate(self.zone_analyzer.entry_zones)
        ]
        self.exit_zone_annotators = [
            sv.PolygonZoneAnnotator(zone=zone, color=COLORS.colors[(i + len(self.entry_zone_annotators)) % len(COLORS.colors)])
            for i, zone in enumerate(self.zone_analyzer.exit_zones)
        ]
        
        # Analytics storage
        self.frame_analytics = []
        
    def process_frame(self, frame: np.ndarray, detections: sv.Detections, timestamp: float) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and return annotated frame with analytics"""
        
        # Filter detections by confidence
        confidence_mask = detections.confidence >= self.config.confidence_threshold
        detections = detections[confidence_mask]
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Update speed estimates
        speeds = self.speed_estimator.update(detections, timestamp)
        
        # Update zone analytics
        zone_results = self.zone_analyzer.update(detections, timestamp)
        
        # Update metrics for each detection
        self._update_metrics(detections, speeds, timestamp, zone_results)
        
        # Create annotations
        annotated_frame = self._annotate_frame(frame, detections, speeds, zone_results)
        
        # Compile frame analytics
        frame_analytics = {
            'timestamp': float(timestamp),
            'detection_count': int(len(detections)),
            'speeds': {int(k): float(v) for k, v in speeds.items()},
            'zone_analytics': zone_results,
            'active_tracks': int(len(set(detections.tracker_id)) if detections.tracker_id is not None else 0)
        }
        
        self.frame_analytics.append(frame_analytics)
        
        return annotated_frame, frame_analytics
    
    def _update_metrics(self, detections: sv.Detections, speeds: Dict[int, float], timestamp: float, zone_results: Dict):
        """Update metrics for tracked objects"""
        if detections.tracker_id is None:
            return
            
        centroids = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is not None:
                if track_id not in self.metrics:
                    self.metrics[track_id] = DetectionMetrics(
                        track_id=track_id,
                        class_id=detections.class_id[i] if detections.class_id is not None else 0,
                        confidence=detections.confidence[i],
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                
                # Update existing metrics
                metric = self.metrics[track_id]
                metric.last_seen = timestamp
                metric.total_detections += 1
                metric.positions.append(tuple(centroids[i]))
                
                # Add speed if available
                if track_id in speeds:
                    metric.speeds.append(speeds[track_id])
                
                # Update zone information
                for entry in zone_results['entries']:
                    if entry['track_id'] == track_id:
                        metric.zone_entries[entry['zone_id']] = timestamp
                        
                for exit_info in zone_results['exits']:
                    if exit_info['track_id'] == track_id:
                        metric.zone_exits[exit_info['zone_id']] = timestamp
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections, speeds: Dict[int, float], zone_results: Dict) -> np.ndarray:
        """Apply all annotations to frame"""
        annotated_frame = frame.copy()
        
        # Draw zones first (background)
        for annotator in self.entry_zone_annotators:
            annotated_frame = annotator.annotate(annotated_frame)
            
        for annotator in self.exit_zone_annotators:
            annotated_frame = annotator.annotate(annotated_frame)
        
        # Draw detections
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        
        # Create labels with speed information
        labels = []
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is not None:
                    # Get class name from model if available
                    class_id = detections.class_id[i] if detections.class_id is not None else 0
                    if self.model and hasattr(self.model, 'class_names') and self.model.class_names:
                        if isinstance(self.model.class_names, dict):
                            class_name = self.model.class_names.get(class_id, f"Class_{class_id}")
                        elif isinstance(self.model.class_names, list) and class_id < len(self.model.class_names):
                            class_name = self.model.class_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                    else:
                        class_name = f"Object_{class_id}"
                    
                    confidence = detections.confidence[i]
                    speed_text = f"{speeds.get(track_id, 0.0):.1f} km/h" if track_id in speeds else "N/A"
                    
                    label = f"#{track_id} {class_name} {confidence:.2f} | {speed_text}"
                    labels.append(label)
                else:
                    labels.append("Unknown")
        
        if labels:
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
        # Add analytics overlay
        annotated_frame = self._add_analytics_overlay(annotated_frame, zone_results)
        
        return annotated_frame
    
    def _add_analytics_overlay(self, frame: np.ndarray, zone_results: Dict) -> np.ndarray:
        """Add analytics information overlay"""
        h, w = frame.shape[:2]
        overlay_height = 150
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        
        # Add analytics text
        y_offset = 30
        cv2.putText(overlay, f"Active Tracks: {len(self.metrics)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        total_entries = sum(len(zone['entries']) for zone in [zone_results] if 'entries' in zone)
        cv2.putText(overlay, f"Total Entries: {total_entries}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        for i, occupancy in enumerate(zone_results['current_occupancy']):
            cv2.putText(overlay, f"Zone {i} Count: {occupancy['count']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 20
        
        # Combine overlay with frame
        result = np.vstack([frame, overlay])
        return result
    
    def export_analytics(self, output_path: Path) -> None:
        """Export analytics to JSON and CSV files"""
        # Export summary metrics
        summary_data = {
            'total_tracks': len(self.metrics),
            'total_frames': len(self.frame_analytics),
            'track_metrics': {}
        }
        
        for track_id, metric in self.metrics.items():
            summary_data['track_metrics'][str(track_id)] = {
                'track_id': int(metric.track_id),
                'class_id': int(metric.class_id),
                'lifetime': float(metric.lifetime),
                'total_detections': int(metric.total_detections),
                'average_speed': float(metric.average_speed),
                'zone_entries': {int(k): float(v) for k, v in metric.zone_entries.items()},
                'zone_exits': {int(k): float(v) for k, v in metric.zone_exits.items()},
                'max_speed': float(max(metric.speeds)) if metric.speeds else 0.0,
                'min_speed': float(min(metric.speeds)) if metric.speeds else 0.0,
                'total_zone_entries': int(len(metric.zone_entries)),
                'total_zone_exits': int(len(metric.zone_exits))
            }
        
        # Save JSON summary
        json_path = output_path / "analytics_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save frame analytics to CSV
        csv_path = output_path / "frame_analytics.csv"
        import csv
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'detection_count', 'active_tracks']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in self.frame_analytics:
                writer.writerow({
                    'timestamp': frame_data['timestamp'],
                    'detection_count': frame_data['detection_count'],
                    'active_tracks': frame_data['active_tracks']
                })
        
        print(f"Analytics exported to {json_path} and {csv_path}")


def create_yolo_model(model_path: str = "yolov8n.pt"):
    """Create a YOLO detection model"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        class YOLOModel:
            def __init__(self, yolo_model):
                self.model = yolo_model
                self.class_names = self.model.names
                
            def __call__(self, frame):
                results = self.model(frame, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
                return detections
        
        return YOLOModel(model)
    except ImportError:
        print("Warning: ultralytics not available. Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


def create_roboflow_model(model_id: str, api_key: str = None):
    """Create a Roboflow Inference model"""
    try:
        from inference.models.utils import get_roboflow_model
        import os
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            
        if api_key is None:
            print("Warning: Roboflow API key not provided. Set ROBOFLOW_API_KEY environment variable.")
            return None
            
        model = get_roboflow_model(model_id=model_id, api_key=api_key)
        
        class RoboflowModel:
            def __init__(self, rf_model):
                self.model = rf_model
                self.class_names = []  # Will be populated from first inference
                
            def __call__(self, frame):
                results = self.model.infer(frame)
                detections = sv.Detections.from_inference(results[0])
                return detections
        
        return RoboflowModel(model)
    except ImportError:
        print("Warning: The 'inference' package is required only for using Roboflow cloud models. Install it with: pip install inference")
        return None
    except Exception as e:
        print(f"Error loading Roboflow model: {e}")
        return None


def create_demo_model():
    """Create a mock detection model for demo purposes"""
    class MockModel:
        def __init__(self):
            self.class_names = ["vehicle", "person", "bicycle"]
            
        def __call__(self, frame):
            # Generate mock detections for demo
            h, w = frame.shape[:2]
            num_detections = np.random.randint(2, 8)
            
            xyxy = []
            confidences = []
            class_ids = []
            
            for _ in range(num_detections):
                x1 = np.random.randint(0, w - 100)
                y1 = np.random.randint(0, h - 100)  
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(50, 100)
                
                xyxy.append([x1, y1, x2, y2])
                confidences.append(np.random.uniform(0.3, 0.9))
                class_ids.append(np.random.randint(0, len(self.class_names)))
            
            return sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )
    
    return MockModel()


def create_model(model_type: str = "mock", model_path: str = None, model_id: str = None, api_key: str = None):
    """Create detection model based on type
    
    Args:
        model_type: Type of model ("yolo", "roboflow", "mock")
        model_path: Path to model file (for YOLO)
        model_id: Model ID (for Roboflow)
        api_key: API key (for Roboflow)
    
    Returns:
        Model instance or None if creation failed
    """
    if model_type.lower() == "yolo":
        model_path = model_path or "yolov8n.pt"  # Default to nano model
        print(f"Loading YOLO model: {model_path}")
        model = create_yolo_model(model_path)
        if model:
            print("✓ YOLO model loaded successfully")
            return model
        else:
            print("✗ Failed to load YOLO model, falling back to mock model")
            
    elif model_type.lower() == "roboflow":
        if not model_id:
            raise ValueError("Model ID is required for Roboflow models. Please provide a valid model_id.")
        print(f"Loading Roboflow model: {model_id}")
        model = create_roboflow_model(model_id, api_key)
        if model:
            print("✓ Roboflow model loaded successfully")
            return model
        else:
            print("✗ Failed to load Roboflow model, falling back to mock model")
    
    # Default to mock model
    print("Using mock model for demonstration")
    return create_demo_model()


def process_video_demo(source_path: str, output_path: str, config: AnalyticsConfig, 
                      model_type: str = "mock", model_path: str = None, 
                      model_id: str = None, api_key: str = None):
    """Process video with full analytics pipeline"""
    
    # Initialize model
    model = create_model(model_type, model_path, model_id, api_key)
    
    # Initialize analytics system
    analytics = SmartTrafficAnalytics(config, model)
    
    # Setup video processing
    video_info = sv.VideoInfo.from_video_path(source_path)
    fps_monitor = sv.FPSMonitor()
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    with sv.VideoSink(
        target_path=str(output_path / "annotated_video.mp4"),
        video_info=video_info
    ) as sink:
        
        frame_generator = sv.get_video_frames_generator(source_path)
        
        for frame_idx, frame in enumerate(tqdm(frame_generator, desc="Processing video")):
            fps_monitor.tick()
            timestamp = frame_idx / video_info.fps
            
            # Run detection
            detections = model(frame)
            
            # Process with analytics
            annotated_frame, frame_analytics = analytics.process_frame(
                frame, detections, timestamp
            )
            
            # Add FPS counter
            fps_text = f"FPS: {fps_monitor.fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            sink.write_frame(annotated_frame)
    
    # Export analytics
    analytics.export_analytics(output_path)
    
    print(f"Video processing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total tracks processed: {len(analytics.metrics)}")
    print(f"Average FPS: {fps_monitor.fps:.2f}")


def create_demo_video(output_path: str, duration: int = 30, fps: int = 30):
    """Create a demo video with moving objects for testing"""
    width, height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    # Create moving objects
    objects = []
    for i in range(5):
        objects.append({
            'start_x': np.random.randint(0, width//4),
            'start_y': np.random.randint(height//4, 3*height//4),
            'speed_x': np.random.uniform(2, 8),
            'speed_y': np.random.uniform(-2, 2),
            'color': tuple(map(int, np.random.randint(0, 255, 3))),
            'size': np.random.randint(20, 50)
        })
    
    for frame_idx in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Draw moving objects
        for obj in objects:
            x = int(obj['start_x'] + obj['speed_x'] * frame_idx)
            y = int(obj['start_y'] + obj['speed_y'] * frame_idx)
            
            # Wrap around screen
            x = x % width
            y = abs(y) % height
            
            cv2.circle(frame, (x, y), obj['size'], obj['color'], -1)
        
        # Add some reference lines
        cv2.line(frame, (0, height//3), (width, height//3), (100, 100, 100), 2)
        cv2.line(frame, (0, 2*height//3), (width, 2*height//3), (100, 100, 100), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Demo video created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Supervision Demo")
    parser.add_argument("--source", type=str, help="Source video path")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--create-demo-video", action="store_true", help="Create demo video")
    parser.add_argument("--demo-duration", type=int, default=30, help="Demo video duration in seconds")
    
    # Model selection arguments
    parser.add_argument("--model-type", type=str, choices=["mock", "yolo", "roboflow"], 
                       default="yolo", help="Type of detection model to use")
    parser.add_argument("--model-path", type=str, help="Path to YOLO model file (default: yolov8n.pt)")
    parser.add_argument("--model-id", type=str, help="Roboflow model ID (default: vehicle-count-in-drone-video/6)")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = AnalyticsConfig()
    
    if args.create_demo_video:
        demo_video_path = output_path / "demo_input.mp4"
        create_demo_video(str(demo_video_path), args.demo_duration)
        args.source = str(demo_video_path)
    
    if args.source:
        if not Path(args.source).exists():
            print(f"Error: Source video {args.source} not found")
            return
            
        print("Starting comprehensive traffic analytics demo...")
        print(f"Source: {args.source}")
        print(f"Output: {output_path}")
        print(f"Model type: {args.model_type}")
        
        process_video_demo(
            source_path=args.source,
            output_path=str(output_path), 
            config=config,
            model_type=args.model_type,
            model_path=args.model_path,
            model_id=args.model_id,
            api_key=args.api_key
        )
    else:
        print("Please provide a source video with --source or use --create-demo-video")
        print("\nExamples:")
        print("  # Use YOLO model (default nano model)")
        print("  python demo_comprehensive.py --create-demo-video --model-type yolo")
        print("  # Use specific YOLO model")
        print("  python demo_comprehensive.py --source video.mp4 --model-type yolo --model-path yolov8s.pt") 
        print("  # Use Roboflow model")
        print("  python demo_comprehensive.py --source video.mp4 --model-type roboflow --model-id your-model/1")
        print("  # Use mock model for testing")
        print("  python demo_comprehensive.py --create-demo-video --model-type mock")


if __name__ == "__main__":
    main()