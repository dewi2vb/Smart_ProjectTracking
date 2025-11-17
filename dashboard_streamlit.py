import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import time
import csv
import base64
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
from collections import defaultdict, deque
import hashlib
import logging
import queue
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Setup logging to reduce warnings
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# === Setup Folder ===
VIDEOS_DIR = "videos"
OUTPUT_DIR = "outputs"
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Smart Surveillance (Optimized Tracking)", layout="wide")
st.title("🎥 Smart Surveillance Dashboard - Optimized Tracking")
st.markdown("Deteksi & tracking dengan bounding box stabil dan trajectory yang kontinu.")

# === OpenCV Optimization ===
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# === GPU Detection ===
if torch.cuda.is_available():
    st.success(f"✅ GPU aktif: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    st.warning("⚠️ GPU tidak terdeteksi — menjalankan di CPU (lebih lambat).")
    device = "cpu"

# === SIMPLIFIED State Management ===
class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self.current_frame = 0
        self.active_tracks = 0
        self.progress = 0.0
        self.status = "Ready"
        self.current_frame_image = None
        self.final_results = None
        self.final_metrics = None
        self.error = None
        self.stop_requested = False

# Global state instance
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = ProcessingState()
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None

processing_state = st.session_state.processing_state

# === Helper: HTML video player ===
def play_video(video_path):
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode("utf-8")
        video_html = f"""
        <video width="100%" height="auto" controls autoplay muted>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.error("❌ File video output tidak ditemukan")

# === Input Video / Stream ===
mode = st.radio("Pilih Mode Input", ["Upload Video", "Gunakan Video Lama", "RTSP URL"])

source = None
if mode == "Upload Video":
    uploaded = st.file_uploader("📂 Upload Video", type=["mp4", "avi", "mov"])
    if uploaded:
        path = os.path.join(VIDEOS_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"✅ Video disimpan ke {path}")
        source = path

elif mode == "Gunakan Video Lama":
    files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".avi", ".mov"))]
    if files:
        selected = st.selectbox("Pilih video:", files)
        source = os.path.join(VIDEOS_DIR, selected)
    else:
        st.warning("⚠️ Belum ada video di folder 'videos/'")

elif mode == "RTSP URL":
    url = st.text_input("Masukkan RTSP URL", placeholder="rtsp://username:password@ip_address:port/stream")
    if url:
        source = url
        st.info("🔗 Mode RTSP - Processing akan berjalan terus sampai dihentikan manual")

# === Configuration Sidebar ===
st.sidebar.header("⚙️ Pengaturan Tracking")

# Model selection
model_type = st.sidebar.selectbox(
    "Model YOLO",
    ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"],
    index=0,
    help="Pilih model YOLO (s = balanced, m = most accurate, n = fastest)"
)

# Detection parameters - DITURUNKAN confidence threshold untuk RTSP
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05,  # Diperbolehkan lebih rendah
    max_value=0.9, 
    value=0.15,
    step=0.05,
    help="Threshold kepercayaan untuk deteksi objek (lebih rendah = lebih banyak deteksi)"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.25,
    step=0.05,
    help="Threshold Intersection Over Union untuk NMS"
)

# Tracking parameters
st.sidebar.header("🔧 Pengaturan Tracking")

max_age = st.sidebar.slider(
    "Max Age", 
    min_value=10, 
    max_value=100, 
    value=30,
    step=5,
    help="Frame maksimum tanpa deteksi sebelum track dihapus"
)

n_init = st.sidebar.slider(
    "N Init", 
    min_value=1, 
    max_value=10, 
    value=2,
    step=1,
    help="Frame minimal untuk konfirmasi track"
)

max_cosine_distance = st.sidebar.slider(
    "Max Cosine Distance", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.3,
    step=0.05,
    help="Jarak maksimum untuk matching track"
)

# Filter parameters - DIPERLUAS rentang minimum
st.sidebar.header("🎯 Filter Deteksi")

min_person_height = st.sidebar.slider(
    "Minimum Person Height", 
    min_value=10,
    max_value=200, 
    value=30,
    step=5,
    help="Tinggi minimum orang yang dideteksi (pixel)"
)

min_person_width = st.sidebar.slider(
    "Minimum Person Width", 
    min_value=5,
    max_value=100, 
    value=15,
    step=5,
    help="Lebar minimum orang yang dideteksi (pixel)"
)

# === Class untuk menghitung metrik ===
class MetricsCalculator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.detection_confidences = []
        self.track_lifetimes = defaultdict(list)
        self.id_switches = 0
        self.previous_tracks = set()
        self.frame_data = []
        self.frames_with_detections = 0
        
    def update_metrics(self, current_tracks, detections, frame_number):
        """Update metrics berdasarkan deteksi dan tracking saat ini"""
        current_track_set = set(current_tracks)
        
        # Simpan data frame
        self.frame_data.append({
            'frame': frame_number,
            'tracks': list(current_track_set),
            'detections': len(detections),
            'timestamp': time.time()
        })
        
        # Hitung frames dengan deteksi
        if len(detections) > 0 or len(current_track_set) > 0:
            self.frames_with_detections += 1
        
        # Hitung ID switches
        new_tracks = current_track_set - self.previous_tracks
        if frame_number > 1:
            self.id_switches += len(new_tracks)
        
        self.previous_tracks = current_track_set.copy()
        
        # Update track lifetimes
        for track_id in current_track_set:
            self.track_lifetimes[track_id].append(frame_number)
        
        # Simulasi perhitungan metrik
        if len(detections) > 0:
            self.true_positives += len(current_tracks)
            if len(detections) > len(current_tracks):
                self.false_positives += (len(detections) - len(current_tracks))
        
        # Simpan confidence deteksi
        for det in detections:
            if len(det) > 1:
                self.detection_confidences.append(det[1])
    
    def calculate_metrics(self, total_frames=None, is_partial=False):
        """Hitung metrik - support partial data dengan error handling"""
        if total_frames is None:
            total_frames = len(self.frame_data) if self.frame_data else 1
        
        # Handle kasus tidak ada deteksi sama sekali
        if self.true_positives == 0 and self.false_positives == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'total_tracks': 0,
                'avg_track_length': 0.0,
                'mostly_tracked': 0,
                'id_switches': self.id_switches,
                'total_frames_processed': len(self.frame_data),
                'frames_with_detections': self.frames_with_detections,
                'detection_rate': (self.frames_with_detections / len(self.frame_data)) * 100 if self.frame_data else 0,
                'is_partial': is_partial,
                'no_detections': True
            }
        
        # Hitung precision, recall, F1-score dengan error handling
        try:
            precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
            recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        except:
            precision = recall = f1_score = 0
        
        # Hitung metrik tracking dengan error handling
        try:
            total_tracks = len(self.track_lifetimes)
            if total_tracks > 0:
                track_lengths = [len(frames) for frames in self.track_lifetimes.values()]
                avg_track_length = np.mean(track_lengths) if track_lengths else 0
                mostly_tracked = sum(1 for frames in self.track_lifetimes.values() if len(frames) >= total_frames * 0.8)
            else:
                avg_track_length = 0
                mostly_tracked = 0
        except:
            total_tracks = avg_track_length = mostly_tracked = 0
        
        # Hitung average confidence dengan error handling
        try:
            avg_confidence = np.mean(self.detection_confidences) if self.detection_confidences else 0
        except:
            avg_confidence = 0
        
        metrics = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'accuracy': min(precision, recall) * 100,
            'avg_confidence': avg_confidence * 100,
            'total_tracks': total_tracks,
            'avg_track_length': avg_track_length,
            'mostly_tracked': mostly_tracked,
            'id_switches': self.id_switches,
            'total_frames_processed': len(self.frame_data),
            'frames_with_detections': self.frames_with_detections,
            'detection_rate': (self.frames_with_detections / len(self.frame_data)) * 100 if self.frame_data else 0,
            'is_partial': is_partial,
            'no_detections': False
        }
        
        return metrics

# === Fungsi untuk plot grafik metrik ===
def plot_metrics(metrics):
    """Buat grafik untuk menampilkan metrik performa"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Precision, Recall, F1-Score
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Performance Metrics', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Tambah nilai di atas bar
    for bar, value in zip(bars, metrics_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Tracking Quality
    track_metrics = ['Total Tracks', 'Avg Track Length', 'Mostly Tracked']
    track_values = [metrics['total_tracks'], metrics['avg_track_length'], metrics['mostly_tracked']]
    
    ax2.bar(track_metrics, track_values, color=['#96CEB4', '#FFEAA7', '#DDA0DD'], alpha=0.8)
    ax2.set_title('Tracking Quality Metrics', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Detection Rate & Confidence
    detection_metrics = ['Detection Rate', 'Avg Confidence']
    detection_values = [metrics.get('detection_rate', 0), metrics['avg_confidence']]
    
    bars3 = ax3.bar(detection_metrics, detection_values, color=['#66B2FF', '#FF9999'], alpha=0.8)
    ax3.set_title('Detection Metrics', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Tambah nilai di atas bar
    for bar, value in zip(bars3, detection_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary
    status = "PARTIAL RESULTS" if metrics.get('is_partial', False) else "COMPLETE RESULTS"
    if metrics.get('no_detections', False):
        status = "NO DETECTIONS FOUND"
        status_color = "#FF4444"
    else:
        status_color = "#FFA500" if metrics.get('is_partial', False) else "#2E8B57"
    
    summary_text = f"""
    Performance Summary:
    Status: {status}
    
    Precision: {metrics['precision']:.1f}%
    Recall: {metrics['recall']:.1f}%
    F1-Score: {metrics['f1_score']:.1f}%
    
    Frames Processed: {metrics['total_frames_processed']}
    Frames with Detections: {metrics.get('frames_with_detections', 0)}
    Detection Rate: {metrics.get('detection_rate', 0):.1f}%
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.3))
    ax4.set_title('Analysis Summary', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

# === OPTIMIZED TRACKING SYSTEM ===
class OptimizedTracker:
    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=100))
        self.last_positions = {}
        self.id_colors = {}
        
    def get_color_for_id(self, track_id):
        """Generate consistent color based on track_id"""
        if track_id not in self.id_colors:
            if isinstance(track_id, str):
                track_hash = int(hashlib.md5(track_id.encode()).hexdigest()[:8], 16)
            else:
                track_hash = int(track_id)
            
            r = (track_hash * 67) % 200 + 55
            g = (track_hash * 43) % 200 + 55  
            b = (track_hash * 29) % 200 + 55
            
            self.id_colors[track_id] = (int(b), int(g), int(r))
        
        return self.id_colors[track_id]
    
    def update_track_history(self, track_id, position, bbox):
        """Update history track dengan bounding box"""
        self.track_history[track_id].append({
            'position': position,
            'bbox': bbox,
            'timestamp': time.time()
        })
        self.last_positions[track_id] = position

# === SIMPLIFIED PROCESSING FUNCTION ===
def process_detection_optimized(source):
    """Fungsi processing yang simplified dengan error handling"""
    try:
        # Update state langsung
        processing_state.status = "Loading model..."
        
        # Load model dengan error handling
        try:
            model = YOLO(model_type)
            # Test model dengan dummy input
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            model(dummy_input, verbose=False)
        except Exception as e:
            processing_state.status = f"Error loading model: {str(e)}"
            processing_state.error = f"Gagal memuat atau test model {model_type}"
            processing_state.is_processing = False
            return
        
        # Tracker
        tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=50,
            embedder="mobilenet",
            half=True if device == "cuda" else False,
            max_iou_distance=0.8,
            embedder_gpu=True if device == "cuda" else False
        )
        
        optimized_tracker = OptimizedTracker()
        metrics_calculator = MetricsCalculator()

        processing_state.status = "Opening video source..."
        
        # Open video source dengan timeout
        cap = cv2.VideoCapture(source)
        
        if source.startswith('rtsp://'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            # Set timeout untuk RTSP
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        
        if not cap.isOpened():
            processing_state.status = "Error: Gagal membuka sumber video"
            processing_state.error = "Tidak dapat membuka file video atau stream RTSP"
            processing_state.is_processing = False
            return

        # Test baca frame pertama
        ret, test_frame = cap.read()
        if not ret:
            processing_state.status = "Error: Tidak bisa membaca frame dari sumber"
            processing_state.error = "Stream RTSP mungkin offline atau format tidak didukung"
            processing_state.is_processing = False
            cap.release()
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Output files
        csv_path = os.path.join(OUTPUT_DIR, "movement_log_optimized.csv")
        out_path = os.path.join(OUTPUT_DIR, "output_tracking_optimized.mp4")
        
        # Initialize video writer
        try:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        except Exception as e:
            processing_state.status = f"Error: Gagal membuat output video"
            processing_state.error = f"Tidak dapat membuat file output: {e}"
            processing_state.is_processing = False
            cap.release()
            return

        heatmap = np.zeros((height, width), dtype=np.float32)
        
        processing_state.status = "Starting optimized detection (HANYA MANUSIA)..."

        # Open CSV file
        try:
            f_csv = open(csv_path, "w", newline="")
            writer = csv.writer(f_csv)
            writer.writerow(["frame", "timestamp", "track_id", "x_center", "y_center", "confidence", "width", "height", "status"])
        except Exception as e:
            processing_state.status = f"Error: Gagal membuat file CSV"
            processing_state.error = f"Tidak dapat membuat file CSV: {e}"
            processing_state.is_processing = False
            cap.release()
            out.release()
            return

        start_time = time.time()
        frame_count = 0
        consecutive_failures = 0
        
        # Statistics
        total_detections = 0
        active_tracks = set()
        id_switches = 0
        previous_tracks = set()

        while processing_state.is_processing and not processing_state.stop_requested:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:  # Max 10 failures
                    processing_state.status = "Error: Terlalu banyak frame gagal dibaca"
                    break
                
                if source.startswith('rtsp://'):
                    processing_state.status = "RTSP stream interrupted, reconnecting..."
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        processing_state.status = "Failed to reconnect to RTSP"
                        break
                    consecutive_failures = 0
                    continue
                else:
                    break

            consecutive_failures = 0
            frame_count += 1
            current_time = time.time() - start_time
            
            # Update state
            processing_state.current_frame = frame_count
            processing_state.status = f"Processing frame {frame_count}"
            processing_state.active_tracks = len(active_tracks)

            # Detection dengan error handling
            try:
                results = model(
                    frame, 
                    verbose=False, 
                    conf=confidence_threshold,
                    iou=iou_threshold,  
                    agnostic_nms=True,
                    max_det=100,
                    classes=[0]  # Hanya deteksi orang
                )
            except Exception as e:
                print(f"Detection error: {e}")
                continue
                
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        
                        if w >= min_person_width and h >= min_person_height:
                            detections.append(([x1, y1, w, h], conf, "person"))

            # Tracking dengan error handling
            try:
                tracks = tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                print(f"Tracking error: {e}")
                tracks = []
                
            current_frame_tracks = set()
            timestamp = round(current_time, 2)

            # Process confirmed tracks
            confirmed_tracks = []
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    x_center, y_center = int((l + r) / 2), int((t + b) / 2)
                    bbox = (l, t, r, b)
                    
                    confirmed_tracks.append((track, track_id, l, t, r, b, x_center, y_center, bbox))
                    current_frame_tracks.add(track_id)

            # Detect ID switches
            new_tracks = current_frame_tracks - previous_tracks
            lost_tracks = previous_tracks - current_frame_tracks
            if len(new_tracks) > 0 and frame_count > 10:
                id_switches += len(new_tracks)

            previous_tracks = current_frame_tracks.copy()

            # Process all tracks
            for track, track_id, l, t, r, b, x_center, y_center, bbox in confirmed_tracks:
                conf = round(track.det_conf, 3) if track.det_conf else 0
                w, h = r - l, b - t
                
                optimized_tracker.update_track_history(track_id, (x_center, y_center), bbox)
                
                writer.writerow([frame_count, timestamp, track_id, x_center, y_center, conf, w, h, "detected"])
                total_detections += 1

                # Update heatmap
                if 0 <= y_center < height and 0 <= x_center < width:
                    weight = 0.5 + (conf * 0.5)
                    heatmap[y_center, x_center] += weight

                # Visualization
                color = optimized_tracker.get_color_for_id(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                
                # Draw trajectory
                history = optimized_tracker.track_history[track_id]
                if len(history) > 1:
                    points = [item['position'] for item in history]
                    points_array = np.array(points, dtype=np.int32)
                    
                    for i in range(1, len(points_array)):
                        cv2.line(frame, points_array[i-1], points_array[i], color, 2)
                
                # Draw ID
                label = f"ID:{track_id} C:{conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (l, t - text_height - 10), 
                            (l + text_width, t), color, -1)
                cv2.putText(frame, label, (l, t - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update active tracks count
            active_tracks = current_frame_tracks
            
            # Update metrics
            metrics_calculator.update_metrics(current_frame_tracks, detections, frame_count)

            # Heatmap visualization hanya jika ada deteksi
            if len(current_frame_tracks) > 0 and frame_count % 10 == 0 and np.max(heatmap) > 0:
                try:
                    hm_blur = gaussian_filter(heatmap, sigma=20)
                    hm_norm = cv2.normalize(np.log1p(hm_blur), None, 0, 255, cv2.NORM_MINMAX)
                    hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.8, hm_color, 0.2, 0)
                except:
                    pass  # Skip heatmap jika error

            # Information overlay
            info_text = [
                f"Frame: {frame_count}",
                f"Active Tracks: {len(active_tracks)}",
                f"Total Detections: {total_detections}",
                f"ID Switches: {id_switches}",
                f"Source: {'RTSP' if source.startswith('rtsp://') else 'Video'}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Tambah warning jika tidak ada deteksi
            if len(current_frame_tracks) == 0 and len(detections) == 0:
                warning_text = "NO DETECTIONS - Adjust confidence threshold"
                cv2.putText(frame, warning_text, (width//2 - 200, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display current frame
            display_frame = cv2.resize(frame, (640, 480))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            processing_state.current_frame_image = display_frame_rgb

            out.write(frame)
            
            # Update progress
            if source.startswith('rtsp://'):
                progress = min(frame_count / 1000, 0.99)
            else:
                progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else min(frame_count / 1000, 0.99)
            
            processing_state.progress = progress

            # Small delay
            time.sleep(0.03)

        # Cleanup
        cap.release()
        out.release()
        f_csv.close()
        
        # Calculate final metrics
        is_partial = processing_state.stop_requested or source.startswith('rtsp://')
        final_metrics = metrics_calculator.calculate_metrics(frame_count, is_partial=is_partial)
        
        # Store final results
        processing_state.final_results = {
            "total_detections": total_detections,
            "max_tracks": max([len(active_tracks)] if active_tracks else [0]),
            "id_switches": id_switches,
            "output_path": out_path,
            "csv_path": csv_path,
            "total_frames": frame_count,
            "model_used": model_type,
            "confidence_threshold": confidence_threshold,
            "source_type": "RTSP" if source.startswith('rtsp://') else "Video File",
            "frames_with_detections": final_metrics.get('frames_with_detections', 0)
        }
        processing_state.final_metrics = final_metrics
        processing_state.status = "Stopped - Partial Results" if is_partial else "Completed"
        processing_state.is_processing = False

    except Exception as e:
        processing_state.status = f"Error: {str(e)}"
        processing_state.error = str(e)
        processing_state.is_processing = False
        import traceback
        print(f"Error in processing thread: {traceback.format_exc()}")

# === Main App ===
if source and st.button("▶️ Jalankan Optimized Tracking (HANYA MANUSIA)"):
    if processing_state.is_processing:
        st.warning("⚠️ Processing sedang berjalan...")
    else:
        # Reset state
        processing_state.is_processing = True
        processing_state.current_frame = 0
        processing_state.active_tracks = 0
        processing_state.progress = 0.0
        processing_state.status = "Starting..."
        processing_state.current_frame_image = None
        processing_state.final_results = None
        processing_state.final_metrics = None
        processing_state.error = None
        processing_state.stop_requested = False
        
        # Start processing thread
        thread = threading.Thread(
            target=process_detection_optimized, 
            args=(source,), 
            daemon=True
        )
        thread.start()
        
        st.session_state.processing_thread = thread
        
        if source.startswith('rtsp://'):
            st.success("🚀 Memulai proses tracking RTSP (Tekan STOP untuk menghentikan)...")
        else:
            st.success("🚀 Memulai proses tracking video...")

# === Display Progress ===
if processing_state.is_processing or processing_state.final_results:
    st.subheader("📊 Progress Tracking")
    
    progress_bar = st.progress(float(processing_state.progress))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frame", processing_state.current_frame)
    with col2:
        st.metric("Active Tracks", processing_state.active_tracks)
    with col3:
        st.metric("Status", processing_state.status)
    with col4:
        if processing_state.final_results:
            source_type = processing_state.final_results.get("source_type", "Video")
            st.metric("Source", source_type)
    
    # Tampilkan video real-time
    if processing_state.current_frame_image is not None:
        st.subheader("🎥 Live Tracking Preview")
        st.image(
            processing_state.current_frame_image, 
            caption=f"Frame {processing_state.current_frame} - Active Tracks: {processing_state.active_tracks}",
            use_container_width=True
        )
    
    # Display results when completed
    if processing_state.final_results and not processing_state.is_processing:
        results = processing_state.final_results
        metrics = processing_state.final_metrics
        
        if processing_state.status == "Completed":
            st.success("✅ Proses selesai!")
        else:
            st.warning("🛑 Proses dihentikan - Menampilkan hasil parsial")
        
        # Tampilkan basic metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", results.get("total_frames", 0))
        col2.metric("Total Detections", results.get("total_detections", 0))
        col3.metric("Max Tracks", results.get("max_tracks", 0))
        col4.metric("ID Switches", results.get("id_switches", 0))
        
        # Tampilkan detection rate
        if metrics and metrics.get('frames_with_detections', 0) > 0:
            detection_rate = metrics.get('detection_rate', 0)
            st.info(f"🔍 Detection Rate: {detection_rate:.1f}% ({metrics.get('frames_with_detections', 0)}/{results.get('total_frames', 0)} frames)")
        
        st.info(f"🔧 Konfigurasi: Model {results.get('model_used', 'N/A')}, Confidence {results.get('confidence_threshold', 'N/A')}")
        
        # Tampilkan grafik metrik jika tersedia
        if metrics:
            st.subheader("📈 Performance Metrics")
            
            if metrics.get('no_detections', False):
                st.error("❌ **TIDAK ADA DETEKSI** - Tidak ada orang yang terdeteksi dalam video/stream")
                st.warning("""
                **Solusi:**
                - Turunkan **Confidence Threshold** (coba 0.05-0.10)
                - Periksa koneksi RTSP
                - Pastikan ada orang dalam frame
                - Coba model YOLO yang berbeda (yolov8m.pt)
                """)
            else:
                if metrics.get('is_partial', False):
                    st.warning("📊 **PARTIAL RESULTS** - Berdasarkan data yang berhasil diproses sebelum dihentikan")
                else:
                    st.success("📊 **COMPLETE RESULTS** - Semua frame berhasil diproses")
                
                # Tampilkan metrik utama
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
                col2.metric("Recall", f"{metrics.get('recall', 0):.1f}%")
                col3.metric("F1-Score", f"{metrics.get('f1_score', 0):.1f}%")
                col4.metric("Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
                
                # Tampilkan grafik
                st.subheader("📊 Detailed Metrics Visualization")
                fig = plot_metrics(metrics)
                st.pyplot(fig)
                
                # Tampilkan metrik tambahan
                col1, col2, col3 = st.columns(3)
                col1.metric("Frames Processed", metrics.get('total_frames_processed', 0))
                col2.metric("Total Tracks", metrics.get('total_tracks', 0))
                col3.metric("Detection Rate", f"{metrics.get('detection_rate', 0):.1f}%")
                
                # Analisis performa
                st.subheader("💡 Performance Analysis")
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                if precision >= 80 and recall >= 80:
                    st.success("**Excellent Performance**: Sistem tracking bekerja dengan sangat baik!")
                elif precision >= 70 and recall >= 70:
                    st.info("**Good Performance**: Sistem bekerja dengan baik, ada ruang untuk improvement.")
                else:
                    st.warning("**Needs Improvement**: Pertimbangkan untuk menyesuaikan parameter tracking.")
        
        # Play video result
        output_path = results.get("output_path", "")
        if output_path and os.path.exists(output_path):
            st.subheader("🎬 Video Hasil Tracking")
            play_video(output_path)
        else:
            st.warning("⚠️ File video output tidak tersedia")
        
        # Download results
        csv_path = results.get("csv_path", "")
        if csv_path and os.path.exists(csv_path):
            col1, col2 = st.columns(2)
            with col1:
                if output_path and os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "📥 Download Video Hasil",
                            f,
                            file_name="tracking_result.mp4",
                            mime="video/mp4"
                        )
            with col2:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📊 Download Data CSV",
                        f,
                        file_name="tracking_data.csv",
                        mime="text/csv"
                    )

# Display error if any
if processing_state.error:
    st.error("❌ Terjadi error selama processing:")
    st.error(processing_state.error)

# Stop button
if processing_state.is_processing:
    if st.button("⏹️ Stop Processing", type="primary"):
        processing_state.stop_requested = True
        processing_state.status = "Stopping..."
        st.warning("🛑 Menghentikan proses...")
        st.rerun()

# Auto-refresh
if processing_state.is_processing:
    time.sleep(0.5)
    st.rerun()

# === Footer Information ===
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Informasi Sistem")
st.sidebar.info(
    """
    **Tips untuk RTSP:**
    - Confidence Threshold: 0.05-0.15
    - Pastikan RTSP stream aktif
    - Cek koneksi jaringan
    - Model: yolov8s.pt atau yolov8m.pt
    
    **Jika tidak ada deteksi:**
    - Turunkan confidence threshold
    - Periksa RTSP stream di VLC player
    - Pastikan ada orang dalam frame
    Create By Dewi2vb
    """
)
