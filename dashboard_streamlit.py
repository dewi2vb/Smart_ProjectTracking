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

# === Thread-safe State Management ===
class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self.current_frame = 0
        self.active_tracks = 0
        self.progress = 0.0
        self.status = "Ready"
        self.current_frame_image = None
        self.results = None
        self.error = None
        self.metrics = None
        self.partial_data = None  # ✅ Untuk menyimpan data partial saat stop
        self.update_queue = queue.Queue()
    
    def update_from_queue(self):
        """Process updates from the thread"""
        try:
            while not self.update_queue.empty():
                update = self.update_queue.get_nowait()
                for key, value in update.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except Exception as e:
            print(f"Error updating state: {e}")
    
    def queue_update(self, **kwargs):
        """Queue an update to be processed in the main thread"""
        try:
            self.update_queue.put(kwargs)
        except Exception as e:
            print(f"Error queuing update: {e}")

# Global state instance
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = ProcessingState()

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

# Detection parameters
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.15,
    step=0.05,
    help="Threshold kepercayaan untuk deteksi objek"
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

# Filter parameters
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

# === TAMBAHAN: Class untuk menghitung metrik ===
class MetricsCalculator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.detection_confidences = []
        self.track_lifetimes = defaultdict(list)
        self.id_switches = 0
        self.previous_tracks = set()
        self.frame_data = []  # ✅ Simpan data per frame untuk partial calculation
        
    def update_metrics(self, current_tracks, detections, frame_number):
        """Update metrics berdasarkan deteksi dan tracking saat ini"""
        current_track_set = set(current_tracks)
        
        # Simpan data frame untuk partial calculation
        self.frame_data.append({
            'frame': frame_number,
            'tracks': list(current_track_set),
            'detections': len(detections),
            'timestamp': time.time()
        })
        
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
        """Hitung metrik - support partial data"""
        if total_frames is None:
            total_frames = len(self.frame_data) if self.frame_data else 1
        
        # Hitung precision, recall, F1-score
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Hitung metrik tracking
        total_tracks = len(self.track_lifetimes)
        if total_tracks > 0:
            avg_track_length = np.mean([len(frames) for frames in self.track_lifetimes.values()])
            mostly_tracked = sum(1 for frames in self.track_lifetimes.values() if len(frames) >= total_frames * 0.8)
        else:
            avg_track_length = 0
            mostly_tracked = 0
            
        avg_confidence = np.mean(self.detection_confidences) if self.detection_confidences else 0
        
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
            'is_partial': is_partial  # ✅ Flag untuk menandai data partial
        }
        
        return metrics

# === TAMBAHAN: Fungsi untuk plot grafik metrik ===
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
    
    # Plot 3: ID Switches & Confidence
    performance_metrics = ['ID Switches', 'Avg Confidence']
    performance_values = [metrics['id_switches'], metrics['avg_confidence']]
    
    bars3 = ax3.bar(performance_metrics, performance_values, color=['#FF9999', '#66B2FF'], alpha=0.8)
    ax3.set_title('Performance Indicators', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Tambah nilai di atas bar untuk confidence
    ax3.text(1, performance_values[1] + 1, f'{performance_values[1]:.1f}%', 
             ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary dengan info partial
    status = "PARTIAL RESULTS" if metrics.get('is_partial', False) else "COMPLETE RESULTS"
    status_color = "#FFA500" if metrics.get('is_partial', False) else "#2E8B57"
    
    summary_text = f"""
    Performance Summary:
    Status: {status}
    
    Precision: {metrics['precision']:.1f}%
    Recall: {metrics['recall']:.1f}%
    F1-Score: {metrics['f1_score']:.1f}%
    Accuracy: {metrics['accuracy']:.1f}%
    
    Frames Processed: {metrics['total_frames_processed']}
    ID Switches: {metrics['id_switches']}
    Avg Confidence: {metrics['avg_confidence']:.1f}%
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
        self.missing_frames = {}
        
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
        self.missing_frames[track_id] = 0
    
    def get_predicted_position(self, track_id):
        """Prediksi posisi berdasarkan history"""
        history = self.track_history[track_id]
        if len(history) < 2:
            return None
            
        pos1 = history[-2]['position']
        pos2 = history[-1]['position']
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        predicted_x = pos2[0] + dx
        predicted_y = pos2[1] + dy
        
        return (int(predicted_x), int(predicted_y))
    
    def handle_missing_tracks(self, current_tracks):
        """Handle tracks yang tidak terdeteksi di frame ini"""
        missing_tracks = set(self.last_positions.keys()) - set(current_tracks)
        
        for track_id in missing_tracks:
            if track_id in self.missing_frames:
                self.missing_frames[track_id] += 1
            else:
                self.missing_frames[track_id] = 1

def process_detection_optimized(source, state_queue):
    """Fungsi processing dengan tracking yang dioptimalkan - HANYA DETEKSI MANUSIA"""
    try:
        # Kirim status awal via queue
        state_queue.put({
            "is_processing": True,
            "status": "Loading model..."
        })
        
        # Load model dengan error handling
        try:
            model = YOLO(model_type)
        except Exception as e:
            state_queue.put({
                "status": f"Error loading model: {str(e)}",
                "error": f"Gagal memuat model {model_type}",
                "is_processing": False
            })
            return
        
        # Tracker dengan parameter yang DIOPTIMALKAN
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
        
        # Initialize optimized tracker
        optimized_tracker = OptimizedTracker()
        
        # ✅ TAMBAHAN: Initialize metrics calculator
        metrics_calculator = MetricsCalculator()

        state_queue.put({"status": "Opening video source..."})
        
        # Open video source - handle RTSP khusus
        cap = cv2.VideoCapture(source)
        
        # ✅ TAMBAHAN: Set timeout untuk RTSP
        if source.startswith('rtsp://'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not cap.isOpened():
            state_queue.put({
                "status": "Error: Gagal membuka sumber video",
                "error": "Tidak dapat membuka file video atau stream",
                "is_processing": False
            })
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
            state_queue.put({
                "status": f"Error: Gagal membuat output video - {str(e)}",
                "error": f"Tidak dapat membuat file output: {e}",
                "is_processing": False
            })
            cap.release()
            return

        heatmap = np.zeros((height, width), dtype=np.float32)
        
        state_queue.put({"status": "Starting optimized detection (HANYA MANUSIA)..."})

        # Open CSV file
        try:
            f_csv = open(csv_path, "w", newline="")
            writer = csv.writer(f_csv)
            writer.writerow(["frame", "timestamp", "track_id", "x_center", "y_center", "confidence", "width", "height", "status"])
        except Exception as e:
            state_queue.put({
                "status": f"Error: Gagal membuat file CSV - {str(e)}",
                "error": f"Tidak dapat membuat file CSV: {e}",
                "is_processing": False
            })
            cap.release()
            out.release()
            return

        start_time = time.time()
        frame_count = 0
        
        # Statistics
        total_detections = 0
        active_tracks = set()
        id_switches = 0
        previous_tracks = set()

        while True:
            # Check if we should stop processing - ✅ DIPERBAIKI: priority check
            try:
                if not state_queue.empty():
                    control_msg = state_queue.get_nowait()
                    if control_msg.get("stop", False):
                        state_queue.put({"status": "Stopping process..."})
                        break
            except:
                pass

            ret, frame = cap.read()
            if not ret:
                # ✅ TAMBAHAN: Handle RTSP connection issues
                if source.startswith('rtsp://'):
                    state_queue.put({"status": "RTSP stream interrupted, reconnecting..."})
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        state_queue.put({"status": "Failed to reconnect to RTSP"})
                        break
                    continue
                else:
                    break

            frame_count += 1
            current_time = time.time() - start_time
            
            # Update state via queue
            state_queue.put({
                "current_frame": frame_count,
                "status": f"Processing frame {frame_count}",
                "active_tracks": len(active_tracks)
            })

            # === OPTIMIZED DETECTION - HANYA MANUSIA (CLASS 0) ===
            try:
                results = model(
                    frame, 
                    verbose=False, 
                    conf=confidence_threshold,
                    iou=iou_threshold,  
                    agnostic_nms=True,
                    max_det=100,
                    classes=[0]
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

            # === OPTIMIZED TRACKING ===
            try:
                tracks = tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                print(f"Tracking error: {e}")
                tracks = []
                
            current_frame_tracks = set()
            timestamp = round(current_time, 2)

            # Process ALL confirmed tracks
            confirmed_tracks = []
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    x_center, y_center = int((l + r) / 2), int((t + b) / 2)
                    bbox = (l, t, r, b)
                    
                    confirmed_tracks.append((track, track_id, l, t, r, b, x_center, y_center, bbox))
                    current_frame_tracks.add(track_id)

            # === DETECT ID SWITCHES ===
            new_tracks = current_frame_tracks - previous_tracks
            lost_tracks = previous_tracks - current_frame_tracks
            if len(new_tracks) > 0 and frame_count > 10:
                id_switches += len(new_tracks)

            previous_tracks = current_frame_tracks.copy()

            # === PROCESS ALL TRACKS ===
            for track, track_id, l, t, r, b, x_center, y_center, bbox in confirmed_tracks:
                conf = round(track.det_conf, 3) if track.det_conf else 0
                w, h = r - l, b - t
                
                # Update tracking history
                optimized_tracker.update_track_history(track_id, (x_center, y_center), bbox)
                
                # Log data
                writer.writerow([frame_count, timestamp, track_id, x_center, y_center, conf, w, h, "detected"])
                total_detections += 1

                # Update heatmap
                if 0 <= y_center < height and 0 <= x_center < width:
                    weight = 0.5 + (conf * 0.5)
                    heatmap[y_center, x_center] += weight

                # === VISUALIZATION ===
                color = optimized_tracker.get_color_for_id(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                
                # Draw trajectory
                history = optimized_tracker.track_history[track_id]
                if len(history) > 1:
                    points = [item['position'] for item in history]
                    points_array = np.array(points, dtype=np.int32)
                    
                    # Draw smooth trajectory
                    for i in range(1, len(points_array)):
                        cv2.line(frame, points_array[i-1], points_array[i], color, 2)
                
                # Draw ID dengan background
                label = f"ID:{track_id} C:{conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (l, t - text_height - 10), 
                            (l + text_width, t), color, -1)
                cv2.putText(frame, label, (l, t - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Handle missing tracks
            optimized_tracker.handle_missing_tracks(current_frame_tracks)

            # Update active tracks count
            active_tracks = current_frame_tracks
            
            # ✅ TAMBAHAN: Update metrics setiap frame
            metrics_calculator.update_metrics(current_frame_tracks, detections, frame_count)

            # === HEATMAP VISUALIZATION ===
            if frame_count % 10 == 0 and np.max(heatmap) > 0:
                hm_blur = gaussian_filter(heatmap, sigma=20)
                hm_norm = cv2.normalize(np.log1p(hm_blur), None, 0, 255, cv2.NORM_MINMAX)
                hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.8, hm_color, 0.2, 0)

            # === INFORMATION OVERLAY ===
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

            # === DISPLAY CURRENT FRAME ===
            display_frame = cv2.resize(frame, (640, 480))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            state_queue.put({"current_frame_image": display_frame_rgb})

            out.write(frame)
            
            # Update progress untuk RTSP (infinite progress)
            if source.startswith('rtsp://'):
                progress = min(frame_count / 1000, 0.99)  # Progress simulasi untuk RTSP
            else:
                progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else min(frame_count / 1000, 0.99)
            
            state_queue.put({"progress": progress})

            # Delay untuk real-time feeling
            time.sleep(0.03)

        # Cleanup
        cap.release()
        out.release()
        f_csv.close()
        
        # ✅ TAMBAHAN: Hitung metrik akhir - TANDAI sebagai PARTIAL jika di-stop
        is_partial = True  # Selalu partial untuk RTSP atau stopped process
        final_metrics = metrics_calculator.calculate_metrics(frame_count, is_partial=is_partial)
        
        # Kirim hasil akhir
        state_queue.put({
            "progress": 1.0,
            "status": "Completed" if not is_partial else "Stopped - Partial Results",
            "results": {
                "total_detections": total_detections,
                "max_tracks": max([len(active_tracks)] if active_tracks else [0]),
                "id_switches": id_switches,
                "output_path": out_path,
                "csv_path": csv_path,
                "total_frames": frame_count,
                "model_used": model_type,
                "confidence_threshold": confidence_threshold,
                "source_type": "RTSP" if source.startswith('rtsp://') else "Video File"
            },
            "metrics": final_metrics,
            "is_processing": False
        })

    except Exception as e:
        state_queue.put({
            "status": f"Error: {str(e)}",
            "error": str(e),
            "is_processing": False
        })
        import traceback
        print(f"Error in processing thread: {traceback.format_exc()}")

# === Main App ===
processing_state.update_from_queue()

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
        processing_state.results = None
        processing_state.metrics = None
        processing_state.error = None
        
        # Buat queue untuk komunikasi dengan thread
        state_queue = queue.Queue()
        
        # Start processing thread
        thread = threading.Thread(
            target=process_detection_optimized, 
            args=(source, state_queue), 
            daemon=True
        )
        thread.start()
        
        # Simpan queue di session state
        st.session_state.processing_queue = state_queue
        
        if source.startswith('rtsp://'):
            st.success("🚀 Memulai proses tracking RTSP (Tekan STOP untuk menghentikan)...")
        else:
            st.success("🚀 Memulai proses tracking video...")
        st.rerun()

# === Process updates from thread ===
if 'processing_queue' in st.session_state:
    state_queue = st.session_state.processing_queue
    try:
        while True:
            update = state_queue.get_nowait()
            for key, value in update.items():
                if hasattr(processing_state, key):
                    setattr(processing_state, key, value)
    except queue.Empty:
        pass

# === Display Progress ===
if processing_state.is_processing or processing_state.results:
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
        if processing_state.results:
            source_type = processing_state.results.get("source_type", "Video")
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
    if processing_state.results and not processing_state.is_processing:
        results = processing_state.results
        st.success("✅ Proses selesai!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", results["total_frames"])
        col2.metric("Total Detections", results["total_detections"])
        col3.metric("Max Tracks", results["max_tracks"])
        col4.metric("ID Switches", results["id_switches"])
        
        st.info(f"🔧 Konfigurasi: Model {results['model_used']}, Confidence {results['confidence_threshold']}")
        
        # ✅ TAMBAHAN: Tampilkan grafik metrik jika tersedia
        if processing_state.metrics:
            st.subheader("📈 Performance Metrics")
            
            # Tampilkan status partial/complete
            metrics = processing_state.metrics
            if metrics.get('is_partial', False):
                st.warning("📊 **PARTIAL RESULTS** - Berdasarkan data yang berhasil diproses sebelum dihentikan")
            else:
                st.success("📊 **COMPLETE RESULTS** - Semua frame berhasil diproses")
            
            # Tampilkan metrik utama
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", f"{metrics['precision']:.1f}%")
            col2.metric("Recall", f"{metrics['recall']:.1f}%")
            col3.metric("F1-Score", f"{metrics['f1_score']:.1f}%")
            col4.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
            
            # Tampilkan grafik
            st.subheader("📊 Detailed Metrics Visualization")
            fig = plot_metrics(metrics)
            st.pyplot(fig)
            
            # Tampilkan metrik tambahan
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Processed", metrics['total_frames_processed'])
            col2.metric("Total Tracks", metrics['total_tracks'])
            col3.metric("Avg Track Length", f"{metrics['avg_track_length']:.1f} frames")
            
            # Analisis performa
            st.subheader("💡 Performance Analysis")
            if metrics['precision'] >= 80 and metrics['recall'] >= 80:
                st.success("**Excellent Performance**: Sistem tracking bekerja dengan sangat baik!")
            elif metrics['precision'] >= 70 and metrics['recall'] >= 70:
                st.info("**Good Performance**: Sistem bekerja dengan baik, ada ruang untuk improvement.")
            else:
                st.warning("**Needs Improvement**: Pertimbangkan untuk menyesuaikan parameter tracking.")
        
        # Play video result jika ada file output
        if os.path.exists(results["output_path"]):
            st.subheader("🎬 Video Hasil Tracking")
            play_video(results["output_path"])
        else:
            st.warning("⚠️ File video output tidak tersedia (mungkin proses dihentikan terlalu cepat)")
        
        # Download results
        if os.path.exists(results["csv_path"]):
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(results["output_path"]):
                    with open(results["output_path"], "rb") as f:
                        st.download_button(
                            "📥 Download Video Hasil",
                            f,
                            file_name="tracking_result.mp4",
                            mime="video/mp4"
                        )
            with col2:
                with open(results["csv_path"], "rb") as f:
                    st.download_button(
                        "📊 Download Data CSV",
                        f,
                        file_name="tracking_data.csv",
                        mime="text/csv"
                    )
        else:
            st.error("❌ File data tidak tersedia untuk didownload")

# Display error if any
if processing_state.error:
    st.error("❌ Terjadi error selama processing:")
    st.error(processing_state.error)

# Stop button - ✅ TAMBAHAN: Selalu tampilkan tombol stop untuk RTSP
if processing_state.is_processing:
    if st.button("⏹️ Stop Processing"):
        if 'processing_queue' in st.session_state:
            st.session_state.processing_queue.put({"stop": True})
        processing_state.is_processing = False
        st.warning("🛑 Menghentikan proses... Metrik akan ditampilkan berdasarkan data yang sudah diproses.")
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
    **Fitur RTSP:**
    - ✅ Support RTSP streams
    - ✅ Auto-reconnect jika terputus
    - ✅ Metrik real-time
    - ✅ Partial results saat stop
    
    **Metrik Tracking:**
    - Precision, Recall, F1-Score
    - ID Switches counting
    - Track quality analysis
    - Confidence distribution
    """
)