import os
import csv
import numpy as np
import onnxruntime as ort

class YAMNetClassifier:
    def __init__(self, model_path=None, class_map_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "assets", "yamnet.onnx")
        if class_map_path is None:
            class_map_path = os.path.join(base_dir, "assets", "yamnet_class_map.csv")
            
        self.model_path = model_path
        self.class_map_path = class_map_path
        
        # Load labels
        self.class_map = self._load_class_map(self.class_map_path)
        
        # Load ONNX model session
        print(f"Initializing YAMNet ONNX session with model: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Threat categories definition mapping display_name to general threat category
        self.threat_map = {
            "Scream": "Scream",
            "Screaming": "Scream",
            "Yell": "Scream",
            "Shout": "Scream",
            "Crying, sobbing": "Distress/Crying",
            "Crying": "Distress/Crying",
            "Sobbing": "Distress/Crying",
            "Baby cry, infant cry": "Distress/Crying",
            "Whimper": "Distress/Crying",
            "Glass": "Glass Breaking",
            "Shatter": "Glass Breaking",
            "Glass breaking": "Glass Breaking",
            "Gunshot, gunfire": "Gunshot",
            "Explosion": "Gunshot",
            "Cap gun": "Gunshot",
            "Firearm": "Gunshot",
            "Siren": "Siren/Alarm",
            "Alarm": "Siren/Alarm"
        }

        # Category-specific confidence thresholds to prevent false positives on everyday sounds
        self.thresholds = {
            "Scream": 0.30,
            "Distress/Crying": 0.20,
            "Glass Breaking": 0.35,
            "Gunshot": 0.20,
            "Siren/Alarm": 0.45
        }

    def _load_class_map(self, class_map_path):
        class_map = {}
        if not os.path.exists(class_map_path):
            print(f"Warning: Class map file not found at {class_map_path}")
            return class_map
            
        try:
            with open(class_map_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header if it exists
                header = next(reader)
                has_header = "index" in header or "display_name" in header
                if not has_header:
                    f.seek(0)
                else:
                    print("Skipping class map header row...")
                    
                for row in reader:
                    if len(row) >= 3:
                        idx = int(row[0])
                        display_name = row[2].strip().strip('"')
                        class_map[idx] = display_name
        except Exception as e:
            print(f"Error loading class map CSV: {e}")
        return class_map

    def energy_vad(self, waveform, threshold=0.003):
        """
        Simple Voice/Audio Activity Detection based on Root Mean Square (RMS) energy.
        """
        if len(waveform) == 0:
            return False, 0.0
        rms = np.sqrt(np.mean(waveform ** 2))
        return rms > threshold, float(rms)

    def classify(self, waveform, vad_threshold=0.003):
        """
        Takes a 1D float32 numpy array representing raw audio sampled at 16000Hz mono.
        Returns a dictionary containing raw output and classified safety/threat events.
        """
        # Ensure correct type and shape
        waveform = np.asarray(waveform, dtype=np.float32)
        
        # Run Voice Activity Detection first to save processing
        is_active, rms_energy = self.energy_vad(waveform, vad_threshold)
        
        if not is_active:
            return {
                "active": False,
                "rms_energy": rms_energy,
                "threat_detected": False,
                "detections": []
            }
            
        # Run ONNX inference
        # The model expects a float32 tensor representing the raw audio waveform
        outputs = self.session.run(None, {self.input_name: waveform})
        
        # Output structure:
        # output_0: scores (shape: [num_frames, 521])
        # output_1: embeddings (shape: [num_frames, 1024])
        # output_2: log-mel spectrogram frames (shape: [num_spectrogram_frames, 64])
        scores = outputs[0]
        
        num_frames = scores.shape[0]
        detections = []
        threat_detected = False
        primary_threat = None
        highest_threat_score = 0.0
        
        # YAMNet analysis frame windowing parameters:
        # Each prediction frame corresponds to 0.96 seconds of audio, with a hop of 0.48 seconds
        for frame_idx in range(num_frames):
            frame_scores = scores[frame_idx]
            timestamp_offset = frame_idx * 0.48
            
            # Find the top predicted classes for this frame
            top_indices = np.argsort(frame_scores)[::-1][:5]
            
            frame_detections = []
            for idx in top_indices:
                score = float(frame_scores[idx])
                display_name = self.class_map.get(idx, f"Unknown ({idx})")
                
                # Check if class matches threat categories
                threat_category = None
                display_name_lower = display_name.lower()
                
                # Exclude non-threat alarms/clocks
                if "alarm clock" in display_name_lower:
                    pass
                # Exclude crying if it is actually laughter or giggling (often misclassified)
                elif "crying" in display_name_lower or "sobbing" in display_name_lower:
                    is_actually_laughter = False
                    for other_idx in top_indices:
                        other_score = float(frame_scores[other_idx])
                        other_name = self.class_map.get(other_idx, "").lower()
                        if any(kw in other_name for kw in ["laughter", "laugh", "giggle", "snicker", "chuckle", "guffaw"]) and other_score > 0.10:
                            is_actually_laughter = True
                            break
                    if not is_actually_laughter:
                        threat_category = "Distress/Crying"
                else:
                    for key, val in self.threat_map.items():
                        if key.lower() in display_name_lower:
                            threat_category = val
                            break
                
                detection_item = {
                    "display_name": display_name,
                    "score": score,
                    "index": int(idx),
                    "threat_category": threat_category
                }
                frame_detections.append(detection_item)
                
                # Update threat status
                if threat_category is not None:
                    threshold = self.thresholds.get(threat_category, 0.15)
                    # If crying/sobbing is not the primary (top 1) prediction, require higher confidence
                    if threat_category == "Distress/Crying" and idx != top_indices[0]:
                        threshold = 0.40
                        
                    if score > threshold:
                        threat_detected = True
                        if score > highest_threat_score:
                            highest_threat_score = score
                            primary_threat = {
                                "category": threat_category,
                                "display_name": display_name,
                                "score": score,
                                "timestamp": timestamp_offset
                            }
            
            detections.append({
                "time_offset": timestamp_offset,
                "predictions": frame_detections
            })
            
        return {
            "active": True,
            "rms_energy": rms_energy,
            "threat_detected": threat_detected,
            "primary_threat": primary_threat,
            "detections": detections
        }
