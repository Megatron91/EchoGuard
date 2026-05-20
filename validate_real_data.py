import os
import librosa
from audio_engine import YAMNetClassifier

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "assets", "samples")
    
    # Initialize classifier
    classifier = YAMNetClassifier()
    
    # Files to validate
    real_files = [
        "real_baby_crying.wav",
        "real_woman_screaming.wav",
        "real_man_screaming.wav",
        "real_child_crying.wav",
        "real_woman_crying.wav"
    ]
    
    # Normal control files (should not detect threats)
    normal_files = [
        "normal_forest.wav",
        "normal_laughter.wav",
        "normal_typewriter.wav"
    ]
    
    print("\n" + "="*80)
    print("YAMNET OFFLINE VALIDATION OF REAL-WORLD AUDIO SAMPLES")
    print("="*80)
    
    print("\n>>> VALIDATING THREAT FILES (Expected: Threat Detected = True)")
    for filename in real_files:
        file_path = os.path.join(samples_dir, filename)
        if not os.path.exists(file_path):
            print(f"\n[ERROR] File not found: {filename}")
            continue
            
        print(f"\nProcessing: {filename} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)...")
        try:
            waveform, sr = librosa.load(file_path, sr=16000)
            duration = len(waveform) / 16000
            print(f"  Loaded successfully. Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
            result = classifier.classify(waveform)
            
            print(f"  Voice Activity Detected (VAD): {result['active']} (RMS Energy: {result['rms_energy']:.5f})")
            print(f"  Threat Detected: {result['threat_detected']} (Expected: True/True-ish)")
            
            if result['threat_detected']:
                primary = result['primary_threat']
                print(f"  >>> PRIMARY THREAT: Category: [{primary['category']}] | Name: '{primary['display_name']}' | Score: {primary['score']:.4f} | Time: {primary['timestamp']:.2f}s")
            
            print("  Top frame detections:")
            frames_shown = 0
            for frame in result.get("detections", []):
                top_pred = frame["predictions"][0]
                if top_pred["score"] > 0.15 or frames_shown < 3:
                    preds_str = ", ".join([f"'{p['display_name']}' ({p['score']:.2f})" for p in frame["predictions"][:3]])
                    print(f"    Offset {frame['time_offset']:.2f}s: {preds_str}")
                    frames_shown += 1
                if frames_shown >= 5:
                    print("    ...")
                    break
                    
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    print("\n" + "="*80)
    print(">>> VALIDATING NORMAL CONTROL FILES (Expected: Threat Detected = False)")
    for filename in normal_files:
        file_path = os.path.join(samples_dir, filename)
        if not os.path.exists(file_path):
            print(f"\n[ERROR] File not found: {filename}")
            continue
            
        print(f"\nProcessing: {filename} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)...")
        try:
            waveform, sr = librosa.load(file_path, sr=16000)
            duration = len(waveform) / 16000
            print(f"  Loaded successfully. Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
            result = classifier.classify(waveform)
            
            print(f"  Voice Activity Detected (VAD): {result['active']} (RMS Energy: {result['rms_energy']:.5f})")
            print(f"  Threat Detected: {result['threat_detected']} (Expected: False)")
            
            if result['threat_detected']:
                primary = result['primary_threat']
                print(f"  [WARNING] FALSE POSITIVE TRIGGER: Category: [{primary['category']}] | Name: '{primary['display_name']}' | Score: {primary['score']:.4f} | Time: {primary['timestamp']:.2f}s")
                # Print the exact triggering frame's details
                for frame in result.get("detections", []):
                    if abs(frame['time_offset'] - primary['timestamp']) < 0.01:
                        preds_str = ", ".join([f"'{p['display_name']}' ({p['score']:.2f})[threat={p['threat_category']}]" for p in frame["predictions"]])
                        print(f"    Triggering Frame predictions at {frame['time_offset']:.2f}s: {preds_str}")
            else:
                print("  [PASS] No safety threat detected.")
            
            print("  Top frame detections:")
            frames_shown = 0
            for frame in result.get("detections", []):
                preds_str = ", ".join([f"'{p['display_name']}' ({p['score']:.2f})" for p in frame["predictions"][:3]])
                print(f"    Offset {frame['time_offset']:.2f}s: {preds_str}")
                frames_shown += 1
                if frames_shown >= 3:
                    break
                    
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")
            
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
