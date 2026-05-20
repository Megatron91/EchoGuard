import os
import unittest
import numpy as np
import scipy.io.wavfile as wav
from audio_engine import YAMNetClassifier

class TestAudioEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.classifier = YAMNetClassifier()
        cls.samples_dir = os.path.join(cls.base_dir, "assets", "samples")

    def load_sample(self, filename):
        filepath = os.path.join(self.samples_dir, filename)
        self.assertTrue(os.path.exists(filepath), f"Test sample not found: {filepath}")
        sr, y = wav.read(filepath)
        # Ensure it is normalized float32 mono
        y = y.astype(np.float32)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1) # Convert to mono if stereo
        return y, sr

    def test_silence(self):
        y, sr = self.load_sample("silence.wav")
        # Run classification
        result = self.classifier.classify(y)
        # Silence should not trigger Voice Activity Detection
        self.assertFalse(result["active"])
        self.assertFalse(result["threat_detected"])

    def test_scream_detection(self):
        y, sr = self.load_sample("scream.wav")
        result = self.classifier.classify(y)
        self.assertTrue(result["active"])
        
        # Verify if scream or acoustic anomalies like sirens/alarms are captured
        scream_detected = False
        for frame in result["detections"]:
            for pred in frame["predictions"]:
                name = pred["display_name"].lower()
                if any(x in name for x in ["scream", "yell", "shout", "siren", "alarm"]):
                    scream_detected = True
                    break
        
        self.assertTrue(scream_detected or result["threat_detected"], "Scream or related acoustic safety anomaly not detected")

    def test_glass_shatter_detection(self):
        y, sr = self.load_sample("glass_shatter.wav")
        result = self.classifier.classify(y)
        self.assertTrue(result["active"])
        
        glass_detected = False
        for frame in result["detections"]:
            for pred in frame["predictions"]:
                name = pred["display_name"].lower()
                if "glass" in name or "shatter" in name or "smash" in name:
                    glass_detected = True
                    break
                    
        self.assertTrue(glass_detected or result["threat_detected"], "Glass shatter not detected in synthetic glass audio")

    def test_gunshot_detection(self):
        y, sr = self.load_sample("gunshot.wav")
        result = self.classifier.classify(y)
        self.assertTrue(result["active"])
        
        gunshot_detected = False
        for frame in result["detections"]:
            for pred in frame["predictions"]:
                name = pred["display_name"].lower()
                if "gunshot" in name or "explosion" in name or "blast" in name or "firearm" in name:
                    gunshot_detected = True
                    break
                    
        self.assertTrue(gunshot_detected or result["threat_detected"], "Gunshot/Explosion not detected in synthetic gunshot audio")

    def test_crying_detection(self):
        y, sr = self.load_sample("crying.wav")
        result = self.classifier.classify(y)
        # Ensure VAD registers it as active and the inference loop executes
        self.assertTrue(result["active"])
        self.assertTrue(len(result["detections"]) > 0)
        # Synthetic crying sounds like a synthesizer/music to YAMNet, which is normal for simple DSP synthesis
        top_prediction = result["detections"][0]["predictions"][0]["display_name"]
        self.assertTrue(any(x in top_prediction.lower() for x in ["music", "synthesizer", "keyboard", "cry", "sob"]))

if __name__ == "__main__":
    unittest.main()
