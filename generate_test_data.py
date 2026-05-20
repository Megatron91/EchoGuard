import os
import numpy as np
import scipy.io.wavfile as wav

def generate_silence(duration=2.0, sr=16000):
    return np.zeros(int(sr * duration), dtype=np.float32)

def generate_scream(duration=2.0, sr=16000):
    """
    Generates a synthetic screaming sound:
    Multiple frequency-modulated sine waves in the vocal distress range (1000 - 3000 Hz)
    with frequency jitter and amplitude modulation.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Fundamental frequency swept from 1000Hz to 1600Hz and back
    f_mod = 1200 + 400 * np.sin(2 * np.pi * 5 * t) + 100 * np.sin(2 * np.pi * 20 * t) # adding jitter
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    signal = np.sin(phase)
    
    # Add harmonics (screams have rich harmonic structures)
    signal += 0.5 * np.sin(2 * phase)
    signal += 0.25 * np.sin(3 * phase)
    
    # Add some high-frequency vocal friction noise
    noise = np.random.normal(0, 0.15, len(t))
    # Bandpass filter the noise roughly around distress frequencies
    # Simple spectral shaping using moving average
    noise = np.convolve(noise, np.ones(5)/5, mode='same')
    signal += noise
    
    # Apply fade in/out envelope
    envelope = np.ones_like(t)
    fade_len = int(sr * 0.1)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    # Amplitude modulation to sound shaky
    am = 0.8 + 0.2 * np.sin(2 * np.pi * 8 * t)
    signal = signal * envelope * am
    
    # Normalize to [-0.8, 0.8]
    signal = 0.8 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def generate_glass_shatter(duration=2.0, sr=16000):
    """
    Generates a synthetic glass shatter sound:
    High-pass filtered white noise burst with a sharp onset and exponential decay,
    combined with transient ringing high-frequency sinusoidal tones (resonant frequencies of glass shards).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Exponential decay envelope
    envelope = np.exp(-4 * t)
    
    # High frequency noise
    noise = np.random.normal(0, 0.5, len(t))
    # Simple high-pass: difference between adjacent samples
    noise = np.diff(noise, prepend=0)
    
    # Ringing glass shard resonant frequencies (e.g., 3000Hz, 4500Hz, 6000Hz)
    f_shards = [3200, 4800, 6500, 7500]
    shards_signal = np.zeros_like(t)
    for f in f_shards:
        decay = np.exp(-np.random.uniform(5, 15) * t)
        shards_signal += np.sin(2 * np.pi * f * t) * decay
        
    signal = (noise * 0.4 + shards_signal * 0.6) * envelope
    
    # Normalize
    signal = 0.8 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def generate_gunshot(duration=1.5, sr=16000):
    """
    Generates a synthetic gunshot sound:
    Extremely sharp attack (impulse) followed by an explosive blast decay
    and lower frequency rumble/echo decay.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Sharp shockwave onset + exponential decay (fast for blast, slow for echo)
    blast_env = np.exp(-25 * t)
    reverb_env = np.exp(-3 * t)
    
    # White noise
    noise = np.random.normal(0, 0.8, len(t))
    
    # Lower frequency explosive thump (sine wave starting high and decaying fast)
    sweep_freq = 300 * np.exp(-50 * t)
    sweep_phase = 2 * np.pi * np.cumsum(sweep_freq) / sr
    thump = np.sin(sweep_phase) * blast_env
    
    signal = (noise * blast_env * 0.7) + (noise * reverb_env * 0.2) + (thump * 0.5)
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def generate_crying(duration=2.5, sr=16000):
    """
    Generates a sobbing/crying sound:
    Repeated low frequency sigh-like pitch drops (200Hz -> 100Hz) with intake gasps.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Repeated cycles of sobbing (every 0.6 seconds)
    cycle = 0.6
    t_mod = t % cycle
    
    # Pitch drops from 350Hz down to 180Hz in each sob
    freq = 180 + 170 * np.exp(-8 * t_mod)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    
    # Sobbing envelope (high intensity at start of sob, fading)
    sob_env = np.exp(-4 * t_mod) * (1 - np.exp(-40 * t_mod)) # fast attack, slower decay
    
    signal = np.sin(phase) * sob_env
    
    # Normalize
    signal = 0.6 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def generate_police_siren(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sweeps from 600Hz to 1300Hz every 1.5 seconds (wail pattern)
    f_mod = 950 + 350 * np.sin(2 * np.pi * 0.67 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    signal = np.sin(phase)
    # Add simple harmonic
    signal += 0.3 * np.sin(2 * phase)
    # Fade in / out slightly
    envelope = np.ones_like(t)
    fade_len = int(sr * 0.15)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    signal = signal * envelope
    # Normalize
    signal = 0.7 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def generate_industrial_alarm(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pulsing square envelope (frequency of 3.3 Hz, duty cycle 50%)
    pulse_env = (np.sin(2 * np.pi * 3.3 * t) > 0).astype(np.float32)
    # 880 Hz tone (A5 note, very piercing)
    signal = np.sin(2 * np.pi * 880 * t) * pulse_env
    # Normalize
    signal = 0.75 * signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)

def main():
    samples_dir = os.path.join(os.path.dirname(__file__), "assets", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    generators = {
        "silence.wav": generate_silence,
        "scream.wav": generate_scream,
        "glass_shatter.wav": generate_glass_shatter,
        "gunshot.wav": generate_gunshot,
        "crying.wav": generate_crying,
        "police_siren.wav": generate_police_siren,
        "industrial_alarm.wav": generate_industrial_alarm
    }
    
    for filename, generator in generators.items():
        filepath = os.path.join(samples_dir, filename)
        data = generator()
        wav.write(filepath, 16000, data)
        print(f"Generated sample: {filepath} (shape: {data.shape})")

if __name__ == "__main__":
    main()
