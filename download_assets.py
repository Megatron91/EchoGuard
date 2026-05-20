import os
import urllib.request

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        # User-Agent header to avoid potential bot blocks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response:
            with open(destination, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Successfully downloaded to {destination}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e

def main():
    base_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(base_dir, "assets")
    samples_dir = os.path.join(assets_dir, "samples")
    
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # YAMNet ONNX Model URL from Hugging Face
    model_url = "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx"
    model_dest = os.path.join(assets_dir, "yamnet.onnx")
    
    # Class map CSV URL
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    class_map_dest = os.path.join(assets_dir, "yamnet_class_map.csv")
    
    if not os.path.exists(model_dest):
        download_file(model_url, model_dest)
    else:
        print(f"Model file already exists: {model_dest}")
        
    if not os.path.exists(class_map_dest):
        download_file(class_map_url, class_map_dest)
    else:
        print(f"Class map already exists: {class_map_dest}")

    # Real-world screaming, crying and normal samples from Archive.org
    real_samples = {
        "real_baby_crying.wav": "https://archive.org/download/valentinosfxdvd1cd9/01.%20Baby%20-%20Crying%20%28infant%29.wav",
        "real_woman_screaming.wav": "https://archive.org/download/valentinosfxdvd1cd9/16.%20Screaming%20-%20Woman.wav",
        "real_man_screaming.wav": "https://archive.org/download/valentinosfxdvd1cd9/36.%20Screaming%20-%20Man.wav",
        "real_child_crying.wav": "https://archive.org/download/Red_Library_Voices_Children/R15-57-Child%20Crying.wav",
        "real_woman_crying.wav": "https://archive.org/download/Red_Library_Voices_Women/R15-44-Woman%20Crying%20Angrily.wav",
        "normal_typewriter.wav": "https://archive.org/download/valentinosfxdvd1cd9/59.%20Typewriter%20-%20Two%2C%20Manual.wav",
        "normal_laughter.wav": "https://archive.org/download/valentinosfxdvd1cd9/14.%20Laughing%20-%20Woman.wav",
        "normal_forest.wav": "https://archive.org/download/valentinosfxdvd1cd9/38.%20Forest%20-%20Ambience%20W%20Birds.wav"
    }

    for name, url in real_samples.items():
        dest = os.path.join(samples_dir, name)
        if not os.path.exists(dest):
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"Skipping {name} due to download error: {e}")
        else:
            print(f"Sample file already exists: {dest}")

if __name__ == "__main__":
    main()
