import os
import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}")

def download_midi_files(urls, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for url in urls:
        file_name = os.path.basename(url)
        save_path = os.path.join(save_dir, file_name)
        download_file(url, save_path)

if __name__ == "__main__":
    # Example URLs of MIDI files
    classical_urls = [
        "http://example.com/path/to/classical/file1.mid",
        "http://example.com/path/to/classical/file2.mid",
    ]

    jazz_urls = [
        "http://example.com/path/to/jazz/file1.mid",
        "http://example.com/path/to/jazz/file2.mid",
    ]

    rock_urls = [
        "http://example.com/path/to/rock/file1.mid",
        "http://example.com/path/to/rock/file2.mid",
    ]

    download_midi_files(classical_urls, "../data/classical/")
    download_midi_files(jazz_urls, "../data/jazz/")
    download_midi_files(rock_urls, "../data/rock/")
