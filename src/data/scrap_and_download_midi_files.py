import os
import requests
from bs4 import BeautifulSoup

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}")

def scrape_midi_files(base_url, save_dir, genre):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.mid'):
            file_url = os.path.join(base_url, href)
            file_name = f"{genre}_{os.path.basename(href)}"
            save_path = os.path.join(save_dir, file_name)
            download_file(file_url, save_path)

if __name__ == "__main__":
    base_urls = {
        'classical': 'http://example.com/classical/',
        'jazz': 'http://example.com/jazz/',
        'rock': 'http://example.com/rock/'
    }
    save_dir = '../data/'

    for genre, url in base_urls.items():
        genre_save_dir = os.path.join(save_dir, genre)
        scrape_midi_files(url, genre_save_dir, genre)
