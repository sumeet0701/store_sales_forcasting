import os
import urllib.parse
import requests
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class GoogleDriveDownloader:
    def __init__(self, url, file_name, destination_folder):
        self.url = url
        self.file_name = file_name
        self.destination_folder = destination_folder

    def download(self):
        file_id = urllib.parse.urlparse(self.url).path.split("/")[3]
        destination_path = os.path.join(self.destination_folder, self.file_name)

        drive_service = build('drive', 'v3', cache_discovery=False)
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        print("File download complete! File saved at:", destination_path)
