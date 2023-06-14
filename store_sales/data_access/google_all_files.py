import requests
import os
from urllib.parse import urlparse

def download_file_from_google_drive(url, destination_folder, file_name):
    file_id = urlparse(url).path.split("/")[3]
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={ 'id': file_id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id': file_id, 'confirm': token }
        response = session.get(URL, params=params, stream=True)

    destination_path = os.path.join(destination_folder, file_name)
    save_response_content(response, destination_path)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination_path):
    CHUNK_SIZE = 32768

    with open(destination_path, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    file_urls = [
        "https://drive.google.com/file/d/1eTYCjUxAP_Z8swP1SNRBFfD2tiwwFc_L/view?usp=sharing",
        "https://drive.google.com/file/d/1Wxh7TuaGk51fXClkvUkPk4f07uZ5qz2_/view?usp=sharing",
        "https://drive.google.com/file/d/18i8Hz-Si-D3BC6Q_dGmIODJypUkURB_0/view?usp=sharing",
        "https://drive.google.com/file/d/1th36FID8aN7J98Ws7ezrvIP_Azl9lexj/view?usp=sharing",
        "https://drive.google.com/file/d/1Bz4aTX7oDYpkjT0s5dlhEJWVipk_MYaG/view?usp=sharing",
        "https://drive.google.com/file/d/15IqyMCN45W36ChGQI_omseFnbQgiSJCq/view?usp=sharing",
        "https://drive.google.com/file/d/1ZjvNA8dmUWo-l2ybE2D_RGiEWtO59sIX/view?usp=sharing"
    ]

    file_names = [
        "holiday_events.csv",
        "oil.csv",
        "sample_submission.csv",
        "Stores.csv",
        "test.csv",
        "train.csv",
        "transaction.csv"
    ]

    destination_folder = r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\Store Sales Forcasting using Time series\store-sales-forcasting\store_sales\google_drive_file"
    os.makedirs(destination_folder, exist_ok=True)

    for file_url, file_name in zip(file_urls, file_names):
        download_file_from_google_drive(file_url, destination_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        print("File download complete! File saved at:", destination_path)
