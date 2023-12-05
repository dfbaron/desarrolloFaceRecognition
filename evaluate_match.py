import  face_recognition as fr
import numpy as np
import pandas as pd
from glob import glob

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import requests
from io import BytesIO
from PIL import Image
import os
import base64

def download_image(url : str, image_file_path : str) -> str:
    r = requests.get(url, timeout=2.0)
    image_name = os.path.join(image_file_path, url.split('/')[-1])
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_name)
    return image_name

def find_biggest_bounding_box(bounding_boxes):
    return max(bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]), default=None)


def extract_face_encodings(image_file):
    try:
        image = fr.load_image_file(image_file)
        face_locations = fr.face_locations(image, model="hog")
        bigger_face = find_biggest_bounding_box(face_locations)
        face_encodings = fr.face_encodings(image, [bigger_face])
        return face_encodings[0]
    except:
        print('Failed to process image')
        return None
    
def find_matches(url, known_face_encodings):
    image_path = download_image(url, 'images_tmp')
    face_enc = extract_face_encodings(image_path)
    matches = fr.compare_faces(known_face_encodings, face_enc, 0.4)
    os.remove(image_path)
    return max(matches)

#TEST
df_urls = pd.read_csv('dataset/face_reco_urls.csv')
df_face_encs = pd.read_csv('dataset/face_encodings_processed.csv')
df_face_encs['face_encoding'] = df_face_encs['blob_face_encoding'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.float64))
known_face_encodings = np.vstack(df_face_encs['face_encoding'])

url = df_urls['docUrl'].iloc[-1]
find_matches(url, known_face_encodings)