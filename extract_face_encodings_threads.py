import concurrent.futures
from tqdm import tqdm
from glob import glob
import  face_recognition as fr
import pandas as pd
import base64
import multiprocessing

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
        print(f'Failed to process image: {image_file}')
        return None

def process_image(image_file):
    face_enc = extract_face_encodings(image_file)
    return [image_file, face_enc]

def save_results(image_dataset):
    df_face_encs = pd.DataFrame(image_dataset, columns=['image_name', 'face_encoding'])
    df_face_encs['blob_face_encoding'] = df_face_encs['face_encoding'].apply(lambda x: base64.b64encode(x.tobytes()).decode('utf-8') if x is not None else None)

    df_face_encs_processed = df_face_encs[~pd.isna(df_face_encs['face_encoding'])]
    df_face_encs_processed[['image_name', 'blob_face_encoding']].to_csv('dataset/face_encodings_processed.csv', index=False)

    df_face_encs_non_processed = df_face_encs[pd.isna(df_face_encs['face_encoding'])]
    df_face_encs_non_processed[['image_name', 'blob_face_encoding']].to_csv('dataset/face_encodings_non_processed.csv', index=False)

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', force=True)
    image_files = glob('images/*')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_files), total=len(image_files)))

    image_dataset = [result for result in results if result[1] is not None]

    print('Saving results')
    save_results(image_dataset)
