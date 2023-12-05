from typing import Any, Iterator, Optional, Protocol
import boto3
from itertools import chain

import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import requests
from PIL import Image
from toolz.functoolz import pipe

Response = dict[str, Any]


class RekognitionClient(Protocol):
    def create_collection(self, CollectionId: str, **kwargs) -> Response:
        ...

    def index_faces(
        self,
        CollectionId: str,
        Image: dict[str, dict[str, str]],
        DetectionAttributes: list[str],
        **kwargs,
    ) -> Response:
        ...


def initialize_rekognition(region_name: str, **kwargs) -> RekognitionClient:
    """
    This function takes a region names and any required kwargs and initializes
    a Rekognition object with the appropriate parameters.

    Args:
        region_name (str): The name of the AWS region.

    Returns:
        Rekognition: An instantiated Rekognition client.
    """
    return boto3.client("rekognition", region_name, **kwargs)


def initialize_empty_collection(
    client: RekognitionClient, collection_name: str
) -> Response:
    """
    This function takes a Rekognition object and a collection name and initializes
    an empty collection with that ID.

    Args:
        client (Rekognition): The initialized rekognition client.
        collection_name (str): The CollectionId.

    Returns:
        Response: The response json from the rekognition service.
    """
    response = client.create_collection(CollectionId=collection_name)
    return response


def add_single_face_to_collection(
    client: RekognitionClient, bucket_name: str, image_path: str, collection_name: str
) -> Response:
    """
    This function takes a rekognition client, a bucket name, the path of a
    single image and the name of an existing collection and adds the image
    to that collection.

    Args:
        client (Rekognition): The initialized rekognition client.
        bucket_name (str): The name of the selected bucket.
        image_path (str): The path to the image to add.
        collection_name (str): The ID of the collection to add the image to.

    Returns:
        Response: The response json from the rekognition service.
    """
    response = client.index_faces(
        CollectionId=collection_name,
        Image={"S3Object": {"Bucket": bucket_name, "Name": image_path}},
        ExternalImageId=image_path.replace('/', '__'),
        DetectionAttributes=["ALL"],
    )
    return response


def _list_objects_by_format(
    bucket_name: str, file_formats: list[str] = ["jpeg", "png", "jpg"]
) -> Iterator[str]:
    """
    List all objects in an S3 bucket with a specified file format.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_format (list[str]): List of desired file formats
        (e.g., '.jpg', '.txt'). Defaults to ['jpeg', 'png']

    Returns:
        Iterator: Full paths of all objects in the bucket with
        the specified format.
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    # Iterate over the bucket contents using pagination
    pages_with_content = filter(
        lambda page: "Contents" in page, paginator.paginate(Bucket=bucket_name)
    )
    contents = chain(*map(lambda page: page["Contents"], pages_with_content))
    correct_format_objects = filter(
        lambda file: any(
            file["Key"].endswith(file_format) for file_format in file_formats
        ),
        contents,
    )
    object_names = map(lambda file: file["Key"], correct_format_objects)

    return object_names


def add_multiple_faces_to_collection(
    client: RekognitionClient,
    bucket_name: str,
    collection_name: str,
    objects: Optional[list[str]] = None,
    file_formats: list[str] = ["jpeg", "png", "jpg"],
) -> list[Response]:
    """
    This function takes a rekognition client, a bucket name, a collection name and a list
    of file formats and adds all the files in the bucket with the appropriate formats into
    the existing collection.

    Args:
        client (Rekognition): The initialized rekognition client.
        bucket_name (str): The name of the bucket where the images are located.
        collection_name (str): The CollectionId.
        objects (Optional[list[str]]): The explicit list of file names to include
        in the collection, all files must be in the same bucket for the function
        to work correctly. Defaults to None.
        file_formats (list[str], optional): The list of file formats to include.
        Defaults to ["jpeg", "png", "jpg"].

    Returns:
        list[Response]: The list of responses for the addition of each image.
    """
    if objects is not None:
        responses = map(
            lambda file: add_single_face_to_collection(
                client, bucket_name, file, collection_name
            ),
            objects,
        )
    else:
        responses = map(
            lambda file: add_single_face_to_collection(
                client, bucket_name, file, collection_name
            ),
            _list_objects_by_format(bucket_name, file_formats),
        )
    return [*responses]

def download_image(url : str, image_file_path : str) -> str:
    r = requests.get(url, timeout=2.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))
    return image_file_path

def get_image_name(path : str) -> str:
    return path.split('/')[-1]

def download_image(url, image_file_path) -> str:
    response = requests.get(url)
    with Image.open(BytesIO(response.content)) as im:
        im.save(image_file_path)
    return image_file_path

def upload_to_s3(path, bucket_name, client) -> str:
    file_name = path.split('/')[-1]
    client.upload_file(path, bucket_name, file_name)
    return file_name

def download_images_s3_parallel(df, client, image_folder, bucket_name, collection_name, max_workers=12) -> None:

    download = lambda url: download_image(url, image_file_path=os.path.join(image_folder, get_image_name(url)))
    #upload = lambda path: client.upload_file(path, bucket_name, path.split('/')[-1])
    #add_to_collection = lambda path: add_single_face_to_collection(client, bucket_name, path, collection_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(
            tqdm(
                executor.map(
                    lambda url: pipe(url,
                                     download),
                                     #upload,
                                     #add_to_collection),
                    df['docUrl'],
                ),
            total=len(df)
            )
        )

    return

if __name__=='__main__':

    filename = 'dataset/face_reco_urls.csv'
    urls = pd.read_csv(filename)
    selfies_urls = urls[urls['docType']=='selfie'].sample(1000)
    image_folder = 'images'

    s3_client = boto3.client(
        's3',
        aws_access_key_id='', 
        aws_secret_access_key='', 
        region_name='us-east-2'
    )
    #rekognition_client = initialize_rekognition('us-east-2')
    #collection_response = initialize_empty_collection(rekognition_client, 'face_uniqueness_all_images')

    download_images_s3_parallel(selfies_urls, s3_client, image_folder, image_folder, 'baubab-dev')