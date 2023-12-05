import requests
from functools import partial
from itertools import chain
import json
from typing import Any, Callable, Iterator, Optional, Protocol
import boto3

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


class S3Client(Protocol):
    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs=Optional[dict[str, any]],
        Callback=Optional[Callable],
    ) -> None:
        ...

    def put_object(self, Bucket: str, Key: str, Body: str, **kwargs) -> None:
        ...

    def delete_object(self, Bucket: str, Key: str, **kwargs) -> None:
        ...


def initialize_rekognition(region_name: str, **kwargs) -> RekognitionClient:
    """
    This function takes a region names and any required kwargs and initializes
    a Rekognition object with the appropriate parameters.

    Args:
        region_name (str): The name of the AWS region.

    Returns:
        RekognitionClient: An instantiated Rekognition client.
    """
    return boto3.client("rekognition", region_name, **kwargs)


def initialize_s3(region_name: str, **kwargs) -> S3Client:
    """
    This function takes a region names and any required kwargs and initializes
    an S3 client object with the appropriate parameters.

    Args:
        region_name (str): The name of the AWS region.

    Returns:
        S3Client: An instantiated S3 client.
    """
    return boto3.client("s3", region_name, **kwargs)


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
        ExternalImageId=image_path.replace("/", "__"),
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


def upload_file_to_s3(path: str, bucket_name: str, client: S3Client) -> str:
    file_name = path.split("/")[-1]
    client.upload_file(path, bucket_name, file_name)
    return file_name


def upload_object_to_s3(
    obj: Any, file_name: str, bucket_name: str, client: S3Client
) -> bool:
    """
    This function uploads a python object to S3.

    Args:
        obj (Any): The object to upload.
        file_name (str): The name to use for storing the file.
        bucket_name (str): The name of the bucket.
        client (S3Client): An initialized S3 client.

    Returns:
        bool: True if it was uploaded correctly, False otherwise.
    """
    try:
        client.put_object(Bucket=bucket_name, Key=file_name, Body=obj)
        return True
    except Exception as e:
        return False


def compare_image(
    file: str, bucket: str, collection: str, client: RekognitionClient
) -> Response:
    """
    This function takes the path of an image in an S3 bucket and compares
    it to a provided collection.

    Args:
        bucket (str): The name of the bucket where the file is located.
        collection (str): The name of the collection to use for comparing.
        file (str): The name of the file.
        client (RekognitionClient): An initialized Rekognition client.

    Returns:
        Response: The response of the comparison.
    """
    try:
        response = client.search_faces_by_image(
            CollectionId=collection,
            Image={"S3Object": {"Bucket": bucket, "Name": file}},
            MaxFaces=1,
            FaceMatchThreshold=0,
        )
        return response
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(e),
        }


def compare_images(
    file_names: Iterator[str], bucket: str, collection: str, client: RekognitionClient
) -> list[Response]:
    """
    This function takes the paths of multiple images in an S3 bucket
    and compares them to a provided collection.

    Args:
        bucket (str): The name of the bucket where the files are located.
        collection (str): The name of the collection to use for comparing.
        file_names (Iterator[str]): The name of the files.
        client (RekognitionClient): An initialized Rekognition client.

    Returns:
        list[Response]: A list of responses for each comparison.
    """
    compare = partial(
        compare_image, bucket=bucket, collection=collection, client=client
    )
    return [*map(compare, file_names)]


def construct_filename(url: str, user_id: int, img_type: str) -> str:
    """Constructs the filename based on the provided criteria."""
    # Extract the filename without the extension from the URL
    original_filename = url.split("/")[-1].rsplit(".", 1)[0]

    # Extract date and time details from the URL
    date_parts = url.split("/")[-6:-1]

    # Construct the new filename
    new_filename = (
        f"{user_id}_{img_type}_{'_'.join(date_parts)}_{original_filename}.jpeg"
    )

    return new_filename


def lambda_compare_images(event: dict[str, str], context: dict[str, str]) -> Response:
    """
    The function will add all three images to a temporary bucket, compare
    them to the provided rekognition collection, and then add them to the
    collection. Afterwards, they will be deleted from the bucket.

    'adding results', is referent to the information returner by AWS at
    the moment of adding one image to a collection.

    Args:
        event (dict[str, str]): The event, on the other hand, must be
        structured as follows:
        {
            'userId': 'user_id',
            'legalFront': 'legalFront_url',
            'selfie1': 'selfie1_url',
            'selfie2': 'selfie2_url'
        }
        context (dict[str, str]): The context must be a dictionary as
        follows:
        {
            'collection': 'collection_name',
            'bucket': 'bucket_name',
            'region': 'region_name'
        }

    Returns:
        Response: The response will be structured in the following way:
        {
            'user_id': 'user_id',
            'comparisons': {...},
            'adding_results': {...}
        }
    """
    collection = context.get("collection")
    bucket = context.get("bucket")
    region = context.get("region")

    rekognition_client = initialize_rekognition(region)
    s3_client = initialize_s3(region)

    url_types = ["selfie1", "selfie2", "legalFront"]

    user_id = event.get("userId")
    image_urls = [*map(event.get, url_types)]
    image_names = [
        *map(
            lambda url, url_type: construct_filename(url, user_id, url_type),
            image_urls,
            url_types,
        )
    ]

    responses = map(requests.get, image_urls)
    images_data = map(lambda response: response.content, responses)

    put_object = partial(s3_client.put_object, Bucket=bucket)
    for name, data in zip(image_names, images_data):
        put_object(Key=name, Body=data)

    comparison_results = compare_images(
        image_names, bucket, collection, rekognition_client
    )

    adding_collection_results = add_multiple_faces_to_collection(
        rekognition_client, bucket, collection, image_names
    )

    for file in image_names:
        s3_client.delete_object(Key=file, Bucket=bucket)

    return {
        "user_id": user_id,
        "comparisons": comparison_results,
        "adding_results": adding_collection_results,
    }


if __name__ == "__main__":
    event = {"userId": "98",
             "selfie1": "https://baubap.s3.us-east-2.amazonaws.com/documents/selfie/2023/07/16/07/35/lMd4oSiEEzumduxG.jpeg",
             "selfie2": "https://baubap.s3.us-east-2.amazonaws.com/documents/selfie/2023/07/16/07/35/M0238waGes89fhDX.jpeg",
             "legalFront": "https://baubap.s3.us-east-2.amazonaws.com/documents/legalFront/2023/07/16/07/34/rleH8jntWxPKgC0U.jpeg",}
    context = {
        "collection": "face_uniqueness_all_images",
        "bucket": "faces-bucket-test",  # Make sure this bucket exist in PROD, or substitute for the designated bucket.
    }
    response_json = lambda_compare_images(event, context)
    print(response_json)