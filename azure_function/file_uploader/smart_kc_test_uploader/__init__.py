import logging

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os

# Get connection string and secrets
account_url = "https://smartkcstorage1.blob.core.windows.net"
default_credential = DefaultAzureCredential()
container_name = 'test-container'
upload_secret = os.environ['upload_secret']


def main(req: func.HttpRequest) -> func.HttpResponse:
    files = list(req.files.values())
    logging.info('Python HTTP trigger function processed a request.')
    
    # Check for upload_secret
    user_secret = req.headers.get('upload_secret', '')
    if (user_secret != upload_secret):
        return func.HttpResponse(
             "Unauthorized",
             status_code=401
        )
    
    if (len(list(files)) > 1):
        return func.HttpResponse(
             "Please upload one file at a time",
             status_code=400
        )
    input_file = files[0]
    filename = input_file.filename
    contents = input_file.stream.read()
    
    logging.info('Filename: %s' % filename)
    logging.info('Contents:')
    
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    if blob_client.exists():
        return func.HttpResponse(
             "The file already exists",
             status_code=200
        )
    blob_client.upload_blob(contents)
    return func.HttpResponse(
             "Uploaded file successfully",
             status_code=200
        )
