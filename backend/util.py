import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
from fastapi import HTTPException

load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

def upload_file_to_s3(file, file_name, folder="uploads/"):
    """Upload a file to an S3 bucket and return the public URL."""
    try:
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, f"{folder}{file_name}", ExtraArgs={"ACL": "public-read"})
        file_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{folder}{file_name}"
        print(file_url)
        return file_url
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found")
    
def delete_file_from_s3(file_name, folder="uploads/"):
    """Delete a file from an S3 bucket."""
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=f"{folder}{file_name}.png")
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=f"{folder}{file_name}.obj")
        return {"message": f"{file_name} File deleted successfully"}
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found")