import pathlib
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
)

local = pathlib.Path(__file__).parent.parent / "data"
bucket = "alphachess"
prefix = "data"

for f in local.rglob("*"):
    if f.is_file():
        key = f"{prefix}/{f.relative_to(local).as_posix()}"
        s3.upload_file(str(f), bucket, key)
        print(f"uploaded {key}")