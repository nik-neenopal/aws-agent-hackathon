import os
import json
import boto3

# Environment variables
S3_BUCKET = os.environ['S3_BUCKET']

s3 = boto3.client("s3") 

def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    filename = body.get("filename")
    if not filename:
        return {"statusCode": 400, "body": "Missing filename"}

    # Upload path: enrichment-folder/users_input/<filename>
    key = f"users_input/{filename}"

    # Generate presigned URL
    url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
            "ContentType": "text/csv"
        },
        ExpiresIn=300  # 5 minutes
    )

    return {
    "statusCode": 200,
    "body": json.dumps({"url": url, "filename": key}),
    "headers": {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET,PUT",
        "Access-Control-Allow-Headers": "*",
        'Content-Type': 'application/json'
    }
}

