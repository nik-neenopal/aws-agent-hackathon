import os
import json
import uuid
import boto3
import urllib.parse
from datetime import datetime


# Initialize AWS clients
s3 = boto3.client('s3')
bedrock_client = boto3.client('bedrock-agentcore', region_name='us-west-2')

# Environment Variables
agent_runtime_arn = os.environ.get("BEDROCK_AGENT_ARN")

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        #  Extract S3 info from trigger event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])

        input_s3_uri = f"s3://{bucket}/{key}"
        print(f"Triggered for file: {input_s3_uri}")

        # ---  Parse bucket & key ---
        parsed = urllib.parse.urlparse(input_s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # ---  Derive base name from file 
        base_name = os.path.splitext(os.path.basename(key))[0]
        output_key = f"outputs/rds_output.csv"
        output_s3_uri = f"s3://{bucket}/{output_key}"

        print(f"Generated Output CSV Path: {output_s3_uri}")

        # ---  Build payload as per Bedrock Agent input format ---
        payload = json.dumps({
            "csv_s3_path": input_s3_uri,
            "output_s3_path": output_s3_uri
        })

        # ---  Generate unique runtime session ID ---

        runtime_session_id = f"session-{uuid.uuid4().hex}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # ---  Invoke Bedrock Agent Runtime ---
        response = bedrock_client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            runtimeSessionId=runtime_session_id,
            payload=payload,
            qualifier="DEFAULT"
        )

        # ---  Parse Bedrock response ---
        response_body = response['response'].read()
        try:
            response_data = json.loads(response_body)
        except json.JSONDecodeError:
            response_data = {"raw_response": response_body.decode("utf-8")}

        print("Agent Response:", json.dumps(response_data, indent=2))

        # ---  Return success info ---
        return {
            "statusCode": 200,
            "body": json.dumps({
                "input_csv": input_s3_uri,
                "output_csv": output_s3_uri,
                "agent_response": response_data
            })
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
