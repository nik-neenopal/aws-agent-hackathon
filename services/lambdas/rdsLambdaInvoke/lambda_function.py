import os
import csv
import json
import boto3
import logging
from io import StringIO
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Environment variables

S3_BUCKET = os.environ['S3_BUCKET']
S3_FOLDER = os.environ.get('S3_FOLDER')

# Initialize S3 client
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to process row data from PostgreSQL RDS
    Convert to CSV and upload to S3
    
    Args:
        event: Contains the row data from PostgreSQL
        context: Lambda context object
    
    Returns:
        Response object with status and processed data
    """
    
    
    try:
        logger.info("Processing row data from PostgreSQL RDS")
        logger.info(f"Raw Event: {json.dumps(event, indent=2, default=str)}")
        
        # Convert row data to CSV
        csv_buffer = StringIO()
        
        if event:
            # Get column names from the event keys
            fieldnames = list(event.keys())
            
            # Create CSV writer
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            
            # Write header and row
            writer.writeheader()
            writer.writerow(event)
            
            # Get CSV content
            csv_content = csv_buffer.getvalue()
            
            logger.info(f"Generated CSV Content:\n{csv_content}")
            
            # Generate unique filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"rds_input.csv"
            s3_key = f"{S3_FOLDER}{filename}"
            
            # Upload to S3
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=csv_content.encode('utf-8'),
                ContentType='text/csv'
            )
            
            logger.info(f"CSV uploaded to s3://{S3_BUCKET}/{s3_key}")
            
            # Process the data
            processed_data = {
                "status": "success",
                "message": "Row data converted to CSV and uploaded to S3",
                "s3_bucket": S3_BUCKET,
                "s3_key": s3_key,
                "s3_uri": f"s3://{S3_BUCKET}/{s3_key}"
            }
            
            # Return response
            return {
                'statusCode': 200,
                'body': json.dumps(processed_data, default=str),
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        else:
            raise ValueError("No data received in event")
        
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'message': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }