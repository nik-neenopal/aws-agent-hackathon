import os
import json
import psycopg2
from decimal import Decimal

DB_HOST = os.environ['DB_HOST']
DB_PORT = int(os.environ.get('DB_PORT'))
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
TABLE_NAME = os.environ.get('TABLE_NAME')


# Custom JSON serializer for Decimal
def json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)  # or str(obj) if you prefer
    raise TypeError


def lambda_handler(event, context):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Fetch all rows
        cursor.execute(f"SELECT * FROM {TABLE_NAME};")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Convert to list of dicts
        data = [dict(zip(columns, r)) for r in rows]

        return {
            "statusCode": 200,
            "body": json.dumps({"rows": data}, default=json_default),
                "headers": {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "OPTIONS,GET",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json"
    }
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "OPTIONS,GET",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json"
    }
        }

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
