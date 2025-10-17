import os
import io
import time
import json
import boto3
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text

# --- Environment variables ---
DB_HOST = os.environ['DB_HOST']
DB_PORT = int(os.environ.get('DB_PORT'))
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

S3_BUCKET = os.environ['S3_BUCKET']
S3_KEY_PREFIX = os.environ.get('S3_KEY_PREFIX')

TABLE_NAME = os.environ.get('TABLE_NAME')
UNIQUE_KEY = os.environ.get('UNIQUE_KEY')
STAGING_TABLE = os.environ.get('STAGING_TABLE')

POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', 10))
POLL_TIMEOUT = int(os.environ.get('POLL_TIMEOUT', 300))

KNOWLEDGE_KEY = os.environ.get("KNOWLEDGE_KEY", "knowledge_base/media_series.csv")

s3_client = boto3.client('s3')


# ---------- Helper Functions ----------

def get_table_schema(conn, table_name: str, schema: str = "public"):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table_name.split('.')[-1]))
    schema_data = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(schema_data, columns=["column_name", "data_type"])


def enforce_dataframe_types(df: pd.DataFrame, df_schema: pd.DataFrame) -> pd.DataFrame:
    type_map = {
        "text": "string",
        "character varying": "string",
        "varchar": "string",
        "integer": "Int64",
        "bigint": "Int64",
        "numeric": "float64",
        "double precision": "float64",
        "boolean": "boolean",
        "timestamp without time zone": "datetime64[ns]",
        "timestamp with time zone": "datetime64[ns]",
        "date": "datetime64[ns]"
    }
    for _, row in df_schema.iterrows():
        col = row["column_name"]
        pg_type = row["data_type"]
        pandas_type = type_map.get(pg_type, "string")
        if col in df.columns:
            try:
                df[col] = df[col].astype(pandas_type)
            except Exception as e:
                print(f"Could not convert column {col}: {e}")
    return df


def get_table_row_count(conn, table_name: str) -> int:
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def wait_for_staging_sync(conn, inserted_count: int):
    print(f"Waiting for {STAGING_TABLE} to match row count of {TABLE_NAME}...")
    start_time = time.time()
    while True:
        orig_count = get_table_row_count(conn, TABLE_NAME)
        staging_count = get_table_row_count(conn, STAGING_TABLE)
        print(f"original_media_series={orig_count}, media_series_staging_in={staging_count}")

        if staging_count >= orig_count:
            print("Row counts match — SP copy complete.")
            return True

        if time.time() - start_time > POLL_TIMEOUT:
            print("Timeout reached while waiting for staging sync.")
            return False

        print(f"Sleeping {POLL_INTERVAL}s before next check...")
        time.sleep(POLL_INTERVAL)


def upload_to_s3(bucket, key, content, content_type="text/csv"):
    s3_client.put_object(Bucket=bucket, Key=key, Body=content, ContentType=content_type)
    print(f"Uploaded to s3://{bucket}/{key}")


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

def merge_knowledge_and_input(bucket_name, knowledge_key, input_key, output_prefix):
    """
    Union (append) the knowledge base CSV (media_series.csv)
    with the uploaded user input CSV, and store the merged file in S3.
    """
    print("Starting union merge of knowledge base and user input files")

    # --- Read knowledge base file from S3 ---
    knowledge_obj = s3_client.get_object(Bucket=bucket_name, Key=knowledge_key)
    knowledge_df = pd.read_csv(io.BytesIO(knowledge_obj['Body'].read()))
    print(f"Loaded knowledge base file with {len(knowledge_df)} rows")

    # --- Read uploaded input file from S3 ---
    input_obj = s3_client.get_object(Bucket=bucket_name, Key=input_key)
    input_df = pd.read_csv(io.BytesIO(input_obj['Body'].read()))
    print(f"Loaded user input file with {len(input_df)} rows")

    # --- Union (append) both dataframes ---
    merged_df = pd.concat([knowledge_df, input_df], ignore_index=True)
    print(f"Merged dataframe has {len(merged_df)} rows (union of both)")

    # --- Write merged file to S3 ---
    merged_csv_buffer = io.StringIO()
    merged_df.to_csv(merged_csv_buffer, index=False)

    # Store final output as inputs/rds_input.csv
    output_key = f"{output_prefix.rstrip('/')}/rds_input.csv"
    upload_to_s3(bucket_name, output_key, merged_csv_buffer.getvalue())

    print(f"Uploaded merged file to s3://{bucket_name}/{output_key}")

    return {
        "status": "merged",
        "rows_knowledge": len(knowledge_df),
        "rows_input": len(input_df),
        "rows_merged": len(merged_df),
        "output_key": output_key
    }


# ---------- Main Lambda ----------

def lambda_handler(event, context):
    try:
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print(f"Triggered by S3 object: s3://{bucket_name}/{key}")
    except Exception as e:
        print("Error parsing S3 event:", e)
        raise

    s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)
    csv_content = s3_object['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_content))

    if df.empty:
        print("CSV file is empty.")
        return {"status": "no_data"}

    print(f"Loaded {len(df)} rows from CSV.")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        schema_df = get_table_schema(conn, TABLE_NAME)
        df = enforce_dataframe_types(df, schema_df)

        columns = df.columns.tolist()
        cols = ', '.join(columns)
        placeholders = ', '.join([f'%({c})s' for c in columns])
        update_clause = ', '.join([f"{c} = EXCLUDED.{c}" for c in columns if c != UNIQUE_KEY])

        sql = f"""
            INSERT INTO {TABLE_NAME} ({cols})
            VALUES ({placeholders})
            ON CONFLICT ({UNIQUE_KEY})
            DO UPDATE SET {update_clause};
        """

        cursor = conn.cursor()
        for _, row in df.iterrows():
            if pd.isna(row.get(UNIQUE_KEY)):
                print(f"Skipping row without {UNIQUE_KEY}")
                continue
            cursor.execute(sql, row.to_dict())
        conn.commit()
        inserted_count = len(df)
        print(f"Upserted {inserted_count} records into {TABLE_NAME}.")

        # Wait until staging sync
        wait_for_staging_sync(conn, inserted_count)

        # --- 🔁 Once while loop is done, trigger fetch/upload/mark workflow ---
        batch_result = merge_knowledge_and_input(
            bucket_name=S3_BUCKET,
            knowledge_key=KNOWLEDGE_KEY,
            input_key=key,  # the uploaded file that triggered Lambda
            output_prefix=S3_KEY_PREFIX  # e.g. 'inputs/'
        )

        return {
            "status": "success",
            "records_processed": inserted_count,
            "batch_result": batch_result
        }

    except Exception as e:
        conn.rollback()
        print("Database error:", e)
        raise
    finally:
        conn.close()
