import os
import io
import csv
import boto3
import psycopg2
import pandas as pd

# --- Environment variables ---
DB_HOST = os.environ['DB_HOST']
DB_PORT = int(os.environ.get('DB_PORT', 5432))
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

S3_BUCKET = os.environ.get('S3_BUCKET')
S3_KEY_PREFIX = os.environ.get('S3_KEY_PREFIX')  # folder in bucket

TABLE_NAME = os.environ.get('TABLE_NAME')
UNIQUE_KEY = os.environ.get('UNIQUE_KEY')  # required for upsert

s3_client = boto3.client('s3')


def get_table_schema(conn, table_name: str, schema: str = "public"):
    """
    Fetch column names and PostgreSQL data types for a given table.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table_name.split('.')[-1]))
    schema_data = cursor.fetchall()
    cursor.close()
    df_schema = pd.DataFrame(schema_data, columns=["column_name", "data_type"])
    return df_schema


def enforce_dataframe_types(df: pd.DataFrame, df_schema: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the pandas DataFrame columns match PostgreSQL data types.
    Converts mismatched columns automatically.
    """
    type_map = {
        "text": "string",
        "character varying": "string",
        "varchar": "string",
        "integer": "Int64",     # nullable int
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
                print(f"Could not convert column {col} to {pandas_type} ({pg_type}): {e}")
        else:
            print(f"Column {col} missing in CSV, skipping conversion.")

    return df


def lambda_handler(event, context):
    # Find the single file in outputs/ folder
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_KEY_PREFIX)
        if 'Contents' not in response or len(response['Contents']) == 0:
            raise Exception(f"No files found in S3 {S3_KEY_PREFIX} folder")

        files = [obj for obj in response['Contents'] if not obj['Key'].endswith('/')]
        if len(files) == 0:
            raise Exception(f"No files found in S3 {S3_KEY_PREFIX} folder after filtering")

        # Pick the first (and only) file
        S3_KEY = files[0]['Key']
        print(f"Reading from s3://{S3_BUCKET}/{S3_KEY}")
    except Exception as e:
        print(f"Error finding file in S3: {e}")
        raise

    # Read CSV from S3
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        csv_content = s3_object['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        raise

    df = pd.read_csv(io.StringIO(csv_content))
    if df.empty:
        print("No rows in CSV file.")
        return {"status": "no_data"}

    print(f"Found {len(df)} rows and {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")

    # Connect to DB
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        # Enforce types to match PostgreSQL
        schema_df = get_table_schema(conn, TABLE_NAME)
        df = enforce_dataframe_types(df, schema_df)

        # Build dynamic SQL strings for upsert
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
        # Execute insert/update for each row
        for _, row in df.iterrows():
            if pd.isna(row.get(UNIQUE_KEY)):
                print(f"Skipping row without {UNIQUE_KEY}")
                continue
            cursor.execute(sql, row.to_dict())

        conn.commit()
        print(f"Upserted {len(df)} records into {TABLE_NAME}.")
        return {"status": "success", "records_processed": len(df)}

    except Exception as e:
        conn.rollback()
        print("Database error:", e)
        raise
    finally:
        conn.close()
