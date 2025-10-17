
## Lambda Structure

```
lambda-functions/
├── RdsLambdaInvoke/
├── MovieMetadataEnricher/
├── DbtoS3/
├── GeneratePresignedUrl/
├── S3toDbupdate/
└── FetchfixedTableresults/
```

---

## 🔧 Lambda Functions

### 1. RdsLambdaInvoke
**Purpose:** Export PostgreSQL data to S3

**Process:**
- Receives PostgreSQL RDS row event
- Converts row data to CSV format
- Uploads CSV to S3 bucket
- Returns S3 file location

**Use Case:** Initial data extraction for enrichment pipeline

---

### 2. MovieMetadataEnricher
**Purpose:** AI-powered metadata enrichment

**Trigger:** S3 CSV upload event

**Process:**
- Detects new CSV uploads to S3
- Sends file path to AWS Bedrock Agent Runtime
- AI agent processes and enriches metadata
- Generates enhanced CSV with additional insights
- Stores enriched data in designated S3 path

**Innovation:** Leverages AWS Bedrock for intelligent content enhancement

---

### 3. DbtoS3
**Purpose:** Bidirectional data synchronization

**Process:**
- Upserts CSV records into PostgreSQL staging table
- Monitors staging table sync completion
- Merges updated CSV with existing knowledge base
- Uploads merged dataset to S3
- Returns processing status

**Key Feature:** Maintains data consistency across storage layers

---

### 4. GeneratePresignedUrl
**Purpose:** Secure file upload gateway

**Process:**
- Generates time-limited presigned S3 URL
- Configures upload destination (`users_input/` folder)
- Returns secure upload URL and file path

**Security:** No AWS credentials exposed to client applications

**Target Bucket:** `enrichment-agent-s3`

---

### 5. S3toDbupdate
**Purpose:** Import enriched data to database

**Process:**
- Reads enriched CSV from S3
- Validates and converts data types for PostgreSQL compatibility
- Upserts records using unique key (`series_id`)
- Handles conflicts intelligently

**Data Integrity:** Ensures type safety and conflict resolution

---

### 6. FetchfixedTableresults
**Purpose:** Data retrieval API endpoint

**Process:**
- Connects to PostgreSQL database
- Fetches all records from specified table
- Converts to JSON (handles Decimal types)
- Returns with CORS-enabled headers

**Integration:** Ready for web application consumption

---
