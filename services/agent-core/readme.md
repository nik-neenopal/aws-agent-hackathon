# AgentCore Data Enrichment 

## Overview

This enrichment pipeline leverages AWS Bedrock AgentCore to intelligently fill missing CSV data through automated web search and LLM-powered extraction.

The system incorporates a robust confidence scoring mechanism to ensure data quality and validation of all extracted values.
By combining multiple data sources and AI-powered analysis, the pipeline automatically identifies gaps in datasets and enriches them with accurate, validated information.

The confidence scoring system evaluates each enrichment based on multiple factors including model certainty, source authority, and recall validation, ensuring only high-quality data enters the production environment.

---

## Core Components

### 1. AgentCore Setup
1. AgentCore Integration: AgentCore is already integrated in the code. Look for these lines:
```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict) -> dict:
    # Entire enrichment pipeline runs here
    # Returns enriched DataFrame
```

**What happens:**

- Standard Python script becomes AgentCore-ready with just the decorator
- invoke() receives payload from Bedrock Agent Runtime
- Expected payload: {"csv_s3_path": "s3://...", "output_s3_path": "s3://...", "logs_s3_path": "s3://..."}
- No additional setup needed - it just works


---

## 2. Confidence Scoring System

### Formula

```
confidence = (0.4 × model_conf) + (0.5 × base_score) + recall_factor
```

| Component | Weight | Range |
|-----------|--------|-------|
| **Model Confidence** | 40% | LLM's self-assessed confidence (0-1) |
| **Source Authority** | 50% | 0.9 (authoritative) or 0.6 (non-auth) |
| **Recall Factor** | 10% | Evidence usage: `min(0.1, (recall_used/recall_hits) × 0.1)` |

### Acceptance Criteria

A value is **ACCEPTED** if ALL pass:

1.  **Verifier says YES** - LLM confirms value in evidence
2.  **Confidence ≥ 0.70** - Computed score meets threshold
3.  **Regex match** - Value format matches expected pattern
4.  **recall_used > 0** or exception granted
5.  Exception Granted Condition:
       - High confidence (≥0.85)
       - Authoritative source  
       - Model confidence ≥0.9

### Example

```python
# Authoritative source with zero recall
model_conf = 0.80
source = "imdb.com"  # authoritative
recall_used = 0
recall_hits = 0

base = 0.9  # authoritative
confidence = (0.4 × 0.80) + (0.5 × 0.9) + 0.0 = 0.77

# ACCEPTED: confidence ≥ 0.70 AND authoritative exempts recall_used
# Note: recall_used = 0 but exception granted due to authoritative source

```
<sub>For more detail on Confidence score please visit [Confidence Score Documentation](ConfidenceScore.md)</sub>
---
## 3. Key Functions

### Enrichment Pipeline

```python
enrich_field(row, field_name, df, min_confidence=0.7)
```
- Builds context from row data
- Generates search queries
- Extracts candidate value
- Verifies with confidence scoring
- Returns accepted value or None

### Confidence Calculation

```python
compute_confidence(model_conf, source_hint, recall_hits, recall_used)
```
- `model_conf`: LLM's confidence (0-1)
- `source_hint`: Domain name (e.g., "imdb.com")
- `recall_hits`: Total search snippets
- `recall_used`: Snippets containing candidate

### Authority Check

```python
domain_authoritative(domain: str) -> bool
```
Returns `True` for 200+ trusted domains:
- `.gov`, `.edu`, `.org`
- IMDB, Wikipedia, TMDB, Rotten Tomatoes
- Netflix, Amazon Prime, Disney+
- Reuters, BBC, New York Times
- Academic: arXiv, JSTOR, PubMed

---

## 4. Enrichment Workflow

```
1. Load CSV from S3
2. Merge with knowledge base
3. For each pass (max 3):
   ├─ Find missing fields
   ├─ For each missing cell:
   │  ├─ Build context
   │  ├─ Generate queries
   │  ├─ Search web (Google/Bing/SerpAPI)
   │  ├─ Extract candidate
   │  ├─ Verify & score confidence
   │  └─ Accept or reject
   └─ Update DataFrame
4. Add metadata columns:
   ├─ {field}_missing_value (0/1 flag)
   ├─ {field}_conf (confidence score)
   └─ {field}_manual_review (1 if needs review)
5. Save to S3
```

---

## 5. Configuration

### Environment Variables

```bash
MIN_CONFIDENCE=0.70          # Acceptance threshold
MAX_WORKERS=6                # Parallel enrichment tasks
LLM_MAX_CONCURRENCY=3        # Concurrent LLM calls
MAX_SNIPPETS_TOTAL=80        # Max search results
ENABLE_FETCH_HTML=true       # Fetch full page content
```

### AgentCore-Specific

```python
# Payload structure
{
  "csv_s3_path": "s3://bucket/input.csv",
  "output_s3_path": "s3://bucket/output.csv",
  "logs_s3_path": "s3://bucket/logs.json"
}
```

---

## 6. Output Format

### Enriched CSV Columns

Original columns + metadata:

```
series_id, title, genre, ...
title_missing_value        # 1 if was missing
title_conf                 # 0.77 (confidence score)
title_manual_review        # 1 if accept-true/fallback
genre_missing_value
genre_conf
genre_manual_review
```

### Confidence Records

```python
CONFIDENCE_RECORDS = [
  {
    "series_id": "12345",
    "field_name": "rating_type",
    "status": "accept",           # or "accept-true", "fallback_accept", "reject"
    "value": "TV-14",
    "confidence": 0.77
  }
]
```

**Status Types:**
- `accept`: Normal acceptance (recall_used > 0)
- `accept-true`: Accepted with zero recall_used (exception granted)
- `fallback_accept`: From row context without web search
- `reject`: Failed acceptance criteria

---

## 7. Manual Review Flags

Values flagged for review when:

1. **accept-true**: High confidence but zero evidence recall_used
2. **fallback_accept**: Extracted from row context only (no web search)

Check these records before production use.

---

## 8. Running the Agent

### Local Development

```bash
python enrichment_agent.py
```

### Production (AgentCore)

```bash
# Deploy as AgentCore app
bedrock-agentcore deploy

# Invoke via AWS SDK
import boto3
client = boto3.client('bedrock-agent-runtime')

response = client.invoke_agent(
    agentId='your-agent-id',
    agentAliasId='your-alias-id',
    inputText=json.dumps({
        "csv_s3_path": "s3://...",
        "output_s3_path": "s3://..."
    })
)
```

---

## 9. Troubleshooting

### Low Acceptance Rate
- Lower `MIN_CONFIDENCE` to 0.65 (be cautious)
- Increase `MAX_SNIPPETS_TOTAL` to 120
- Enable `INCLUDE_DOMAIN_QUERIES` for site-specific searches

### High Manual Review Count
- Tune confidence thresholds
- Review authoritative domain list
- Check `fallback_accept` logic

### Timeout Errors
- Reduce `MAX_WORKERS` or `LLM_MAX_CONCURRENCY`
- Increase `LLM_TIMEOUT_SEC`
- Process in smaller batches (`PASS_BATCH_SIZE`)

---

## Quick Reference

| Metric | Good | Review | Poor |
|--------|------|--------|------|
| Confidence | ≥0.75 | 0.65-0.75 | <0.65 |
| Recall Used | >0 | 0 (with exception) | 0 (rejected) |
| Manual Review | 0 | 1 | N/A |

**Remember:** Confidence ≥ 0.70 is the default acceptance threshold. Values with `manual_review=1` should be spot-checked before use.
