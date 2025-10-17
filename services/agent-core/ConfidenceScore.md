# Confidence Score Calculation and Acceptance Rules

## Overview

This document describes the complete confidence scoring system and acceptance criteria used in the data enrichment pipeline. The system uses a multi-factor approach combining source authority, model confidence, and evidence recall to determine whether extracted values should be accepted.

---

## Table of Contents

1. [Confidence Score Calculation](#confidence-score-calculation)
2. [Acceptance Criteria](#acceptance-criteria)
3. [Decision Flow](#decision-flow)
4. [Detailed Examples](#detailed-examples)
5. [Configuration Parameters](#configuration-parameters)
6. [Rejection Reasons](#rejection-reasons)

---

## Confidence Score Calculation

### Formula

The confidence score is calculated using a weighted combination of three factors:

```
computed_conf = (0.4 × model_conf) + (0.5 × base) + recall_factor
```

Where:
- `computed_conf`: Final confidence score (0.0 to 1.0)
- `model_conf`: LLM's self-assessed confidence (0.0 to 1.0)
- `base`: Source authority score (0.6 or 0.9)
- `recall_factor`: Evidence usage bonus (0.0 to 0.1)

### Weight Distribution

| Factor | Weight | Description |
|--------|--------|-------------|
| Source Authority | 50% | Whether the source is from a trusted/authoritative domain |
| Model Confidence | 40% | The LLM's confidence in its extraction |
| Recall Factor | 10% | How much of the available evidence was utilized |

### Base Score Determination

The base score depends on source authority:

```python
if domain_authoritative(source_hint):
    base = 0.9  # Authoritative source
else:
    base = 0.6  # Non-authoritative source
```

**Authoritative domains include:**
- Government domains (.gov, .gov.uk, .gov.au, .gov.ca)
- Educational institutions (.edu, .org, .ac.uk)
- Major entertainment databases (IMDB, Wikipedia, TMDB, Rotten Tomatoes)
- Streaming services (Netflix, Amazon Prime, Hulu, Disney+, HBO Max)
- Studios (Warner Bros, Universal, Paramount, Sony, MGM, Lionsgate, A24)
- News organizations (Reuters, AP, BBC, CNN, New York Times)
- Regional content sources (Bollywood Hungama, Dawn, Geo TV, Soompi, MyAnimeList)
- Academic sources (arXiv, JSTOR, Nature, Science, PubMed)
- Official documentation (docs., developer., api., support., help.)

Total: Over 200 authoritative domains recognized.

### Recall Factor Calculation

The recall factor measures how effectively the candidate value appears in the search results:

```python
recall_factor = min(0.1, (recall_used / recall_hits) × 0.1)
```

Where:
- `recall_used`: Number of snippet blocks that matched the candidate
- `recall_hits`: Total number of snippet blocks retrieved

---

## Acceptance Criteria

A candidate value is ACCEPTED if ALL of the following conditions are met:

### Required Conditions

1. **Verifier Approval**
   ```
   verdict == "YES"
   ```
   The verifier LLM confirms the value appears in snippets or context.

2. **Confidence Threshold**
   ```
   computed_conf >= MIN_CONFIDENCE
   ```
   Default: MIN_CONFIDENCE = 0.70

3. **Regex Validation**
   ```
   flexible_regex_match(candidate, target_regex)
   ```
   The value matches the expected format pattern for the field.

4. **Recall Requirement (with exceptions)**
   ```
   recall_used > 0
   ```
   OR any of these exceptions apply:
   - `computed_conf >= 0.85` (very high confidence)
   - `domain_authoritative(source_hint) == True` (trusted source)
   - `model_conf >= 0.9` (LLM very confident)

### Acceptance Logic

```python
passes_verdict = (verdict == "YES")
passes_confidence = (computed_conf >= MIN_CONFIDENCE)
passes_regex = flexible_regex_match(candidate, target_regex)
is_authoritative = domain_authoritative(source_hint)
allow_zero_recall = (computed_conf >= 0.85) or is_authoritative or (model_conf >= 0.9)

if passes_verdict and passes_confidence and passes_regex and (recall_used > 0 or allow_zero_recall):
    ACCEPT
else:
    REJECT
```

---

## Decision Flow

```
START
  |
  v
+---------------------------+
| Verifier Verdict == YES?  |
+---------------------------+
  |           |
  NO          YES
  |           |
  v           v
REJECT   +---------------------------+
         | computed_conf >= MIN?     |
         +---------------------------+
              |           |
              NO          YES
              |           |
              v           v
            REJECT   +---------------------------+
                     | Regex Match?              |
                     +---------------------------+
                          |           |
                          NO          YES
                          |           |
                          v           v
                        REJECT   +---------------------------+
                                 | recall_used > 0?          |
                                 +---------------------------+
                                      |           |
                                      NO          YES
                                      |           |
                                      v           v
                                 +----------+   ACCEPT
                                 | Check    |
                                 | Exceptions|
                                 +----------+
                                      |
                         +------------+------------+
                         |            |            |
                         v            v            v
                    computed_conf  auth     model_conf
                      >= 0.85?    source?    >= 0.9?
                         |            |            |
                         +------------+------------+
                                      |
                         YES          |          NO
                          |           |           |
                          v           v           v
                        ACCEPT      ACCEPT      REJECT
```

---

## Detailed Examples

### Example 1: High Confidence from Authoritative Source (Zero Recall)

**Scenario:** Extracting rating type "V, L, S" from IMDB

**Input Parameters:**
```
model_conf = 0.80
source_hint = "imdb.com"
recall_hits = 72
recall_used = 0
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = domain_authoritative("imdb.com") = True
base = 0.9
```

**Step 2: Calculate Confidence**
```
observed = 0.80
recall_factor = min(0.1, (0 / 72) × 0.1) = 0.0

computed_conf = (0.4 × 0.80) + (0.5 × 0.9) + 0.0
computed_conf = 0.32 + 0.45 + 0.0
computed_conf = 0.77
```

**Step 3: Check Acceptance Criteria**
```
passes_verdict = True (verdict == "YES")
passes_confidence = True (0.77 >= 0.70)
passes_regex = True
recall_used = 0

Check exceptions:
  computed_conf >= 0.85? False
  is_authoritative? True  <-- EXCEPTION GRANTED
  model_conf >= 0.9? False

allow_zero_recall = True
```

**Result:** ACCEPTED
**Reason:** Authoritative source (IMDB) exempts from recall requirement
**Log:** `"zero_recall_accepted": true`

---

### Example 2: Very High Confidence, Non-Authoritative Source

**Scenario:** Extracting director name from a blog

**Input Parameters:**
```
model_conf = 0.95
source_hint = "movieblog.com"
recall_hits = 20
recall_used = 0
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = domain_authoritative("movieblog.com") = False
base = 0.6
```

**Step 2: Calculate Confidence**
```
observed = 0.95
recall_factor = 0.0

computed_conf = (0.4 × 0.95) + (0.5 × 0.6) + 0.0
computed_conf = 0.38 + 0.30 + 0.0
computed_conf = 0.68
```

**Step 3: Check Acceptance Criteria**
```
passes_verdict = True
passes_confidence = False (0.68 < 0.70)  <-- FAILS HERE
```

**Result:** REJECTED
**Reason:** `"low_confidence(0.68<0.7)"`
**Note:** Despite high model confidence (0.95), the non-authoritative source pulls down the computed score below threshold.

---

### Example 3: Authoritative Source with Good Evidence

**Scenario:** Extracting movie genre from TheMovieDB

**Input Parameters:**
```
model_conf = 0.85
source_hint = "themoviedb.org"
recall_hits = 50
recall_used = 8
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = domain_authoritative("themoviedb.org") = True
base = 0.9
```

**Step 2: Calculate Confidence**
```
observed = 0.85
recall_factor = min(0.1, (8 / 50) × 0.1)
recall_factor = min(0.1, 0.016)
recall_factor = 0.016

computed_conf = (0.4 × 0.85) + (0.5 × 0.9) + 0.016
computed_conf = 0.34 + 0.45 + 0.016
computed_conf = 0.806
```

**Step 3: Check Acceptance Criteria**
```
passes_verdict = True
passes_confidence = True (0.806 >= 0.70)
passes_regex = True
recall_used = 8 > 0  <-- PASSES WITHOUT EXCEPTION
```

**Result:** ACCEPTED
**Reason:** All criteria met naturally, no exceptions needed
**Log:** `"confidence": 0.806, "recall_used": 8`

---

### Example 4: Low Confidence, No Recovery

**Scenario:** Extracting from unreliable source with poor evidence

**Input Parameters:**
```
model_conf = 0.60
source_hint = "randomsite.blogspot.com"
recall_hits = 30
recall_used = 1
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = False
base = 0.6
```

**Step 2: Calculate Confidence**
```
observed = 0.60
recall_factor = min(0.1, (1 / 30) × 0.1)
recall_factor = min(0.1, 0.003)
recall_factor = 0.003

computed_conf = (0.4 × 0.60) + (0.5 × 0.6) + 0.003
computed_conf = 0.24 + 0.30 + 0.003
computed_conf = 0.543
```

**Step 3: Check Acceptance Criteria**
```
passes_confidence = False (0.543 < 0.70)
```

**Result:** REJECTED
**Reason:** `"low_confidence(0.543<0.7)"`

---

### Example 5: Verifier Rejects Despite High Confidence

**Scenario:** High confidence but verifier cannot confirm in evidence

**Input Parameters:**
```
model_conf = 0.90
source_hint = "imdb.com"
recall_hits = 40
recall_used = 5
MIN_CONFIDENCE = 0.70
verdict = "NO"  <-- VERIFIER REJECTS
regex_match = True
```

**Step 1: Calculate Confidence**
```
base = 0.9
observed = 0.90
recall_factor = min(0.1, (5 / 40) × 0.1) = 0.0125

computed_conf = (0.4 × 0.90) + (0.5 × 0.9) + 0.0125
computed_conf = 0.36 + 0.45 + 0.0125
computed_conf = 0.8225
```

**Step 2: Check Acceptance Criteria**
```
passes_verdict = False (verdict == "NO")  <-- FAILS IMMEDIATELY
```

**Result:** REJECTED
**Reason:** `"verifier_rejected"`
**Note:** Verifier acts as final safety check. Even with high confidence, if the verifier cannot confirm the value in the evidence, it is rejected.

---

### Example 6: Regex Mismatch

**Scenario:** Value doesn't match expected format

**Input Parameters:**
```
model_conf = 0.85
source_hint = "imdb.com"
recall_hits = 30
recall_used = 5
MIN_CONFIDENCE = 0.70
verdict = "YES"
candidate = "ABC"
expected_regex = r"\d{4}"  (expects 4 digits, e.g., year)
regex_match = False
```

**Step 1: Calculate Confidence**
```
base = 0.9
observed = 0.85
recall_factor = min(0.1, (5 / 30) × 0.1) = 0.017

computed_conf = (0.4 × 0.85) + (0.5 × 0.9) + 0.017
computed_conf = 0.34 + 0.45 + 0.017
computed_conf = 0.807
```

**Step 2: Check Acceptance Criteria**
```
passes_verdict = True
passes_confidence = True (0.807 >= 0.70)
passes_regex = False  <-- FAILS HERE
```

**Result:** REJECTED
**Reason:** `"regex_mismatch"`
**Note:** Value "ABC" does not match expected format of 4 digits.

---

### Example 7: Very High Computed Confidence (Exception Path)

**Scenario:** Non-authoritative source but extremely high computed confidence

**Input Parameters:**
```
model_conf = 0.98
source_hint = "fandomwiki.com"
recall_hits = 25
recall_used = 0
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = domain_authoritative("fandomwiki.com") = True  (wiki pattern matches)
base = 0.9
```

**Step 2: Calculate Confidence**
```
observed = 0.98
recall_factor = 0.0

computed_conf = (0.4 × 0.98) + (0.5 × 0.9) + 0.0
computed_conf = 0.392 + 0.45 + 0.0
computed_conf = 0.842
```

**Step 3: Check Acceptance Criteria**
```
passes_verdict = True
passes_confidence = True (0.842 >= 0.70)
passes_regex = True
recall_used = 0

Check exceptions:
  computed_conf >= 0.85? False (0.842 < 0.85)
  is_authoritative? True  <-- EXCEPTION GRANTED (wiki pattern)
  model_conf >= 0.9? True  <-- ALSO EXCEPTION

allow_zero_recall = True
```

**Result:** ACCEPTED
**Reason:** Wiki sites are authoritative AND model confidence exceeds 0.9
**Log:** `"zero_recall_accepted": true`

---

### Example 8: Borderline Case with Perfect Evidence

**Scenario:** Lower confidence but perfect recall ratio

**Input Parameters:**
```
model_conf = 0.70
source_hint = "entertainment.com"
recall_hits = 10
recall_used = 10  (100% recall!)
MIN_CONFIDENCE = 0.70
verdict = "YES"
regex_match = True
```

**Step 1: Calculate Base Score**
```
is_authoritative = domain_authoritative("entertainment.com") = True
base = 0.9
```

**Step 2: Calculate Confidence**
```
observed = 0.70
recall_factor = min(0.1, (10 / 10) × 0.1)
recall_factor = min(0.1, 0.1)
recall_factor = 0.1  (maximum bonus!)

computed_conf = (0.4 × 0.70) + (0.5 × 0.9) + 0.1
computed_conf = 0.28 + 0.45 + 0.1
computed_conf = 0.83
```

**Step 3: Check Acceptance Criteria**
```
passes_verdict = True
passes_confidence = True (0.83 >= 0.70)
passes_regex = True
recall_used = 10 > 0
```

**Result:** ACCEPTED
**Reason:** Perfect evidence utilization provides 0.1 bonus, pushing confidence well above threshold
**Log:** `"confidence": 0.83, "recall_used": 10`

---

## Configuration Parameters

### Environment Variables

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| MIN_CONFIDENCE | 0.70 | 0.0-1.0 | Minimum confidence threshold for acceptance |
| MAX_SNIPPETS_TOTAL | 80 | 1-500 | Maximum search result snippets to collect |
| PER_QUERY_RESULTS | 10 | 1-50 | Results per search query |
| MAX_QUERIES | 12 | 1-50 | Maximum number of search queries to execute |

### Built-in Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Zero Recall Confidence Exception | 0.85 | Allow zero recall if computed confidence exceeds this |
| Model Confidence Exception | 0.9 | Allow zero recall if model confidence exceeds this |
| Authoritative Base Score | 0.9 | Base score for authoritative sources |
| Non-Authoritative Base Score | 0.6 | Base score for non-authoritative sources |
| Maximum Recall Factor | 0.1 | Cap on recall factor contribution |

### Weight Distribution (Fixed)

| Component | Weight |
|-----------|--------|
| Source Authority | 50% (0.5) |
| Model Confidence | 40% (0.4) |
| Recall Factor | 10% (0.1 max) |

---

## Rejection Reasons

The system provides detailed rejection reasons in logs:

| Reason | Description |
|--------|-------------|
| `verifier_rejected` | Verifier returned "NO" - value not confirmed in evidence |
| `low_confidence(X<Y)` | Computed confidence X below threshold Y |
| `regex_mismatch` | Value does not match expected format pattern |
| `zero_recall_not_allowed` | No evidence found and no exception granted |
| Combined reasons | Multiple issues, e.g., `"low_confidence(0.65<0.7),zero_recall_not_allowed"` |

### Example Log Entries

**Acceptance:**
```json
{
  "event": "accept",
  "field": "rating_type",
  "value": "V, L, S",
  "confidence": 0.77,
  "recall_used": 0,
  "zero_recall_accepted": true
}
```

**Rejection:**
```json
{
  "event": "reject",
  "field": "category",
  "reason": "low_confidence(0.65<0.7),zero_recall_not_allowed",
  "regex": ".+",
  "confidence": 0.65,
  "recall_used": 0,
  "model_conf": 0.75
}
```

---

## Summary Table

### Before vs After Recent Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Recall Requirement | Always > 0 | Flexible with exceptions |
| Token Filter | >= 4 characters | >= 2 characters |
| Token Overlap Threshold | 70% | 50% |
| Authoritative Domains | ~20 | ~200+ |
| Rejection Details | Generic message | Detailed reasons |
| Zero Recall + High Conf | Always rejected | Accepted with exceptions |
| Zero Recall + Auth Source | Always rejected | Accepted |

### Key Acceptance Paths

| Path | Conditions | Outcome |
|------|------------|---------|
| Standard | Verifier YES + Conf >= MIN + Regex + Recall > 0 | ACCEPT |
| High Confidence Exception | Verifier YES + Conf >= 0.85 + Regex | ACCEPT (zero recall allowed) |
| Authoritative Source | Verifier YES + Auth Source + Conf >= MIN + Regex | ACCEPT (zero recall allowed) |
| High Model Confidence | Verifier YES + Model >= 0.9 + Conf >= MIN + Regex | ACCEPT (zero recall allowed) |
| Fallback | All above fail | Attempt fallback extraction |

---

## Implementation References

### Code Locations

- **Confidence Calculation:** `compute_confidence()` at line 281-293
- **Acceptance Logic:** `enrich_field()` at line 948-974
- **Recall Calculation:** `intelligent_evidence_search()` at line 811-855
- **Domain Authority:** `domain_authoritative()` at line 204-410
- **Verifier Logic:** `build_verifier_prompt()` at line 741-761

### Related Functions

- `flexible_regex_match()`: Performs lenient regex validation
- `clean_dynamic()`: Type conversion and cleaning of accepted values
- `sanity_check_column_text()`: Pre-validation checks

---

## Best Practices

### For Maximum Acceptance Rate

1. **Use authoritative sources** when available (adds +0.15 to base score)
2. **Ensure good evidence retrieval** (aim for recall_used > 0)
3. **Craft precise search queries** (improves snippet relevance)
4. **Adjust MIN_CONFIDENCE** if needed (can lower to 0.65 for niche content)

### For Maximum Quality

1. **Keep MIN_CONFIDENCE at 0.70+** (ensures reliable extractions)
2. **Monitor verifier_rejected events** (indicates poor evidence quality)
3. **Review fallback_accept events** (may bypass normal validation)
4. **Check zero_recall_accepted logs** (ensure exceptions are justified)

---

## Change History

### Version 2.0 (Current)

- Added flexible recall requirement with three exception paths
- Expanded authoritative domain list to 200+ domains
- Improved recall detection with multi-pass strategy
- Added detailed rejection reason logging
- Reduced token filter from 4 to 2 characters
- Reduced overlap threshold from 70% to 50%

### Version 1.0 (Original)

- Basic confidence calculation with fixed weights
- Strict recall requirement (always > 0)
- Limited authoritative domain list (~20)
- Generic rejection messages

---

End of Document
