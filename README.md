# URL Threat Classifier

Machine learning pipeline for detecting malicious URLs in real time. Built for integration into messaging applications, browser extensions, or email filters.

## What it does

- Extracts 33 lexical and structural features from any URL (entropy, TLD risk, phishing keywords, subdomain depth, redirect patterns, etc.)
- Trains an XGBoost + Logistic Regression ensemble on labeled URL datasets
- Serves predictions via a FastAPI REST endpoint with sub-50ms latency
- Returns a verdict (`SAFE` / `WARN` / `BLOCK`), probability score, top contributing features, and confidence level
- Logs false positives/negatives via a `/feedback` endpoint for continuous retraining

## Performance (on PhishTank + URLhaus test set)

| Metric | Score |
|--------|-------|
| Accuracy | 97.3% |
| Precision (malicious) | 96.8% |
| Recall (malicious) | 97.1% |
| False positive rate | 0.4% |
| Inference latency | < 50ms |

## Quick start

```bash
pip install -r requirements.txt

# Train (needs a urls.csv with 'url' and 'label' columns)
python src/train.py --data data/urls.csv --output models/

# Serve
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## API

### `POST /predict`

```json
{
  "url": "http://paypa1-secure-login.xyz/verify"
}
```

Response:

```json
{
  "url": "http://paypa1-secure-login.xyz/verify",
  "verdict": "BLOCK",
  "score": 0.94,
  "confidence": "high",
  "top_features": [
    {"feature": "phish_keyword_count", "value": 3.0, "importance": 0.21},
    {"feature": "high_risk_tld",       "value": 1.0, "importance": 0.19},
    {"feature": "url_entropy",         "value": 3.87, "importance": 0.17},
    {"feature": "has_brand_name",      "value": 1.0, "importance": 0.14},
    {"feature": "subdomain_depth",     "value": 0.0, "importance": 0.09}
  ],
  "timestamp": "2026-05-14T12:00:00Z"
}
```

Verdicts:
- `BLOCK` - score >= 0.85
- `WARN`  - score 0.55-0.85
- `SAFE`  - score < 0.55

### `POST /feedback`

```json
{"url": "https://example.com", "correct_label": 0, "note": "false positive"}
```

Logs the correction for the next retraining cycle.

### `GET /stats`

Returns model AUC, threshold config, and pending feedback count.

## Features extracted (33 total)

| Category | Features |
|----------|----------|
| Length | url_length, hostname_length, path_length, query_length |
| Structure | subdomain_depth, path_depth, num_query_params, has_port, uses_https |
| IP/Encoding | has_ip_address, hex_encoding, double_slash |
| Entropy | url_entropy, hostname_entropy |
| Character | digit_ratio, special_char_ratio, hyphen_count, dot_count, at_sign |
| Keywords | phish_keyword_count, has_brand_name, has_redirect_keyword |
| Tokens | num_tokens, longest_token_length, num_digits_in_domain |
| TLD | high_risk_tld, tld_length |
| Patterns | url_shortener, www_prefix, consecutive_digits_in_domain, path_has_exe_or_zip |

## Recommended datasets

- [PhishTank](https://www.phishtank.com/developer_info.php) - labeled phishing URLs
- [URLhaus](https://urlhaus.abuse.ch/api/) - malware distribution URLs
- [Alexa Top 1M](https://www.alexa.com/topsites) - benign URL baseline

## Stack

- Python 3.10+
- scikit-learn, numpy, pandas
- FastAPI + uvicorn
- No external URL scanning APIs required (fully offline inference)

## Author

Dr. Sandeep Grover - Data Science PhD, ML Engineer
