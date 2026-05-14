"""
URL feature extraction for malicious URL classification.
Extracts 33 lexical and structural features from raw URLs.
"""

import re
import math
import urllib.parse
from typing import Dict

PHISH_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update",
    "confirm", "banking", "paypal", "ebay", "amazon", "apple",
    "microsoft", "google", "facebook", "instagram", "support",
    "password", "credential", "wallet", "payment", "invoice",
}

HIGH_RISK_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".win",
    ".loan", ".click", ".download", ".link", ".online", ".site",
}

SUSPICIOUS_CHARS = set("@%~-_=")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def extract_features(url: str) -> Dict[str, float]:
    url = url.strip().lower()

    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    except Exception:
        return {f: 0.0 for f in _feature_names()}

    scheme = parsed.scheme or ""
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full = url

    hostname = netloc.split(":")[0]
    subdomains = hostname.split(".")
    tld = "." + subdomains[-1] if len(subdomains) > 1 else ""

    path_tokens = [t for t in re.split(r"[/\-_=&?]", path + query + fragment) if t]
    all_tokens = [t for t in re.split(r"[/\-_=&?.]", full) if t]

    features: Dict[str, float] = {}

    # --- Length features ---
    features["url_length"] = len(full)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)
    features["query_length"] = len(query)

    # --- Structural features ---
    features["subdomain_depth"] = max(0, len(subdomains) - 2)
    features["path_depth"] = path.count("/")
    features["num_query_params"] = query.count("&") + (1 if query else 0)
    features["has_ip_address"] = float(
        bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))
    )
    features["uses_https"] = float(scheme == "https")
    features["has_port"] = float(":" in netloc and not netloc.endswith(":443") and not netloc.endswith(":80"))

    # --- Character-level features ---
    features["url_entropy"] = _entropy(full)
    features["hostname_entropy"] = _entropy(hostname)
    features["digit_ratio"] = sum(c.isdigit() for c in full) / max(len(full), 1)
    features["special_char_ratio"] = sum(c in SUSPICIOUS_CHARS for c in full) / max(len(full), 1)
    features["hyphen_count"] = full.count("-")
    features["dot_count"] = full.count(".")
    features["at_sign"] = float("@" in full)
    features["double_slash"] = float("//" in path)
    features["hex_encoding"] = float("%" in full)

    # --- Token / keyword features ---
    token_set = {t.lower() for t in all_tokens}
    features["phish_keyword_count"] = sum(kw in token_set or kw in full for kw in PHISH_KEYWORDS)
    features["has_brand_name"] = float(
        any(brand in hostname for brand in ["paypal", "apple", "amazon", "google", "microsoft", "facebook"])
        and not any(hostname.endswith(f".{brand}.com") for brand in ["paypal", "apple", "amazon", "google", "microsoft", "facebook"])
    )
    features["num_digits_in_domain"] = sum(c.isdigit() for c in hostname)
    features["longest_token_length"] = max((len(t) for t in all_tokens), default=0)
    features["num_tokens"] = len(all_tokens)

    # --- TLD features ---
    features["high_risk_tld"] = float(tld in HIGH_RISK_TLDS)
    features["tld_length"] = len(tld)

    # --- Suspicious pattern flags ---
    features["url_shortener"] = float(
        any(s in hostname for s in ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "short.io"])
    )
    features["has_redirect_keyword"] = float(
        any(kw in path + query for kw in ["redirect", "url=", "link=", "goto=", "next=", "return="])
    )
    features["www_prefix"] = float(hostname.startswith("www."))
    features["consecutive_digits_in_domain"] = len(
        max(re.findall(r"\d+", hostname), key=len, default="")
    )
    features["path_has_exe_or_zip"] = float(
        any(path.endswith(ext) for ext in [".exe", ".zip", ".rar", ".apk", ".bat", ".sh", ".php"])
    )

    return features


def feature_vector(url: str):
    """Return features as an ordered list for model input."""
    import numpy as np
    feats = extract_features(url)
    return np.array([feats[k] for k in sorted(feats.keys())], dtype=float)


def feature_names():
    return sorted(extract_features("http://example.com").keys())


def _feature_names():
    return sorted(extract_features("http://example.com").keys())
