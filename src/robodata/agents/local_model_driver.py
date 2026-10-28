"""Evidence redaction used by the local model driver; no legacy pilot caps."""
import hashlib

def redact(value):
    """Keep audit text; replace inline image bytes with their deterministic hash.

    The simulator retains the exact public_input_NN.json including images. This
    ledger avoids duplicating those bytes across every historical request.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in {"api_key", "authorization", "access_token", "secret", "password"}:
                result[key] = "[REDACTED]"
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"inline_image_redacted": True,
                "sha256_data_url": hashlib.sha256(value.encode()).hexdigest(),
                "data_url_characters": len(value), "mime": value.split(";", 1)[0][5:]}
    return value

