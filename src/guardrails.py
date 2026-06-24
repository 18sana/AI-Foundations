import re

# Regex patterns for common PII
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# Jailbreak / Injection Keywords
INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore prior instructions",
    "system prompt",
    "reveal your instructions",
    "reveal system instructions",
    "bypass safety guidelines",
    "bypass instructions",
    "act as a developer with no restrictions",
    "unlocked mode"
]

# Restricted Topics (Content Moderation)
TOXIC_KEYWORDS = [
    "bomb creation",
    "malware writing",
    "illegal drug synthesis",
    "hate speech"
]

def redact_pii(text: str) -> str:
    """
    Scans the text and replaces emails, phone numbers, and SSNs with redacted placeholders.
    """
    redacted = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)
    redacted = SSN_REGEX.sub("[REDACTED_SSN]", redacted)
    return redacted

def is_injection_attempt(text: str) -> bool:
    """
    Checks if the user text contains common prompt injection phrases.
    """
    text_lower = text.lower()
    for kw in INJECTION_KEYWORDS:
        if kw in text_lower:
            return True
    return False

def violates_moderation(text: str) -> bool:
    """
    Checks if the input violates basic content moderation guidelines.
    """
    text_lower = text.lower()
    for kw in TOXIC_KEYWORDS:
        if kw in text_lower:
            return True
    return False
