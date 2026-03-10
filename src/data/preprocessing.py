import re

def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\S+', '', text)

def remove_mentions(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

def remove_hashtag_symbol(text: str) -> str:
    return re.sub(r'#', '', text)

def clean_text(text: str) -> str:
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = normalize_whitespace(text)
    return text.strip()

def is_low_information(text: str, min_tokens: int = 3) -> bool:
    tokens = text.split()
    return len(tokens) < min_tokens