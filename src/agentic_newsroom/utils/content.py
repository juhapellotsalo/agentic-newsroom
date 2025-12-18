import re

def count_words(text: str) -> int:
    """
    Count the number of words in a markdown string, excluding special characters and urls.
    """
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove header markers # or ## or ###
    text = re.sub(r'#+', '', text)
    # Remove unordered list checkmarks
    text = re.sub(r'[-*]\s', '', text)
    # Remove special chars (keep only alphanumeric and whitespace and hyphens)
    text = re.sub(r'[^\w\s-]', '', text)
    return len(text.split())


if __name__ == "__main__":
    
    test_text = """
    This is a test paragraph
    """

    count = count_words(test_text)
    print(f"Word count: {count}")