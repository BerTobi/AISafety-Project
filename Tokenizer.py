import re

def separate_tokens(text):
    """
    Separates words from punctuation by adding spaces between them,
    except for apostrophes which remain attached to words.
    
    Args:
        text (str): The input text to process
        
    Returns:
        str: Text with spaces between words and punctuation (except apostrophes)
    """
    # Create a string of punctuation characters excluding apostrophes
    punct_chars = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # All punctuation except apostrophes
    
    # Replace each punctuation character with a space + that character + space
    result = text
    for char in punct_chars:
        result = result.replace(char, f' {char} ')
    
    # Remove extra spaces that might have been created
    cleaned_text = ' '.join(result.split())
    
    return cleaned_text

def tokenize_text(text):
    """
    Tokenizes text by separating words and punctuation,
    and returns a list of tokens.
    
    Args:
        text (str): The input text to process
        
    Returns:
        list: A list of tokens (words and punctuation)
    """
    spaced_text = separate_tokens(text)
    tokens = spaced_text.split()
    return tokens

def split_into_sentences(text):
    """
    Splits text into individual sentences.
    
    Args:
        text (str): The input text to process
        
    Returns:
        list: A list of sentences
    """
    # This regex splits on sentence-ending punctuation followed by whitespace
    # and optionally quotation marks
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z0-9]|$)|(?<=[.!?])"?\s+(?=[A-Z0-9]|$)'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def split_into_paragraphs(text):
    """
    Splits text into individual paragraphs.
    
    Args:
        text (str): The input text to process
        
    Returns:
        list: A list of paragraphs
    """
    # Split on one or more consecutive newline characters
    paragraphs = re.split(r'\n\s*\n', text)
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

# Example usage
if __name__ == "__main__":
    sample_text = "I’ve part again out, you’ll"
    
    # Separate words from punctuation
    processed_text = separate_tokens(sample_text)
    print("Processed text:")
    print(processed_text)
    
    # Get the list of tokens
    tokens = tokenize_text(sample_text)
    print("\nTokens:")
    print(tokens)
    
    # Show token count
    print(f"\nTotal tokens: {len(tokens)}")