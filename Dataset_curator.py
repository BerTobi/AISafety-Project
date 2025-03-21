#!/usr/bin/env python3
"""
Simple Text File Curator with Enhanced Quality Filters

A straightforward script that processes a single text file to create a clean
version suitable for LLM training data with multiple quality filters.

Usage:
    python simple_curator.py input.txt output.txt
"""

import sys
import re
import hashlib
import string
from collections import defaultdict

def clean_text(text):
    """Basic text cleaning and normalization."""
    # Replace carriage returns and normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace multiple newlines with double newline (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Basic HTML tag removal
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

def apply_quality_filters(paragraphs):
    """Apply various quality filters to paragraphs."""
    filtered_paragraphs = []
    rejected_counts = {
        "too_short": 0,
        "too_long": 0,
        "low_alpha_ratio": 0,
        "high_symbol_ratio": 0,
        "high_uppercase_ratio": 0,
        "repetitive_chars": 0,
        "low_info_density": 0,
        "no_sentence_structure": 0,
        "suspicious_patterns": 0
    }
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
        
        # 1. Length filters
        if len(paragraph) < 50:
            rejected_counts["too_short"] += 1
            continue
            
        if len(paragraph) > 10000:  # Limit very long paragraphs
            rejected_counts["too_long"] += 1
            continue
        
        # 2. Character composition filters
        char_count = len(paragraph)
        alpha_count = sum(c.isalpha() for c in paragraph)
        digit_count = sum(c.isdigit() for c in paragraph)
        symbol_count = sum(c in string.punctuation for c in paragraph)
        uppercase_count = sum(c.isupper() for c in paragraph if c.isalpha())
        
        # Skip if alphanumeric ratio is too low
        if (alpha_count + digit_count) / max(1, char_count) < 0.5:
            rejected_counts["low_alpha_ratio"] += 1
            continue
            
        # Skip if symbol ratio is too high
        if symbol_count / max(1, char_count) > 0.3:
            rejected_counts["high_symbol_ratio"] += 1
            continue
            
        # Skip if too much uppercase (shouting)
        if uppercase_count / max(1, alpha_count) > 0.6:
            rejected_counts["high_uppercase_ratio"] += 1
            continue
        
        # 3. Repetition check - detect repetitive character patterns
        if detect_char_repetition(paragraph):
            rejected_counts["repetitive_chars"] += 1
            continue
        
        # 4. Information density check
        if information_density(paragraph) < 0.4:
            rejected_counts["low_info_density"] += 1
            continue
        
        # 5. Check for sentence structure (at least one period, question mark, or exclamation point)
        if not re.search(r'[.!?]', paragraph):
            rejected_counts["no_sentence_structure"] += 1
            continue
            
        # 6. Check for suspicious patterns (URLs lists, code fragments, etc.)
        if has_suspicious_patterns(paragraph):
            rejected_counts["suspicious_patterns"] += 1
            continue
        
        # Paragraph passed all filters
        filtered_paragraphs.append(paragraph)
    
    return filtered_paragraphs, rejected_counts

def detect_char_repetition(text):
    """Detect overly repetitive character patterns."""
    # Check for repeating characters (e.g., "aaaaa")
    if re.search(r'(.)\1{7,}', text):
        return True
        
    # Check for repeating character sequences (e.g., "abcabcabc")
    for seq_len in range(2, 10):
        # Only check sequences up to 1/3 of the text length
        if seq_len > len(text) / 3:
            continue
            
        for i in range(len(text) - seq_len):
            pattern = text[i:i+seq_len]
            # Look for the pattern repeated at least 3 times
            if text.count(pattern) >= 3 and pattern.strip():
                # Verify it's not just common words by checking ratio of pattern to text
                if len(pattern) * text.count(pattern) > len(text) * 0.3:
                    return True
    
    return False

def information_density(text):
    """Calculate a simple information density score."""
    # 1. Unique words ratio
    words = text.lower().split()
    if not words:
        return 0
        
    unique_words = len(set(words))
    word_diversity = unique_words / len(words)
    
    # 2. Word length distribution (longer words often carry more information)
    avg_word_length = sum(len(w) for w in words) / max(1, len(words))
    length_factor = min(1.0, avg_word_length / 8.0)  # Normalize with cap at 1.0
    
    # 3. Sentence complexity (approximated by sentence length variation)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        sentence_complexity = 0.5  # Neutral for short texts
    else:
        sent_lengths = [len(s) for s in sentences]
        avg_sent_length = sum(sent_lengths) / len(sentences)
        # Standard deviation approximation
        variance = sum((l - avg_sent_length) ** 2 for l in sent_lengths) / len(sentences)
        complexity = min(1.0, (variance ** 0.5) / 50.0)  # Cap at 1.0
        sentence_complexity = 0.5 + (complexity * 0.5)  # Scale from 0.5 to 1.0
    
    # Combine factors with weights
    density = (word_diversity * 0.6) + (length_factor * 0.2) + (sentence_complexity * 0.2)
    return density

def has_suspicious_patterns(text):
    """Check for patterns that might indicate low-quality content."""
    # Check for excessive URLs
    url_count = len(re.findall(r'https?://\S+', text))
    if url_count > 2 or (url_count > 0 and url_count * 20 > len(text)):
        return True
    
    # Check for code-like content
    code_indicators = ['import ', 'def ', 'class ', 'function(', 'var ', 'const ', 'let ', '{ ']
    code_matches = sum(1 for indicator in code_indicators if indicator in text)
    if code_matches >= 2:
        return True
    
    # Check for list-like content (bullet points, numbering)
    list_pattern = re.findall(r'(\n\s*[-•*]\s+|\n\s*\d+\.\s+)', text)
    if len(list_pattern) > 3:
        return True
    
    # Check for markdown/formatting characters
    if text.count('```') >= 2 or text.count('###') > 3:
        return True
    
    return False

def deduplicate_paragraphs(paragraphs):
    """Remove duplicate paragraphs based on their content hash."""
    seen_hashes = set()
    unique_paragraphs = []
    
    for paragraph in paragraphs:
        # Create a simple hash of the normalized paragraph
        p_normalized = re.sub(r'\s+', ' ', paragraph.lower().strip())
        p_hash = hashlib.md5(p_normalized.encode('utf-8')).hexdigest()
        
        if p_hash not in seen_hashes:
            seen_hashes.add(p_hash)
            unique_paragraphs.append(paragraph)
    
    return unique_paragraphs

def detect_near_duplicates(paragraphs, threshold=0.7, shingle_size=3):
    """Filter out near-duplicate paragraphs using text shingling."""
    paragraph_shingles = []
    global_shingles = defaultdict(int)
    filtered_paragraphs = []
    
    # Generate shingles for each paragraph
    for paragraph in paragraphs:
        words = paragraph.lower().split()
        if len(words) < shingle_size:
            # Too short to generate shingles, keep it
            filtered_paragraphs.append(paragraph)
            continue
            
        # Generate shingles (word n-grams)
        shingles = set()
        for i in range(len(words) - shingle_size + 1):
            shingle = ' '.join(words[i:i+shingle_size])
            shingles.add(shingle)
        
        paragraph_shingles.append((paragraph, shingles))
    
    # First pass: count all shingles
    for _, shingles in paragraph_shingles:
        for shingle in shingles:
            global_shingles[shingle] += 1
    
    # Second pass: check for near-duplicates
    for paragraph, shingles in paragraph_shingles:
        # If paragraph has no shingles, keep it
        if not shingles:
            filtered_paragraphs.append(paragraph)
            continue
        
        # Count how many shingles appear in other paragraphs
        duplicate_count = sum(1 for shingle in shingles if global_shingles[shingle] > 1)
        duplicate_ratio = duplicate_count / len(shingles)
        
        # If duplicate ratio is below threshold, keep the paragraph
        if duplicate_ratio < threshold:
            filtered_paragraphs.append(paragraph)
    
    return filtered_paragraphs

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_curator.py input.txt output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Print start message
    print(f"Processing {input_file}...")
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Track statistics
    stats = {
        "original_chars": len(text),
        "original_paragraphs": 0,
        "paragraphs_after_quality_filter": 0,
        "paragraphs_after_dedup": 0,
        "final_paragraphs": 0,
        "final_chars": 0,
        "rejected_reasons": {}
    }
    
    # Clean the text
    text = clean_text(text)
    
    # Split into paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    stats["original_paragraphs"] = len(paragraphs)
    
    # Apply quality filters
    paragraphs, rejected_counts = apply_quality_filters(paragraphs)
    stats["paragraphs_after_quality_filter"] = len(paragraphs)
    stats["rejected_reasons"] = rejected_counts
    
    # Remove exact duplicates
    paragraphs = deduplicate_paragraphs(paragraphs)
    stats["paragraphs_after_dedup"] = len(paragraphs)
    
    # Remove near-duplicates
    paragraphs = detect_near_duplicates(paragraphs)
    stats["final_paragraphs"] = len(paragraphs)
    
    # Reassemble text
    cleaned_text = '\n\n'.join(paragraphs)
    stats["final_chars"] = len(cleaned_text)
    
    # Write the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Print summary
    print("\n=== Curation Complete ===")
    print(f"Original length: {stats['original_chars']:,} characters")
    print(f"Original paragraphs: {stats['original_paragraphs']}")
    print("\nQuality filtering:")
    for reason, count in stats["rejected_reasons"].items():
        if count > 0:
            print(f"  - {reason.replace('_', ' ').title()}: {count} paragraphs")
    print(f"Paragraphs after quality filters: {stats['paragraphs_after_quality_filter']}")
    print(f"Paragraphs after deduplication: {stats['paragraphs_after_dedup']}")
    print(f"Final paragraphs: {stats['final_paragraphs']}")
    print(f"Final length: {stats['final_chars']:,} characters")
    print(f"Characters preserved: {stats['final_chars']/max(1, stats['original_chars']):.2%}")
    print(f"Output written to: {output_file}")
    print("========================")

if __name__ == "__main__":
    main()