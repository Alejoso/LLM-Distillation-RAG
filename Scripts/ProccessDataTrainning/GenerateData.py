import os
import re
import json
import spacy
import unicodedata
import argparse
from pathlib import Path
from CleaningPatterns import PATTERNS, apply_patterns

# Load Spanish language model for sentence segmentation
nlp = spacy.load("es_core_news_sm")


# functions for extract only the content after "CONTENIDO:"
def extract_content(text):
    match = re.search(r"CONTENIDO:(.*)", text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()

# Cleaning patterns defined in CleaningPatterns.py
# function to apply all cleaning patterns to the text, with special handling for special cases.
def clean_text(text):
    # Apply cleaning patterns
    text = apply_patterns(text)
    
    # Remove lines that only contain numbers or special characters
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # Keep if it has at least one real word
        if re.search(r'[a-záéíóúñ]', line, re.IGNORECASE):
            filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    return text.strip()

# function to normalize text: Unicode normalization, whitespace cleanup, punctuation spacing.
def normalize_text(text):
    # Normalize Unicode (decompose and recompose for badly encoded accents)
    text = unicodedata.normalize('NFKD', text)
    text = unicodedata.normalize('NFC', text)
    
    # Replace multiple line breaks with a single space
    text = re.sub(r'\n+', ' ', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Clean spaces around punctuation
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    return text

# function to split text into sentences using spaCy, and filter by character length.
def split_into_sentences(text, min_chars=100, max_chars=200):
    # Use spaCy to segment sentences and filter by length
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        sentence = sent.text.strip()
        length = len(sentence)

        if min_chars <= length <= max_chars:
            sentences.append(sentence)

    return sentences

# Function to process all text files in a folder, applying the full pipeline and collecting sentences until a limit is reached.
def process_folder(folder, min_chars=100, max_chars=200, max_sentences=5000):
    all_sentences = []
    processed_files = 0
    failed_files = 0

    for file in os.listdir(folder):
        # Stop if we already reached the limit
        if len(all_sentences) >= max_sentences:
            print(f"Limit of {max_sentences} sentences reached. Stopping processing.")
            break
            
        if file.endswith(".txt"):
            path = os.path.join(folder, file)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Extract content
                content = extract_content(text)
                if not content:
                    failed_files += 1
                    continue

                # Basic cleaning
                clean = clean_text(content)
                
                # Normalization
                normalized = normalize_text(clean)
                
                # Split into sentences
                sentences = split_into_sentences(normalized, min_chars, max_chars)
                
                # Add only up to the limit
                available_sentences = max_sentences - len(all_sentences)
                all_sentences.extend(sentences[:available_sentences])
                
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                failed_files += 1
                continue

    print(f"Processed files: {processed_files}")
    print(f"Failed files: {failed_files}")
    
    return all_sentences


def main():
    # Main function for generating training dataset from legal documents
    parser = argparse.ArgumentParser(
        description='Generate training dataset from cleaned legal documents using spaCy sentence segmentation'
    )
    parser.add_argument(
        '--input', '-i',
        default='dataCleaned/Laws',
        help='Input directory containing cleaned TXT files (default: dataCleaned/Laws)'
    )
    parser.add_argument(
        '--output', '-o',
        default='dataset_leyes.json',
        help='Output JSON file for the dataset (default: datasetTrain.json)'
    )
    parser.add_argument(
        '--min-chars',
        type=int,
        default=100,
        help='Minimum sentence length in characters (default: 100)'
    )
    parser.add_argument(
        '--max-chars',
        type=int,
        default=200,
        help='Maximum sentence length in characters (default: 200)'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=5000,
        help='Maximum number of sentences to generate (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{args.input}' does not exist or is not a directory.")
        return 1
    
    # Validate parameters
    if args.min_chars >= args.max_chars:
        print(f"Error: --min-chars ({args.min_chars}) must be less than --max-chars ({args.max_chars})")
        return 1
    
    if args.max_sentences <= 0:
        print(f"Error: --max-sentences must be a positive number")
        return 1
    
    # Process documents
    print("Starting processing pipeline...")
    print(f"Reading from: {args.input}")
    print(f"Character range: {args.min_chars}-{args.max_chars}")
    print(f"Maximum sentences: {args.max_sentences}")
    
    sentences = process_folder(
        args.input,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_sentences=args.max_sentences
    )

    print(f"\nTotal sentences generated: {len(sentences)}")
    
    if sentences:
        # Save dataset
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"text": sentences}, f, ensure_ascii=False, indent=2)
        
        print(f"\nDataset saved to {args.output}")
        return 0
    else:
        print("\nWarning: No sentences were generated.")
        return 1


if __name__ == "__main__":
    exit(main())