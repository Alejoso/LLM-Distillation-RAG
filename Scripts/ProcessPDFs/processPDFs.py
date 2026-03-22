import re
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from Scripts.ProcessPDFs.cleaningPatterns import PATTERNS

# PDF processing
try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class DocumentCleaner:

    def __init__(self):
        # Common patterns to remove
        self.patterns = PATTERNS
    
    # Normalize whitespace: collapse multiple spaces, tabs, and newlines into a single space or newline.
    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    # Remove lines that are too short, except for protected ones (like "I", "V", "X" which might be legal references).
    def remove_short_lines(self, text: str, min_len: int = 3) -> str:
        protected = {'I', 'V', 'X', 'L', 'C'}
        lines = []

        for line in text.split('\n'):
            stripped = line.strip()
            if stripped in protected or len(stripped) >= min_len:
                lines.append(stripped)

        return '\n'.join(lines)

    # Auxiliary function to cut text before the index section, which often contains navigation and UI elements.
    def cut_before_index(self, text: str) -> str:
        pattern = r'Í\s*N\s*D\s*I\s*C\s*E\s*\[Mostrar\]'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[match.end():]
        return text

    # Auxiliary function to remove the SUIN-Juriscol disclaimer that appears at the end of documents, which is not relevant for content and can be very long.
    def remove_suin_disclaimer(self, text: str) -> str:
        pattern = (
            r'Los datos publicados en SUIN-Juriscol'
            r'[\s\S]*?'
            r'(Ministerio\.?)'
        )
        return re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Auxiliary function to protect legal structure elements like article and paragraph numbering from being broken by cleaning steps. This is crucial to maintain the integrity of the legal document's structure.
    def protect_legal_structure(self, text: str) -> str:
        # Fix broken article numbering (e.g., "Artículo 1\n." -> "Artículo 1.")
        text = re.sub(r'(Art[ií]culo\s+\d+[º°]?)\s*\n\s*\.', r'\g<1>.', text, flags=re.IGNORECASE)
        
        # Fix cases where the dot is on its own line
        text = re.sub(r'(Art[ií]culo\s+\d+[º°]?)\s*\n\s*\.\s*\n', r'\g<1>.\n', text, flags=re.IGNORECASE)
        
        # Fix broken paragraph with ordinal number (e.g., "Parágrafo\n1º" -> "Parágrafo 1º")
        text = re.sub(
            r'(Par[áa]grafo)\s*\n\s*(\d+[º°]?|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo|transitorio\s+\d+[º°]?)',
            r'\g<1> \g<2>',
            text,
            flags=re.IGNORECASE
        )
        
        # Fix paragraph with ordinal and dot on separate line (e.g., "Parágrafo 1º\n." -> "Parágrafo 1º.")
        text = re.sub(
            r'(Par[áa]grafo\s+(?:\d+[º°]?|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo|transitorio\s+\d+[º°]?))\s*\n\s*\.',
            r'\g<1>.',
            text,
            flags=re.IGNORECASE
        )
        
        # Fix cases where the dot is on its own line for paragraphs
        text = re.sub(
            r'(Par[áa]grafo\s+(?:\d+[º°]?|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo|transitorio\s+\d+[º°]?))\s*\n\s*\.\s*\n',
            r'\g<1>.\n',
            text,
            flags=re.IGNORECASE
        )
        
        # Add newlines before articles, chapters, paragraphs, and "DECRETA:" to ensure they are recognized as separate sections. This helps maintain the logical structure of the document and improves readability.
        patterns = {
            r'^\s*Art[ií]culo\s+\d+[º°]?\.?': r'\n\n\g<0>\n',
            r'^\s*Cap[ií]tulo\s+[IVXLC]+': r'\n\n\g<0>\n',
            r'^\s*Par[áa]grafo(\s+(?:\d+[º°]?|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo|transitorio\s+\d+[º°]?))?\.?': r'\n\g<0>\n',
            r'^\s*DECRETA:': r'\n\nDECRETA:\n'
        }

        for p, rpl in patterns.items():
            text = re.sub(p, rpl, text, flags=re.IGNORECASE)

        return text
    
    # Main function to clean text by applying all cleaning steps in a logical order. This function ensures that the document is cleaned while preserving the essential legal structure and content, making it suitable for use in RAG systems.
    def clean_text(self, text: str) -> str:
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = self.cut_before_index(text)
        text = self.remove_suin_disclaimer(text)
        text = self.protect_legal_structure(text)
        
        for _, patterns in self.patterns.items():
            for pattern in patterns:
                text = re.sub(
                    pattern,
                    '',
                    text,
                    flags=re.IGNORECASE | re.MULTILINE
                )

        text = self.normalize_whitespace(text)
        text = self.remove_short_lines(text)

        return text.strip()

    # Generate a report of cleaning results
    def cleaning_report(self, original: str, cleaned: str) -> dict:
        return {
            'original_chars': len(original),
            'cleaned_chars': len(cleaned),
            'reduction_ratio': round(
                1 - len(cleaned) / max(len(original), 1), 4
            )
        }

    # Extract text from PDF using PyPDFium2.
    def extract_text_from_pdf(self, pdf_path: str) -> str:

        if not PDF_SUPPORT:
            raise ImportError("pypdfium2 is not installed. Install it with: pip install pypdfium2")
        
        pdf = pdfium.PdfDocument(pdf_path)
        text_parts = []
        
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            text_parts.append(text)
            textpage.close()
            page.close()
        
        pdf.close()
        return '\n\n'.join(text_parts)
  
    # Funtion to process a single file.  
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        # Process a single PDF file.
        file_ext = Path(input_path).suffix.lower()
        
        # Only process PDF files
        if file_ext != '.pdf':
            raise ValueError(f"Only PDF files are supported. Got: {file_ext}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(input_path)
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        report = self.cleaning_report(text, cleaned_text)
        print(f"\n Cleaning report for {input_path}:")
        print(f"  Original characters: {report['original_chars']}")
        print(f"  Cleaned characters : {report['cleaned_chars']}")
        print(f"  Reduction ratio    : {report['reduction_ratio']}")
        
        # Save if output path provided
        if output_path:
            # Change extension to .txt for output
            output_path_obj = Path(output_path)
            if output_path_obj.suffix == '.pdf':
                output_path = str(output_path_obj.with_suffix('.txt'))
            
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f" Processed: {input_path} -> {output_path}")
        
        return cleaned_text
    
    # Function to process directory
    def process_directory(self, input_dir: str, output_dir: str, extensions: List[str] = ['.pdf']) -> Dict[str, str]:
        
        # Process all PDF files in a directory.
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        processed_files = {}
        
        # Find all matching files
        for ext in extensions:
            for file_path in input_path.rglob(f'*{ext}'):
                # Calculate relative path
                rel_path = file_path.relative_to(input_path)
                out_file = output_path / rel_path
                
                # Change extension to .txt for PDF outputs
                out_file = out_file.with_suffix('.txt')
                
                # Process file
                self.process_file(str(file_path), str(out_file))
                processed_files[str(file_path)] = str(out_file)
        
        return processed_files


def main():
    
    # Cleaner for documents using command-line interface.
    parser = argparse.ArgumentParser(
        description='Clean documents for RAG systems'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file or directory'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file or directory'
    )
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.pdf'],
        help='File extensions to process (default: .pdf)'
    )
    
    args = parser.parse_args()
    
    cleaner = DocumentCleaner()
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        cleaner.process_file(args.input, args.output)
        print(f"\n Document cleaned successfully!")
    elif input_path.is_dir():
        # Process directory
        processed = cleaner.process_directory(
            args.input, 
            args.output, 
            args.extensions
        )
        print(f"\n Processed {len(processed)} documents successfully!")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())


