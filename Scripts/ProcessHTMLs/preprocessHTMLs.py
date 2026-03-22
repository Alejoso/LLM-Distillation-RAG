import argparse
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from text_normalization import normalize_body
sys.path.insert(0, str(Path(__file__).parent.parent))
from CleanlinessMetrics.compute_metrics import compute_quality_score

def strip_unwanted_elements(soup):
    # Remove non-content or boilerplate sections.
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()

    # Remove menus, sliders, TOC, and similar UI blocks.
    for tag in soup.find_all(["table", "div"], id=re.compile(r"toc|Resumen|NotasDestino", re.IGNORECASE)):
        tag.decompose()
    for tag in soup.find_all(["div"], class_=re.compile(r"slider|toc|resumenvigencias", re.IGNORECASE)):
        tag.decompose()

    # Remove elements that are hidden by inline styles.
    for tag in soup.find_all(style=True):
        if not hasattr(tag, "attrs") or tag.attrs is None:
            continue
        style = tag.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()

    # Remove UI toggles and boilerplate labels.
    for tag in soup.find_all(class_=re.compile(r"toctoggle|toc-link", re.IGNORECASE)):
        tag.decompose()

    # Remove paragraphs or spans that only announce auxiliary text blocks.
    for text_node in soup.find_all(string=re.compile(r"TEXTO\s+CORRESPONDIENTE\s+A", re.IGNORECASE)):
        parent = text_node.parent
        if parent:
            parent.decompose()

# Removes metadata lines from the body text based on the extracted metadata values.
def remove_metadata_lines(body: str, source: str, subtipo: str) -> str:
    lines = body.splitlines()
    source = source.strip() if source else ""
    subtipo = subtipo.strip() if subtipo else ""

    cleaned = []
    i = 0
    while i < len(lines):
        current = lines[i].strip()

        if source and current == source:
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        if subtipo and current.lower() == "subtipo:":
            if i + 1 < len(lines) and lines[i + 1].strip() == subtipo:
                i += 2
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue

        cleaned.append(lines[i])
        i += 1

    return "\n".join(cleaned)

# Main function to process all HTML files in the input directory and save cleaned TXT files in the output directory.
def process_directory(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    unusable_dir = output_dir.parent / "unusable_files"
    unusable_dir.mkdir(parents=True, exist_ok=True)

    
    html_files = list(input_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files.\n")
    
    usable_count = 0
    unusable_count = 0

    for file_path in html_files:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Extract metadata before removing hidden spans
        metadata = {}
        for span in soup.find_all("span", attrs={"field": True}):
            field = span.get("field")
            if field:
                metadata[field] = normalize_body(span.get_text(" ", strip=True), apply_body_rules=False)

        strip_unwanted_elements(soup)

        # Remove metadata spans from content extraction
        for span in soup.find_all("span", attrs={"field": True}):
            span.decompose()

        # Extract main text
        body_node = soup.body if soup.body else soup
        body = body_node.get_text(separator="\n")
        body = normalize_body(body)
        body = remove_metadata_lines(
            body,
            metadata.get("documento_fuente", ""),
            metadata.get("subtipo", ""),
        )
        
        # Calcula score de calidad
        metrics = compute_quality_score(body)

        # Build final structured text
        final_text = ""

        final_text += "TIPO: " + metadata.get("tipo", "") + "\n"
        final_text += "NUMERO: " + metadata.get("numero", "") + "\n"
        final_text += "ANIO: " + metadata.get("anio", "") + "\n"
        final_text += "ESTADO: " + metadata.get("estado_documento", "") + "\n"
        final_text += "ENTIDAD: " + metadata.get("entidad_emisora", "") + "\n"
        final_text += "SUBTIPO: " + metadata.get("subtipo", "") + "\n"
        final_text += "FECHA_EXPEDICION: " + metadata.get("fecha_expedicion", "") + "\n"
        final_text += "FECHA_PUBLICACION: " + metadata.get("fecha_diario_oficial", "") + "\n"
        final_text += "FUENTE: " + metadata.get("documento_fuente", "") + "\n"
        final_text += f"QUALITY_SCORE: {metrics['quality_score']}\n"
        final_text += f"QUALITY_STATUS: {metrics['quality_status']}\n"
        
        final_text += "CONTENIDO:\n"
        final_text += body

        # Determine output directory based on quality score
        if metrics['quality_score'] < 70:
            output_path = unusable_dir / file_path.with_suffix(".txt").name
            unusable_count += 1
            status_msg = f"[UNUSABLE - Score: {metrics['quality_score']}]"
        else:
            output_path = output_dir / file_path.with_suffix(".txt").name
            usable_count += 1
            status_msg = f"[USABLE - Score: {metrics['quality_score']}]"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"Processed {file_path.name} {status_msg}")

    print(f"Processing complete:")
    print(f"- Usable files (score >= 70): {usable_count}")
    print(f"- Unusable files (score < 70): {unusable_count}")
    print(f"Total files processed: {len(html_files)}")

# Entry point for command-line execution, allowing specification of input and output directories.
def main():
    parser = argparse.ArgumentParser(
        description="Clean and convert HTML legal documents to structured TXT format"
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing HTML files",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for cleaned TXT files",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print("Error: Input directory is not valid.")
        return 1

    process_directory(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
