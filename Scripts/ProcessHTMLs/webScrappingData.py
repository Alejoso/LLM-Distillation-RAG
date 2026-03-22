import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from urllib.parse import urlparse, parse_qs
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable warnings about insecure requests (since we're using verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def run_scraper(
    sitemap_url,
    save_dir,
    sleep_between_requests,
    max_retries,
    backoff_factor,
    failed_log,
):
    # Define headers to mimic a browser to evit potential blocking by the server
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Retry/backoff configuration to handle flaky server responses
    connect_timeout = 15
    read_timeout = 60

    # Define the directory to save the downloaded HTML files
    save_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    print("Downloading sitemap...")

    # Fetch the sitemap and extract URLs
    response = session.get(
        sitemap_url,
        headers=headers,
        timeout=(connect_timeout, read_timeout),
        verify=False,
    )
    response.raise_for_status()

    # Parse the sitemap XML and extract document URLs
    soup = BeautifulSoup(response.text, "xml")
    urls = [loc.text.strip() for loc in soup.find_all("loc")]

    print(f"Found {len(urls)} documents.")
    print("Downloading all documents...\n")

    for index, url in enumerate(urls, start=1):
        print(f"Processing: {url}")

        # Extract the document ID from the URL query parameters to use as the file name
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        doc_id = query_params.get("id", ["sin_id"])[0]

        # Save the page content as an HTML file
        file_name = f"{doc_id}.html"
        file_path = save_dir / file_name

        # Skip already downloaded files to allow resume
        if file_path.exists():
            print(f"[{index}/{len(urls)}] Skipping existing: {file_name}")
            continue

        try:
            # Make a GET request to the document URL with a timeout and without SSL verification
            page = session.get(
                url,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
                verify=False,
            )

            if page.status_code >= 400:
                raise requests.HTTPError(f"HTTP {page.status_code}")

        except Exception as e:
            print("Error opening the page:", e)
            with open(failed_log, "a", encoding="utf-8") as log_file:
                log_file.write(f"{url}\t{e}\n")
            continue

        # Write the HTML content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page.text)

        print(f"[{index}/{len(urls)}] Saved as {file_name}\n")

        time.sleep(sleep_between_requests)

    print("Scraping finished.")


def main():
    
    # Set up command-line argument parsing for flexible configuration of the scraper
    parser = argparse.ArgumentParser(
        description="Download HTML documents from SUIN sitemap"
    )

    parser.add_argument(
        "--sitemap",
        default="https://www.suin-juriscol.gov.co/sitemapleyes.xml",
        help="Sitemap URL",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save downloaded HTML files",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep time between requests (default: 0.5)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retry attempts (default: 6)",
    )

    parser.add_argument(
        "--backoff",
        type=float,
        default=1.5,
        help="Backoff factor for retries (default: 1.5)",
    )

    parser.add_argument(
        "--failed-log",
        default="failed_downloads.txt",
        help="File to log failed downloads",
    )

    args = parser.parse_args()

    run_scraper(
        sitemap_url=args.sitemap,
        save_dir=Path(args.output),
        sleep_between_requests=args.sleep,
        max_retries=args.max_retries,
        backoff_factor=args.backoff,
        failed_log=Path(args.failed_log),
    )

if __name__ == "__main__":
    main()
