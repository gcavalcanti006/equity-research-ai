"""Run the extractor.

Usage:
    python -m src.main                 # real run (needs ANTHROPIC_API_KEY in the environment)
    python -m src.main --dry-run       # read PDFs and count chunks, no API call
    python -m src.main --only vale_3q26_usd.pdf
"""
import argparse
import os
import sys
from pathlib import Path

import anthropic
import yaml

from src import extractor, pdf_reader, report, validator
from src.filename_parser import parse_filename
from src.validator import Flag


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="no API call; show pages/chunks")
    ap.add_argument("--only", help="process only this PDF file name")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached API answers")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    pdfs = sorted(Path(cfg["paths"]["pdf_dir"]).glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.name == args.only]
    if not pdfs:
        print("No PDFs found.")
        return 1

    # The key is only checked for existence; its value is never printed or stored.
    if not args.dry_run and client is None:
        if not os.environ.get(cfg["api_key_env"]):
            print(f"Environment variable {cfg['api_key_env']} is not set. "
                  "Export it in this terminal, or use --dry-run.")
            return 1
        client = anthropic.Anthropic()

    cache_dir = None if args.no_cache else cfg["paths"]["cache"]
    results, flags = [], []

    for pdf in pdfs:
        try:
            parsed = parse_filename(pdf.name)
        except ValueError as exc:
            flags.append(Flag(pdf.stem, "file name", validator.BAD_FILENAME, str(exc)))
            continue
        pages = pdf_reader.read_pages(pdf)
        if pdf_reader.is_scanned(pages, cfg["min_text_chars"]):
            flags.append(Flag(pdf.stem, "text", validator.SCANNED,
                              "almost no extractable text (scanned?); skipped. OCR is not implemented"))
            continue
        chunks = pdf_reader.split_into_chunks(pages, cfg["chunk_pages"], cfg["max_chars_per_chunk"])
        if args.dry_run:
            chars = sum(len(c.text) for c in chunks)
            print(f"{pdf.name}: {len(pages)} pages, {chars:,} characters, {len(chunks)} chunk(s)")
            continue

        print(f"Extracting {pdf.name} ({len(chunks)} chunk(s))...")
        try:
            result, extract_flags = extractor.extract_document(client, pages, cfg, pdf.stem, cache_dir)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            print(f"Stopped: the API rejected the credentials -> {extractor.describe_error(exc)}")
            return 1
        flags += extract_flags
        if result is None:
            continue
        flags += validator.validate_company(result, parsed, pages, cfg)
        results.append(result)

    if args.dry_run:
        report.print_flags(flags)
        return 0

    # Checks across companies
    flags += validator.check_currency_mix(results)
    flags += validator.check_definitions(results)
    flags += validator.check_pairs(results, cfg["validation"]["pair_margin_tol_pp"])

    df = report.write_outputs(results, flags, cfg["paths"]["outputs"])
    report.print_summary(df)
    report.print_flags(flags)
    print(f"\nSaved results.csv, results.xlsx and flags.csv in {cfg['paths']['outputs']}/")
    if not results:
        print("NOTHING WAS EXTRACTED: the files above are empty (headers only).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
