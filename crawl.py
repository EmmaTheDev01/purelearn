#!/usr/bin/env python3
"""
PureLearn Internet Data Crawler CLI.
Harvests educational materials, encyclopedic STEM articles, research papers,
and documentation from the internet, and synthesizes them into Socratic
neuro-inclusive training datasets.

Usage examples:
  python crawl.py --topics "Calculus,Quantum Mechanics,Algorithms" --output data/crawled_dataset.jsonl
  python crawl.py --stem_pack --max_per_topic 2 --output data/crawled_stem.jsonl
  python crawl.py --urls "https://en.wikibooks.org/wiki/Calculus" --max_pages_per_url 10
"""

from purelearn_model.crawler import run_crawler_cli

if __name__ == "__main__":
    run_crawler_cli()
