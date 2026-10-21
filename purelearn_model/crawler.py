"""
PureLearn Internet Data Crawler & Ingestion Pipeline.
Enables crawling, scraping, and transforming internet educational materials,
Wikipedia articles, arXiv scientific papers, and general web documents into
high-quality, neuro-inclusive, Socratic training datasets for PureLearn.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .dataset import strip_emojis

logger = logging.getLogger("purelearn.crawler")



# Curated high-impact topics for STEM and Neuro-inclusive education
DEFAULT_STEM_TOPICS = [
    # Mathematics
    "Calculus", "Differential calculus", "Integral calculus", "Chain rule", "Product rule",
    "Fundamental theorem of calculus", "Linear algebra", "Eigenvalues and eigenvectors",
    "Matrix multiplication", "Probability theory", "Bayes' theorem", "Fourier transform",
    # Physics
    "Newton's laws of motion", "Classical mechanics", "Conservation of momentum",
    "Kinetic energy", "Thermodynamics", "Electromagnetism", "Maxwell's equations",
    "Special relativity", "Quantum mechanics", "Wave-particle duality",
    # Computer Science & Algorithms
    "Algorithm", "Time complexity", "Big O notation", "Binary search algorithm",
    "Sorting algorithm", "Graph theory", "Breadth-first search", "Depth-first search",
    "Dynamic programming", "Transformer (deep learning architecture)", "Neural network",
    # Chemistry & Biology
    "Chemical reaction", "Chemical equilibrium", "Acid-base reaction",
    "Photosynthesis", "Cellular respiration", "DNA replication", "CRISPR gene editing",
    # Neurodiversity & Socratic Pedagogy
    "Socratic method", "Active recall", "Spaced repetition", "Neurodiversity",
    "Attention deficit hyperactivity disorder", "Dyslexia", "Working memory"
]


@dataclass
class CrawlerConfig:
    """Configuration options for web crawling and scraping."""
    user_agent: str = "PureLearn-Crawler/1.0 (+https://purelearn.vigilance.rw; educational Socratic AI training)"
    timeout: int = 12
    request_delay: float = 0.25
    max_depth: int = 2
    max_pages_per_domain: int = 30
    min_word_count: int = 50
    max_word_count: int = 6000
    verify_ssl: bool = True
    extract_math: bool = True
    generate_socratic: bool = True


class WebTextExtractor:
    """
    Cleans raw HTML and extracts structured educational text, preserving
    mathematical formulas, code blocks, lists, and headings.
    """

    def __init__(self, extract_math: bool = True):
        self.extract_math = extract_math
        self.seen_hashes: Set[str] = set()

    def clean_html(self, html_content: str, base_url: str = "") -> Dict[str, Any]:
        """
        Parses raw HTML and returns extracted title, clean text, and formulas.
        """
        if not html_content:
            return {"title": "", "text": "", "formulas": [], "word_count": 0}

        formulas: List[str] = []

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title_tag = soup.find("title") or soup.find("h1")
            title = title_tag.get_text().strip() if title_tag else "Educational Concept"
            # Clean title
            title = re.sub(r"\s*[-–—|]\s*(Wikipedia|arXiv|Wikiversity|Course|Portal|Home).*$", "", title, flags=re.I).strip()

            # Preserve formulas from MathJax / KaTeX / Wikipedia math elements
            if self.extract_math:
                # 1. Wikipedia math tags
                for math_el in soup.find_all(class_=re.compile(r"mwe-math|katex|math")):
                    # Check for annotation (TeX source)
                    tex_anno = math_el.find("annotation", attrs={"encoding": re.compile(r"x-tex", re.I)})
                    if tex_anno and tex_anno.text:
                        tex_str = tex_anno.text.strip()
                        formulas.append(tex_str)
                        math_el.replace_with(f" ${tex_str}$ ")
                    elif math_el.get("alt"):
                        alt_str = math_el["alt"].strip()
                        if alt_str.startswith("{\\displaystyle") or "\\" in alt_str:
                            formulas.append(alt_str)
                            math_el.replace_with(f" ${alt_str}$ ")

                # 2. Native math elements
                for math_tag in soup.find_all("math"):
                    alt = math_tag.get("alttext") or math_tag.get("alt")
                    if alt:
                        formulas.append(alt)
                        math_tag.replace_with(f" ${alt}$ ")

            # Remove boilerplates, scripts, navigation, footers
            for unwanted in soup.find_all([
                "script", "style", "nav", "footer", "header", "aside", "noscript",
                "iframe", "svg", "form", "button", "input", "select", "figure", "figcaption"
            ]):
                unwanted.decompose()

            # Remove common wiki / webpage chrome classes
            for unwanted in soup.find_all(class_=re.compile(
                r"navbox|mw-jump-link|toc|mw-editsection|reflist|reference|infobox|catlinks|sidebar|footer|ad-|advertisement|cookie",
                re.I
            )):
                unwanted.decompose()

            # Extract paragraphs, headings, and lists
            content_blocks = []
            for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "pre", "blockquote"]):
                tag = elem.name
                text = elem.get_text().strip()
                if not text:
                    continue

                if tag in ["h1", "h2"]:
                    # Skip reference or license sections
                    if any(stop_word in text.lower() for stop_word in ["see also", "references", "external links", "notes", "bibliography", "further reading"]):
                        break
                    content_blocks.append(f"\n\n## {text}\n")
                elif tag in ["h3", "h4"]:
                    content_blocks.append(f"\n### {text}\n")
                elif tag in ["ul", "ol"]:
                    items = [li.get_text().strip() for li in elem.find_all("li") if li.get_text().strip()]
                    for item in items[:15]:
                        content_blocks.append(f"• {item}")
                elif tag == "pre":
                    content_blocks.append(f"```\n{text}\n```")
                else:
                    # Paragraph
                    # Filter out short cookie/copyright blurbs
                    if len(text.split()) >= 4:
                        content_blocks.append(text)

            clean_text = "\n\n".join(content_blocks)
        else:
            # Fallback regex parser if BeautifulSoup is unavailable
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else "Educational Material"

            # Strip scripts and styles
            stripped = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", "", html_content, flags=re.I | re.DOTALL)
            # Replace paragraph tags with newlines
            stripped = re.sub(r"</?(p|h[1-6]|li|tr)[^>]*>", "\n\n", stripped, flags=re.I)
            # Strip remaining tags
            clean_text = re.sub(r"<[^>]+>", " ", stripped)

        # Normalize whitespace
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        clean_text = re.sub(r"[ \t]+", " ", clean_text).strip()

        words = clean_text.split()
        word_count = len(words)

        return {
            "title": title,
            "text": clean_text,
            "formulas": formulas,
            "word_count": word_count,
        }

    def is_duplicate(self, text: str) -> bool:
        """Checks if content has already been seen using normalized MD5 hash."""
        norm_sample = re.sub(r"\W+", "", text[:800].lower())
        if not norm_sample:
            return True
        hash_val = hashlib.md5(norm_sample.encode("utf-8")).hexdigest()
        if hash_val in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_val)
        return False


class WikipediaHarvester:
    """
    Directly fetches encyclopedic and educational STEM articles
    from Wikipedia and Wikiversity using the high-speed MediaWiki API.
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session() if requests else None
        if self.session:
            self.session.headers.update({"User-Agent": self.config.user_agent})

    def search_topics(self, query: str, limit: int = 5) -> List[str]:
        """Searches Wikipedia for article titles matching query."""
        if not self.session:
            return [query]
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,
            "format": "json",
        }
        try:
            time.sleep(self.config.request_delay)
            resp = self.session.get(self.api_url, params=params, timeout=self.config.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 2 and isinstance(data[1], list):
                    return data[1]
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return [query]

    def fetch_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetches full clean text extract of an article by title."""
        if not self.session:
            return None
        params = {
            "action": "query",
            "prop": "extracts|info",
            "titles": title,
            "explaintext": True,
            "exsectionformat": "wiki",
            "redirects": 1,
            "format": "json",
        }
        try:
            time.sleep(self.config.request_delay)
            resp = self.session.get(self.api_url, params=params, timeout=self.config.timeout)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        continue
                    clean_extract = page_data.get("extract", "")
                    if not clean_extract:
                        continue

                    # Filter out sections like 'See also', 'References', etc.
                    clean_extract = re.split(r"\n==\s*(See also|References|Further reading|Notes|External links)\s*==", clean_extract)[0]

                    words = clean_extract.split()
                    if len(words) < self.config.min_word_count:
                        continue

                    return {
                        "title": page_data.get("title", title),
                        "text": clean_extract.strip(),
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_data.get('title', title))}",
                        "source": "wikipedia",
                        "word_count": len(words),
                    }
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed for '{title}': {e}")
        return None


class ArxivHarvester:
    """
    Fetches open scientific and mathematical research papers and abstracts
    from the arXiv API.
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.api_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session() if requests else None
        if self.session:
            self.session.headers.update({"User-Agent": self.config.user_agent})

    def fetch_papers(self, topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Queries arXiv API for papers matching topic."""
        if not self.session:
            return []
        params = {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        papers = []
        try:
            time.sleep(self.config.request_delay)
            resp = self.session.get(self.api_url, params=params, timeout=self.config.timeout)
            if resp.status_code == 200:
                # Parse XML using regex or BeautifulSoup
                content = resp.text
                entries = re.findall(r"<entry>(.*?)</entry>", content, re.DOTALL)
                for entry in entries:
                    title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    id_m = re.search(r"<id>(.*?)</id>", entry, re.DOTALL)

                    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
                    summary = re.sub(r"\s+", " ", summary_m.group(1)).strip() if summary_m else ""
                    paper_url = id_m.group(1).strip() if id_m else ""

                    if title and summary and len(summary.split()) >= self.config.min_word_count:
                        papers.append({
                            "title": title,
                            "text": f"Title: {title}\n\nAbstract:\n{summary}",
                            "url": paper_url,
                            "source": "arxiv",
                            "word_count": len(summary.split()),
                        })
        except Exception as e:
            logger.warning(f"ArXiv query failed for '{topic}': {e}")
        return papers


class RecursiveWebSpider:
    """
    Crawls educational websites, documentation, and tutorials starting from seed URLs.
    Includes depth limit, domain constraints, and polite rate limiting.
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.extractor = WebTextExtractor(extract_math=self.config.extract_math)
        self.session = requests.Session() if requests else None
        if self.session:
            self.session.headers.update({"User-Agent": self.config.user_agent})

    def crawl_url(
        self,
        seed_url: str,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Crawls a seed URL recursively up to max_depth and max_pages.
        """
        if not self.session:
            return []

        max_depth = max_depth if max_depth is not None else self.config.max_depth
        max_pages = max_pages if max_pages is not None else self.config.max_pages_per_domain

        parsed_seed = urllib.parse.urlparse(seed_url)
        allowed_domain = parsed_seed.netloc.lower()

        queue: List[Tuple[str, int]] = [(seed_url, 0)]
        visited: Set[str] = set()
        crawled_articles: List[Dict[str, Any]] = []

        while queue and len(crawled_articles) < max_pages:
            current_url, depth = queue.pop(0)

            # Strip fragments
            current_url = urllib.parse.urldefrag(current_url)[0]
            if current_url in visited:
                continue
            visited.add(current_url)

            # Domain check
            parsed = urllib.parse.urlparse(current_url)
            if parsed.netloc.lower() != allowed_domain:
                continue

            # Skip binaries and images
            if re.search(r"\.(pdf|png|jpg|jpeg|gif|svg|zip|tar|gz|mp4|mp3|exe|iso)$", parsed.path, re.I):
                continue

            try:
                time.sleep(self.config.request_delay)
                resp = self.session.get(
                    current_url,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue

                extracted = self.extractor.clean_html(resp.text, base_url=current_url)

                if (
                    extracted["word_count"] >= self.config.min_word_count
                    and not self.extractor.is_duplicate(extracted["text"])
                ):
                    crawled_articles.append({
                        "title": extracted["title"],
                        "text": extracted["text"],
                        "url": current_url,
                        "formulas": extracted["formulas"],
                        "source": "web_spider",
                        "word_count": extracted["word_count"],
                    })
                    logger.info(f"Crawled ({len(crawled_articles)}/{max_pages}): {extracted['title']} ({extracted['word_count']} words)")

                # If depth allows, discover child links
                if depth < max_depth and BeautifulSoup is not None:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"].strip()
                        abs_url = urllib.parse.urljoin(current_url, href)
                        abs_url = urllib.parse.urldefrag(abs_url)[0]
                        if abs_url not in visited and urllib.parse.urlparse(abs_url).netloc.lower() == allowed_domain:
                            queue.append((abs_url, depth + 1))

            except Exception as e:
                logger.debug(f"Failed to crawl {current_url}: {e}")

        return crawled_articles


class SocraticDataSynthesizer:
    """
    Transforms crawled educational articles, formulas, and STEM explanations
    into structured multi-turn Socratic dialogues and neuro-inclusive training pairs.
    """

    def synthesize_socratic_dialogue(self, article: Dict[str, Any], variant_id: int = 1) -> Dict[str, Any]:
        """
        Converts an article into a multi-turn Socratic conversation with neuro-inclusive tagging.
        """
        title = article["title"]
        text = article["text"]
        formulas = article.get("formulas", [])

        # Extract first 1-2 substantive sentences for concept introduction
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 4]
        lead_summary = sentences[0] if sentences else f"{title} is a fundamental concept."
        elaboration = sentences[1] if len(sentences) > 1 else ""

        # Identify formula or key rule
        if formulas:
            key_formula = formulas[0]
        else:
            # Look for equations in text or generate symbolic representation
            formula_match = re.search(r"(\$[^\$]+\$|[A-Za-z]\s*=\s*[^.\n]+)", text)
            key_formula = formula_match.group(1).strip("$") if formula_match else f"\\text{{{title}}}"

        # Socratic questions aligned with concept
        socratic_questions = [
            f"Looking at the core principle of {title}, what is the foundational variable or condition that must be satisfied?",
            f"If we examine this scenario, how does changing the initial input affect the governing equation?",
            f"What happens to the system when the primary variable approaches its boundary condition?",
            f"Why does this physical or mathematical relationship hold true under ideal conditions?"
        ]
        socratic_q = socratic_questions[variant_id % len(socratic_questions)]

        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"Can you help me understand {title}? How does it work and what are the main principles?"
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Let's break down **{title}** step-by-step.\n\n"
                    f"• **Core Principle:** {lead_summary}\n"
                    f"• **Governing Expression:** <|katex_math|> {key_formula}\n\n"
                    f"<|socratic_question|> {socratic_q}"
                )
            },
            {
                "role": "user",
                "content": f"The relationship depends on how the primary quantities interact according to {title}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> **Exactly right.**\n\n"
                    f"<|scaffold|> Notice how {elaboration}\n\n"
                    f"<|socratic_question|> When we look at <|katex_math|> {key_formula}, does the output scale linearly, exponentially, or inversely with the input?"
                )
            },
            {
                "role": "user",
                "content": "It scales directly based on the mathematical structure of the governing law."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> **Brilliant reasoning.**\n\n"
                    f"<|study_notes|> **Summary of {title}:**\n"
                    f"- **Definition:** {lead_summary}\n"
                    f"- **Key Equation:** ${key_formula}$\n"
                    f"- **Takeaway:** Always identify the constants and boundary constraints before solving!"
                )
            }
        ]


        return {
            "id": f"crawled_socratic_{variant_id:06d}",
            "category": "Crawled - Educational",
            "concept": title,
            "dialogue": dialogue,
            "source_url": article.get("url", ""),
        }

    def synthesize_pretraining_document(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats an article into a high-density causal language model document.
        """
        title = article["title"]
        text = article["text"]

        formatted_doc = (
            f"<|system|>You are Purelearn AI, an educational and STEM Socratic engine.<|im_end|>\n"
            f"<|user|>Provide an in-depth, structured explanation of {title}.<|im_end|>\n"
            f"<|assistant|><|adhd_chunk|> # {title}\n\n{text}\n\n"
            f"<|study_notes|>\n- **Topic:** {title}\n- **Core Summary:** {text[:200]}...<|im_end|>"
        )

        return {
            "title": title,
            "text": formatted_doc,
            "url": article.get("url", ""),
            "word_count": len(text.split()),
        }


class InternetDataPipeline:
    """
    Unified Internet Data Harvesting Pipeline.
    Orchestrates crawling from Wikipedia, ArXiv, and web URLs, cleans content,
    synthesizes Socratic dialogues, and saves production-ready JSONL datasets.
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.wiki = WikipediaHarvester(self.config)
        self.arxiv = ArxivHarvester(self.config)
        self.spider = RecursiveWebSpider(self.config)
        self.synthesizer = SocraticDataSynthesizer()
        self.extractor = WebTextExtractor(extract_math=self.config.extract_math)

    def harvest_topics(
        self,
        topics: List[str],
        include_wikipedia: bool = True,
        include_arxiv: bool = True,
        max_per_topic: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Harvests articles across a list of topics from Wikipedia and ArXiv.
        """
        articles: List[Dict[str, Any]] = []

        for topic in topics:
            topic = topic.strip()
            if not topic:
                continue

            logger.info(f"[Crawler] Harvesting internet data for topic: '{topic}'")

            # 1. Wikipedia harvest
            if include_wikipedia:
                titles = self.wiki.search_topics(topic, limit=max_per_topic)
                for t in titles:
                    art = self.wiki.fetch_article(t)
                    if art and not self.extractor.is_duplicate(art["text"]):
                        art["text"] = strip_emojis(art["text"])
                        articles.append(art)
                        logger.info(f"  [Wikipedia] {art['title']} ({art['word_count']} words)")

            # 2. ArXiv harvest
            if include_arxiv:
                papers = self.arxiv.fetch_papers(topic, max_results=max_per_topic)
                for p in papers:
                    if not self.extractor.is_duplicate(p["text"]):
                        p["text"] = strip_emojis(p["text"])
                        articles.append(p)
                        logger.info(f"  [arXiv] {p['title']} ({p['word_count']} words)")

        return articles

    def harvest_urls(self, urls: List[str], max_pages_per_url: int = 15) -> List[Dict[str, Any]]:
        """
        Crawls a list of custom web URLs.
        """
        all_articles = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            logger.info(f"[Crawler] Crawling URL: {url}")
            crawled = self.spider.crawl_url(url, max_pages=max_pages_per_url)
            for c in crawled:
                c["text"] = strip_emojis(c["text"])
            all_articles.extend(crawled)
        return all_articles

    def process_and_export(
        self,
        articles: List[Dict[str, Any]],
        output_file: str,
        generate_socratic: bool = True,
        append: bool = False,
    ) -> int:
        """
        Converts crawled articles into Socratic dialogues or pretraining docs
        and writes them out to a JSONL dataset file with all emojis stripped.
        """
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        records_written = 0
        with open(out_path, mode, encoding="utf-8") as f:
            for idx, art in enumerate(articles, 1):
                if generate_socratic:
                    # Write Socratic dialogue entry
                    dialogue_record = self.synthesizer.synthesize_socratic_dialogue(art, variant_id=idx)
                    f.write(json.dumps(dialogue_record, ensure_ascii=False) + "\n")
                    records_written += 1

                # Also write full pretraining document for continuous learning
                doc_record = self.synthesizer.synthesize_pretraining_document(art)
                f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
                records_written += 1

        logger.info(f"[Crawler] Successfully exported {records_written} training records to '{output_file}'")
        return records_written


def run_crawler_cli():
    """Command-line interface for the PureLearn Internet Data Crawler."""
    parser = argparse.ArgumentParser(description="PureLearn Internet Data Crawler & Training Data Pipeline")
    parser.add_argument("--topics", type=str, default=None, help="Comma-separated topics (e.g. 'Calculus,Quantum Mechanics,Algorithms')")
    parser.add_argument("--urls", type=str, default=None, help="Comma-separated seed URLs to crawl")
    parser.add_argument("--stem_pack", action="store_true", help="Crawl standard comprehensive STEM and neuro-inclusive topics pack")
    parser.add_argument("--no_wikipedia", action="store_true", help="Disable Wikipedia harvesting")
    parser.add_argument("--no_arxiv", action="store_true", help="Disable ArXiv harvesting")
    parser.add_argument("--max_per_topic", type=int, default=3, help="Max articles per topic")
    parser.add_argument("--max_pages_per_url", type=int, default=15, help="Max pages per seed URL")
    parser.add_argument("--output", type=str, default="data/crawled_dataset.jsonl", help="Output JSONL dataset path")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset file")
    parser.add_argument("--no_socratic", action="store_true", help="Only output raw documents, skip Socratic dialogue synthesis")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    pipeline = InternetDataPipeline()

    topics_to_harvest = []
    if args.stem_pack:
        topics_to_harvest.extend(DEFAULT_STEM_TOPICS)
    if args.topics:
        topics_to_harvest.extend([t.strip() for t in args.topics.split(",") if t.strip()])

    articles = []

    if topics_to_harvest:
        print(f"[Crawler] Harvesting topics across internet: {len(topics_to_harvest)} topics specified")
        topic_articles = pipeline.harvest_topics(
            topics=topics_to_harvest,
            include_wikipedia=not args.no_wikipedia,
            include_arxiv=not args.no_arxiv,
            max_per_topic=args.max_per_topic,
        )
        articles.extend(topic_articles)

    if args.urls:
        seed_urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        print(f"[Crawler] Crawling seed URLs: {seed_urls}")
        url_articles = pipeline.harvest_urls(seed_urls, max_pages_per_url=args.max_pages_per_url)
        articles.extend(url_articles)

    if not articles:
        print("[Crawler] No topics or URLs provided. Use --topics 'Calculus,Physics' or --stem_pack or --urls 'https://...'")
        return

    count = pipeline.process_and_export(
        articles,
        output_file=args.output,
        generate_socratic=not args.no_socratic,
        append=args.append,
    )
    print(f"[Crawler] Crawling complete. {len(articles)} internet articles processed -> {count} training samples saved in '{args.output}'.")


if __name__ == "__main__":
    run_crawler_cli()

