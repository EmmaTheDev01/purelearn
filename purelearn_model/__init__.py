"""
Purelearn AI Model Architecture & Engine
A custom PyTorch-based neural language model and inference engine
built from scratch for the Purelearn Neuro-Inclusive Socratic AI Tutor.
"""

from .config import PureLearnConfig, TokenizerConfig, TrainingConfig
from .model import PureLearnLM
from .tokenizer import PureLearnTokenizer
from .engine import PureLearnEngine
from .crawler import (
    CrawlerConfig,
    WebTextExtractor,
    WikipediaHarvester,
    ArxivHarvester,
    RecursiveWebSpider,
    SocraticDataSynthesizer,
    InternetDataPipeline,
)

__version__ = "1.1.0"
__all__ = [
    "PureLearnConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "PureLearnLM",
    "PureLearnTokenizer",
    "PureLearnEngine",
    "CrawlerConfig",
    "WebTextExtractor",
    "WikipediaHarvester",
    "ArxivHarvester",
    "RecursiveWebSpider",
    "SocraticDataSynthesizer",
    "InternetDataPipeline",
]

