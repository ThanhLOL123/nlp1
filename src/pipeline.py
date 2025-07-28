"""
Main pipeline class for Vietnamese text processing
"""
import yaml
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from .preprocessor import VietnamesePreprocessor
from .tokenizer import VietnameseTokenizer
from .pos_tagger import VietnamesePOSTagger
from .tone_processor import ToneProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineOutput:
    """Data class for pipeline output"""
    original_text: str
    normalized_text: str
    tokens: List[str]
    pos_tags: List[Tuple[str, str]]
    tone_info: Dict[str, any]
    processing_time: Dict[str, float]

class VietnameseNLPPipeline:
    """
    Main pipeline for Vietnamese text processing
    
    Example:
        >>> pipeline = VietnameseNLPPipeline('configs/config.yaml')
        >>> result = pipeline.process("Xin chào Việt Nam")
        >>> print(result.tokens)
        ['Xin', 'chào', 'Việt_Nam']
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.preprocessor = VietnamesePreprocessor()
        self.tokenizer = VietnameseTokenizer(
            method=self.config['pipeline']['tokenizer']['method']
        )
        self.pos_tagger = VietnamesePOSTagger(
            method=self.config['pipeline']['pos_tagger']['method']
        )
        self.tone_processor = ToneProcessor()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def process(self, text: str, verbose: bool = False) -> PipelineOutput:
        """
        Process Vietnamese text through the full pipeline
        
        Args:
            text: Input Vietnamese text
            verbose: Print processing steps
            
        Returns:
            PipelineOutput object containing all results
        """
        import time
        processing_time = {}
        
        # Step 1: Preprocessing
        start_time = time.time()
        normalized_text = self.preprocessor.normalize(text)
        processing_time['preprocessing'] = time.time() - start_time
        
        if verbose:
            logger.info(f"Normalized: {normalized_text}")
        
        # Step 2: Tokenization
        start_time = time.time()
        tokens = self.tokenizer.tokenize(normalized_text)
        processing_time['tokenization'] = time.time() - start_time
        
        if verbose:
            logger.info(f"Tokens: {tokens}")
        
        # Step 3: POS Tagging
        start_time = time.time()
        pos_tags = self.pos_tagger.tag(tokens)
        processing_time['pos_tagging'] = time.time() - start_time
        
        if verbose:
            logger.info(f"POS tags: {pos_tags}")
        
        # Step 4: Tone Processing
        start_time = time.time()
        tone_info = self.tone_processor.analyze_text(normalized_text, tokens)
        processing_time['tone_processing'] = time.time() - start_time
        
        return PipelineOutput(
            original_text=text,
            normalized_text=normalized_text,
            tokens=tokens,
            pos_tags=pos_tags,
            tone_info=tone_info,
            processing_time=processing_time
        )
    
    def process_batch(self, texts: List[str]) -> List[PipelineOutput]:
        """Process multiple texts"""
        return [self.process(text) for text in texts]