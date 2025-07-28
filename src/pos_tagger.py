"""
Vietnamese POS tagging module
"""
from typing import List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class VietnamesePOSTagger:
    """
    Vietnamese Part-of-Speech tagger with multiple backends
    """
    
    # POS tag mappings for standardization
    TAG_MAP = {
        # Noun tags
        'N': 'NOUN', 'Np': 'PROPN', 'Nc': 'NOUN',
        # Verb tags  
        'V': 'VERB', 'Vv': 'VERB', 'Vc': 'VERB',
        # Adjective
        'A': 'ADJ',
        # Others
        'P': 'PRON', 'R': 'ADV', 'L': 'DET',
        'M': 'NUM', 'E': 'ADP', 'C': 'CCONJ',
        'I': 'INTJ', 'T': 'PART', 'Y': 'SYM',
        'X': 'X', 'CH': 'PUNCT'
    }
    
    def __init__(self, method: str = 'underthesea'):
        """
        Initialize POS tagger
        
        Args:
            method: POS tagging method
        """
        self.method = method
        self._init_taggers()
        
    def _init_taggers(self):
        """Initialize available POS taggers"""
        self.taggers = {}
        
        try:
            from underthesea import pos_tag
            self.taggers['underthesea'] = pos_tag
            logger.info("Initialized Underthesea POS tagger")
        except ImportError:
            logger.warning("Underthesea not available for POS tagging")
            
        try:
            from pyvi import ViPosTagger
            self.taggers['pyvi'] = ViPosTagger.postagging
            logger.info("Initialized PyVi POS tagger")
        except ImportError:
            logger.warning("PyVi not available for POS tagging")
    
    def tag(self, tokens: Union[List[str], str]) -> List[Tuple[str, str]]:
        """
        POS tag Vietnamese tokens
        
        Args:
            tokens: List of tokens or raw text
            
        Returns:
            List of (token, pos_tag) tuples
        """
        if self.method not in self.taggers:
            raise ValueError(f"POS tagger {self.method} not available")
        
        tagger = self.taggers[self.method]
        
        if self.method == 'underthesea':
            # Underthesea expects raw text
            if isinstance(tokens, list):
                text = ' '.join(tokens)
            else:
                text = tokens
            return tagger(text)
            
        elif self.method == 'pyvi':
            # PyVi expects tokenized text
            if isinstance(tokens, str):
                from pyvi import ViTokenizer
                tokens_str = ViTokenizer.tokenize(tokens)
            else:
                tokens_str = ' '.join(tokens)
            
            # PyVi returns two lists
            words, tags = tagger(tokens_str)
            return list(zip(words, tags))
    
    def tag_with_confidence(self, tokens: Union[List[str], str]) -> List[Tuple[str, str, float]]:
        """
        POS tag with confidence scores (if available)
        
        Returns:
            List of (token, pos_tag, confidence) tuples
        """
        # For now, return default confidence
        tagged = self.tag(tokens)
        return [(token, tag, 1.0) for token, tag in tagged]
    
    def get_universal_tags(self, tagged_tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Convert to Universal POS tags
        
        Args:
            tagged_tokens: List of (token, pos_tag) tuples
            
        Returns:
            List of (token, universal_tag) tuples
        """
        universal_tagged = []
        for token, tag in tagged_tokens:
            universal_tag = self.TAG_MAP.get(tag, 'X')
            universal_tagged.append((token, universal_tag))
        return universal_tagged