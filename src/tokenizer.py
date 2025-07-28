"""
Vietnamese tokenization module with multiple methods
"""
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class VietnameseTokenizer:
    """
    Vietnamese word segmentation with multiple backend options
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize tokenizer
        
        Args:
            method: Tokenization method (underthesea, pyvi, vncorenlp, hybrid)
        """
        self.method = method
        self._init_tokenizers()
        
    def _init_tokenizers(self):
        """Initialize selected tokenizers"""
        self.tokenizers = {}
        
        try:
            from underthesea import word_tokenize
            self.tokenizers['underthesea'] = word_tokenize
            logger.info("Initialized Underthesea tokenizer")
        except ImportError:
            logger.warning("Underthesea not available")
            
        try:
            from pyvi import ViTokenizer
            self.tokenizers['pyvi'] = ViTokenizer.tokenize
            logger.info("Initialized PyVi tokenizer")
        except ImportError:
            logger.warning("PyVi not available")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.method == 'hybrid':
            return self._hybrid_tokenize(text)
        elif self.method in self.tokenizers:
            return self._single_tokenize(text, self.method)
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def _single_tokenize(self, text: str, method: str) -> List[str]:
        """Tokenize using a single method"""
        tokenizer = self.tokenizers[method]
        
        if method == 'underthesea':
            return tokenizer(text)
        elif method == 'pyvi':
            # PyVi returns string with underscores
            return tokenizer(text).split()
        else:
            return tokenizer(text)
    
    def _hybrid_tokenize(self, text: str) -> List[str]:
        """
        Hybrid tokenization combining multiple methods
        Uses voting mechanism for better accuracy
        """
        if len(self.tokenizers) < 2:
            logger.warning("Not enough tokenizers for hybrid mode")
            return self._single_tokenize(text, list(self.tokenizers.keys())[0])
        
        # Get results from all tokenizers
        all_results = {}
        for name, tokenizer in self.tokenizers.items():
            try:
                if name == 'pyvi':
                    tokens = tokenizer(text).split()
                else:
                    tokens = tokenizer(text)
                all_results[name] = tokens
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
        
        # Simple voting mechanism
        # For now, prefer Underthesea for segmentation accuracy
        if 'underthesea' in all_results:
            return all_results['underthesea']
        else:
            return list(all_results.values())[0]
    
    def tokenize_with_spans(self, text: str) -> List[tuple[str, int, int]]:
        """
        Tokenize and return token spans
        
        Returns:
            List of (token, start_idx, end_idx)
        """
        tokens = self.tokenize(text)
        spans = []
        current_idx = 0
        
        for token in tokens:
            # Handle underscore-connected tokens
            token_clean = token.replace('_', ' ')
            start_idx = text.find(token_clean, current_idx)
            if start_idx == -1:
                # Fallback for tokens not found
                start_idx = current_idx
            end_idx = start_idx + len(token_clean)
            spans.append((token, start_idx, end_idx))
            current_idx = end_idx
        
        return spans