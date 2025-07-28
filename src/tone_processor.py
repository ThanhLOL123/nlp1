"""
Vietnamese tone processing module
"""
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class ToneProcessor:
    """
    Process Vietnamese tones (dấu thanh điệu)
    """
    
    def __init__(self):
        # Tone mappings
        self.tone_map = {
            # a tones
            'à': ('a', 'huyền'), 'á': ('a', 'sắc'),
            'ả': ('a', 'hỏi'), 'ã': ('a', 'ngã'), 'ạ': ('a', 'nặng'),
            # ă tones
            'ằ': ('ă', 'huyền'), 'ắ': ('ă', 'sắc'),
            'ẳ': ('ă', 'hỏi'), 'ẵ': ('ă', 'ngã'), 'ặ': ('ă', 'nặng'),
            # â tones  
            'ầ': ('â', 'huyền'), 'ấ': ('â', 'sắc'),
            'ẩ': ('â', 'hỏi'), 'ẫ': ('â', 'ngã'), 'ậ': ('â', 'nặng'),
            # e tones
            'è': ('e', 'huyền'), 'é': ('e', 'sắc'),
            'ẻ': ('e', 'hỏi'), 'ẽ': ('e', 'ngã'), 'ẹ': ('e', 'nặng'),
            # ê tones
            'ề': ('ê', 'huyền'), 'ế': ('ê', 'sắc'),
            'ể': ('ê', 'hỏi'), 'ễ': ('ê', 'ngã'), 'ệ': ('ê', 'nặng'),
            # i tones
            'ì': ('i', 'huyền'), 'í': ('i', 'sắc'),
            'ỉ': ('i', 'hỏi'), 'ĩ': ('i', 'ngã'), 'ị': ('i', 'nặng'),
            # o tones
            'ò': ('o', 'huyền'), 'ó': ('o', 'sắc'),
            'ỏ': ('o', 'hỏi'), 'õ': ('o', 'ngã'), 'ọ': ('o', 'nặng'),
            # ô tones
            'ồ': ('ô', 'huyền'), 'ố': ('ô', 'sắc'),
            'ổ': ('ô', 'hỏi'), 'ỗ': ('ô', 'ngã'), 'ộ': ('ô', 'nặng'),
            # ơ tones
            'ờ': ('ơ', 'huyền'), 'ớ': ('ơ', 'sắc'),
            'ở': ('ơ', 'hỏi'), 'ỡ': ('ơ', 'ngã'), 'ợ': ('ơ', 'nặng'),
            # u tones
            'ù': ('u', 'huyền'), 'ú': ('u', 'sắc'),
            'ủ': ('u', 'hỏi'), 'ũ': ('u', 'ngã'), 'ụ': ('u', 'nặng'),
            # ư tones
            'ừ': ('ư', 'huyền'), 'ứ': ('ư', 'sắc'),
            'ử': ('ư', 'hỏi'), 'ữ': ('ư', 'ngã'), 'ự': ('ư', 'nặng'),
            # y tones
            'ỳ': ('y', 'huyền'), 'ý': ('y', 'sắc'),
            'ỷ': ('y', 'hỏi'), 'ỹ': ('y', 'ngã'), 'ỵ': ('y', 'nặng'),
        }
        
        # Reverse mapping for tone addition
        self.reverse_tone_map = {}
        for toned, (base, tone) in self.tone_map.items():
            if base not in self.reverse_tone_map:
                self.reverse_tone_map[base] = {}
            self.reverse_tone_map[base][tone] = toned
    
    def remove_tones(self, text: str, use_pyvi: bool = True) -> str:
        """
        Remove all tone marks from Vietnamese text
        
        Args:
            text: Input text with tones
            use_pyvi: Use PyVi library if available
            
        Returns:
            Text without tone marks
        """
        if use_pyvi:
            try:
                from pyvi import ViUtils
                return ViUtils.remove_accents(text).decode('utf-8')
            except ImportError:
                pass
        
        # Fallback to manual removal
        result = []
        for char in text:
            if char in self.tone_map:
                base_char, _ = self.tone_map[char]
                result.append(base_char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def add_tones(self, text: str, use_pyvi: bool = True) -> str:
        """
        Add tone marks to Vietnamese text without tones
        
        Args:
            text: Input text without tones
            use_pyvi: Use PyVi library if available
            
        Returns:
            Text with tone marks added
        """
        if use_pyvi:
            try:
                from pyvi import ViUtils
                return ViUtils.add_accents(text)
            except ImportError:
                pass
        
        # This is a complex task - PyVi is recommended
        logger.warning("Tone addition without PyVi is limited")
        return text
    
    def analyze_word_tones(self, word: str) -> Dict[str, any]:
        """
        Analyze tone patterns in a word
        
        Args:
            word: Vietnamese word
            
        Returns:
            Dictionary with tone analysis
        """
        analysis = {
            'original': word,
            'base_form': '',
            'tones': [],
            'tone_positions': [],
            'tone_count': 0
        }
        
        base_chars = []
        for i, char in enumerate(word):
            if char in self.tone_map:
                base_char, tone = self.tone_map[char]
                base_chars.append(base_char)
                analysis['tones'].append(tone)
                analysis['tone_positions'].append(i)
            else:
                base_chars.append(char)
        
        analysis['base_form'] = ''.join(base_chars)
        analysis['tone_count'] = len(analysis['tones'])
        
        return analysis
    
    def analyze_text(self, text: str, tokens: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Analyze tone patterns in text
        
        Args:
            text: Input text
            tokens: Optional pre-tokenized text
            
        Returns:
            Comprehensive tone analysis
        """
        if tokens is None:
            tokens = text.split()
        
        analysis = {
            'total_words': len(tokens),
            'words_with_tones': 0,
            'words_without_tones': 0,
            'tone_distribution': defaultdict(int),
            'word_analyses': []
        }
        
        for token in tokens:
            word_analysis = self.analyze_word_tones(token)
            analysis['word_analyses'].append(word_analysis)
            
            if word_analysis['tone_count'] > 0:
                analysis['words_with_tones'] += 1
                for tone in word_analysis['tones']:
                    analysis['tone_distribution'][tone] += 1
            else:
                analysis['words_without_tones'] += 1
        
        # Convert defaultdict to regular dict
        analysis['tone_distribution'] = dict(analysis['tone_distribution'])
        
        return analysis
    
    def suggest_tone_corrections(self, word: str) -> List[str]:
        """
        Suggest possible tone corrections for a word
        
        Args:
            word: Word possibly missing tones
            
        Returns:
            List of suggestions
        """
        # This would require a dictionary or language model
        # For now, return empty list
        return []