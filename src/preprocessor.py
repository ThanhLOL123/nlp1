"""
Vietnamese text preprocessing module
"""
import re
import unicodedata
from typing import Optional

class VietnamesePreprocessor:
    """
    Preprocessor for Vietnamese text
    Handles normalization, cleaning, and standardization
    """
    
    def __init__(self):
        # Vietnamese character mappings
        self.char_map = {
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 
            'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 
            'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ',
            'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 
            'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ'
        }
        
        # Common typos and corrections
        self.typo_dict = {
            'tôi': ['toi', 'toii', 'tooi'],
            'không': ['khong', 'khôg', 'ko', 'k'],
            'được': ['duoc', 'đc', 'dc'],
            'người': ['nguoi', 'ngừoi', 'ng'],
        }
        
    def normalize(self, text: str, fix_typos: bool = False) -> str:
        """
        Normalize Vietnamese text
        
        Args:
            text: Input text
            fix_typos: Whether to fix common typos
            
        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Fix common character issues
        for old, new in self.char_map.items():
            text = text.replace(old, new)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Fix typos if requested
        if fix_typos:
            text = self._fix_typos(text)
        
        return text
    
    def _fix_typos(self, text: str) -> str:
        """Fix common Vietnamese typos"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected = word
            for correct, typos in self.typo_dict.items():
                if word.lower() in typos:
                    corrected = correct
                    break
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)
    
    def clean_social_media_text(self, text: str) -> str:
        """
        Clean social media specific elements
        
        Args:
            text: Social media text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (optional - keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle repeated characters (e.g., "đẹpppppp" -> "đẹp")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Normalize emoji (remove or replace)
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        
        return self.normalize(text)
    
    def segment_sentences(self, text: str) -> list[str]:
        """Segment text into sentences"""
        # Simple sentence segmentation for Vietnamese
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences