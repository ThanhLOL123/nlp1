"""
Unit tests for Vietnamese NLP pipeline
"""
import pytest
from pathlib import Path
from src.pipeline import VietnameseNLPPipeline
from src.tokenizer import VietnameseTokenizer
from src.pos_tagger import VietnamesePOSTagger
from src.tone_processor import ToneProcessor

class TestPipeline:
    """Test cases for the main pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
        return VietnameseNLPPipeline(config_path=str(config_path))
    
    def test_simple_sentence(self, pipeline):
        """Test processing simple sentence"""
        result = pipeline.process("Tôi đi học")
        
        assert len(result.tokens) == 3
        assert result.tokens == ["Tôi", "đi", "học"]
        assert len(result.pos_tags) == 3
    
    def test_empty_input(self, pipeline):
        """Test handling empty input"""
        result = pipeline.process("")
        
        assert result.tokens == []
        assert result.pos_tags == []
    
    def test_special_characters(self, pipeline):
        """Test handling special characters"""
        result = pipeline.process("Email: test@example.com!")
        
        assert "test@example.com" in ' '.join(result.tokens)

class TestTokenizer:
    """Test cases for tokenizer"""
    
    def test_underthesea_tokenizer(self):
        """Test Underthesea tokenizer"""
        tokenizer = VietnameseTokenizer('underthesea')
        tokens = tokenizer.tokenize("Việt Nam tươi đẹp")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_pyvi_tokenizer(self):
        """Test PyVi tokenizer"""
        tokenizer = VietnameseTokenizer('pyvi')
        tokens = tokenizer.tokenize("Hà Nội mùa thu")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0

class TestToneProcessor:
    """Test cases for tone processor"""
    
    def test_remove_tones(self):
        """Test tone removal"""
        processor = ToneProcessor()
        
        test_cases = [
            ("Việt Nam", "Viet Nam"),
            ("Hôm nay", "Hom nay"),
            ("Cảm ơn", "Cam on")
        ]
        
        for original, expected in test_cases:
            result = processor.remove_tones(original)
            assert result == expected