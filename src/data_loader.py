"""
Data loading utilities for Vietnamese NLP datasets
"""
import os
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VietnameseDatasetLoader:
    """
    Load and prepare Vietnamese NLP datasets
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.datasets = {}
        
    def load_student_feedback(self) -> Dict[str, pd.DataFrame]:
        """
        Load UIT Vietnamese Students' Feedback dataset from HuggingFace
        
        Returns:
            Dictionary with train/validation/test DataFrames
        """
        logger.info("Loading Vietnamese Students' Feedback dataset...")
        
        dataset = load_dataset("uitnlp/vietnamese_students_feedback")
        
        data_splits = {}
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                df = pd.DataFrame(dataset[split])
                data_splits[split] = df
                logger.info(f"{split}: {len(df)} samples")
        
        self.datasets['student_feedback'] = data_splits
        return data_splits
    
    def load_vlsp2013(self, vlsp_path: Optional[str] = None) -> Dict[str, List]:
        """
        Load VLSP 2013 word segmentation dataset
        
        Args:
            vlsp_path: Path to VLSP 2013 data
            
        Returns:
            Dictionary with train/test data
        """
        if vlsp_path is None:
            vlsp_path = os.path.join(self.data_dir, 'raw', 'vlsp2013')
        
        data_splits = {}
        
        # Load training data
        train_file = os.path.join(vlsp_path, 'train.txt')
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = [line.strip() for line in f if line.strip()]
            data_splits['train'] = train_data
            logger.info(f"VLSP 2013 train: {len(train_data)} sentences")
        
        # Load test data
        test_file = os.path.join(vlsp_path, 'test.txt')
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = [line.strip() for line in f if line.strip()]
            data_splits['test'] = test_data
            logger.info(f"VLSP 2013 test: {len(test_data)} sentences")
        
        self.datasets['vlsp2013'] = data_splits
        return data_splits
    
    def load_social_media_corpus(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load social media corpus (Facebook comments, etc.)
        
        Args:
            file_path: Path to social media data
            
        Returns:
            DataFrame with social media text
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'raw', 'social_media.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            logger.info(f"Social media corpus: {len(df)} samples")
            self.datasets['social_media'] = df
            return df
        else:
            logger.warning(f"Social media corpus not found at {file_path}")
            return pd.DataFrame()
    
    def prepare_pos_tagging_data(self, sentences: List[str], 
                                tagged_sentences: List[List[Tuple[str, str]]]) -> pd.DataFrame:
        """
        Prepare data for POS tagging evaluation
        
        Args:
            sentences: Raw sentences
            tagged_sentences: Tagged sentences with (word, pos) tuples
            
        Returns:
            DataFrame with prepared data
        """
        data = []
        for sent, tagged in zip(sentences, tagged_sentences):
            tokens = [token for token, _ in tagged]
            tags = [tag for _, tag in tagged]
            data.append({
                'sentence': sent,
                'tokens': tokens,
                'tags': tags,
                'num_tokens': len(tokens)
            })
        
        return pd.DataFrame(data)
    
    def create_test_samples(self) -> Dict[str, List[str]]:
        """
        Create diverse test samples for pipeline testing
        
        Returns:
            Dictionary of test samples by category
        """
        test_samples = {
            'simple': [
                "Tôi đi học",
                "Hôm nay trời đẹp",
                "Cảm ơn bạn nhiều",
                "Việt Nam tươi đẹp"
            ],
            'complex': [
                "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò",
                "Trường Đại học Bách khoa Hà Nội công bố điểm chuẩn năm 2024",
                "Công ty Cổ phần Công nghệ ABC vừa ra mắt sản phẩm mới",
                "Hội nghị thượng đỉnh ASEAN diễn ra tại Thủ đô Hà Nội"
            ],
            'social_media': [
                "e ơi cho a hỏi cái này bao nhieu tien vay",
                "sp nay xai ok ko ad",
                "minh da mua hang nhung chua nhan dc",
                "Quá tuyệt vời luôn ạ 😍😍😍"
            ],
            'missing_tones': [
                "toi muon hoc tieng viet",
                "truong dai hoc bach khoa ha noi",
                "cam on ban rat nhieu",
                "chuc mung nam moi"
            ],
            'mixed_language': [
                "Tôi đang học Machine Learning",
                "Download app về điện thoại nhé",
                "Meeting lúc 2h chiều nay",
                "Update status mới trên Facebook"
            ]
        }
        
        return test_samples