
"""
Evaluation module for Vietnamese NLP pipeline
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import pandas as pd

class PipelineEvaluator:
    """
    Comprehensive evaluation for Vietnamese NLP pipeline
    """
    
    def __init__(self, save_dir: str = 'data/results'):
        """
        Initialize evaluator
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.results = {}
        
    def evaluate_tokenization(self, 
                            predictions: List[List[str]], 
                            ground_truth: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate word segmentation performance
        
        Args:
            predictions: List of predicted token lists
            ground_truth: List of ground truth token lists
            
        Returns:
            Dictionary of metrics
        """
        total_exact_match = 0
        total_tp, total_fp, total_fn = 0, 0, 0

        for pred_tokens, true_tokens in zip(predictions, ground_truth):
            if pred_tokens == true_tokens:
                total_exact_match += 1

            pred_set = set(pred_tokens)
            true_set = set(true_tokens)

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'exact_match_rate': total_exact_match / len(predictions) if predictions else 0,
            'token_precision': precision,
            'token_recall': recall,
            'token_f1': f1,
            'total_true_tokens': total_tp + total_fn,
            'total_predicted_tokens': total_tp + total_fp
        }
        return metrics
    
    def evaluate_pos_tagging(self,
                           predictions: List[List[Tuple[str, str]]],
                           ground_truth: List[List[Tuple[str, str]]]) -> Dict[str, any]:
        """
        Evaluate POS tagging performance
        
        Args:
            predictions: List of predicted POS tag sequences
            ground_truth: List of ground truth POS tag sequences
            
        Returns:
            Dictionary containing evaluation metrics
        """
        all_pred_tags = []
        all_true_tags = []
        
        # Flatten the sequences
        for pred_seq, true_seq in zip(predictions, ground_truth):
            # Ensure same length
            if len(pred_seq) == len(true_seq):
                pred_tags = [tag for _, tag in pred_seq]
                true_tags = [tag for _, tag in true_seq]
                all_pred_tags.extend(pred_tags)
                all_true_tags.extend(true_tags)
        
        # Calculate metrics
        accuracy = accuracy_score(all_true_tags, all_pred_tags)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_tags, all_pred_tags, average='weighted'
        )
        
        # Per-tag metrics
        tag_precision, tag_recall, tag_f1, tag_support = precision_recall_fscore_support(
            all_true_tags, all_pred_tags, average=None
        )
        
        # Get unique tags
        unique_tags = sorted(list(set(all_true_tags + all_pred_tags)))
        
        # Create per-tag report
        per_tag_metrics = {}
        for i, tag in enumerate(unique_tags):
            if i < len(tag_precision):
                per_tag_metrics[tag] = {
                    'precision': tag_precision[i],
                    'recall': tag_recall[i],
                    'f1': tag_f1[i],
                    'support': tag_support[i] if i < len(tag_support) else 0
                }
        
        # Confusion matrix
        cm = confusion_matrix(all_true_tags, all_pred_tags, labels=unique_tags)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_tag_metrics': per_tag_metrics,
            'confusion_matrix': cm,
            'tag_labels': unique_tags
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], 
                            title: str = 'Confusion Matrix') -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Label names
            title: Plot title
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix.png', dpi=300)
        plt.close()
    
    def evaluate_tone_processing(self,
                               predictions: List[str],
                               ground_truth: List[str]) -> Dict[str, float]:
        """
        Evaluate tone processing (restoration) performance
        
        Args:
            predictions: Predicted text with tones
            ground_truth: Ground truth text with tones
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'exact_match_rate': 0,
            'character_accuracy': 0,
            'word_accuracy': 0,
            'tone_accuracy': 0
        }
        
        exact_matches = 0
        total_chars_correct = 0
        total_chars = 0
        total_words_correct = 0
        total_words = 0
        
        for pred, true in zip(predictions, ground_truth):
            # Exact match
            if pred == true:
                exact_matches += 1
            
            # Character-level accuracy
            for p_char, t_char in zip(pred, true):
                total_chars += 1
                if p_char == t_char:
                    total_chars_correct += 1
            
            # Word-level accuracy
            pred_words = pred.split()
            true_words = true.split()
            
            for p_word, t_word in zip(pred_words, true_words):
                total_words += 1
                if p_word == t_word:
                    total_words_correct += 1
        
        metrics['exact_match_rate'] = exact_matches / len(predictions)
        metrics['character_accuracy'] = total_chars_correct / total_chars if total_chars > 0 else 0
        metrics['word_accuracy'] = total_words_correct / total_words if total_words > 0 else 0
        
        return metrics
    
    def generate_report(self, all_metrics: Dict[str, any], 
                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            all_metrics: Dictionary containing all evaluation metrics
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("Vietnamese NLP Pipeline Evaluation Report")
        report.append("=" * 80)
        report.append("")
        
        # Tokenization metrics
        if 'tokenization' in all_metrics:
            report.append("## Tokenization Performance")
            report.append("-" * 40)
            tok_metrics = all_metrics['tokenization']
            report.append(f"Exact Match Rate: {tok_metrics.get('exact_match', 0):.4f}")
            report.append(f"Token F1 Score: {tok_metrics.get('token_f1', 0):.4f}")
            report.append("")
        
        # POS Tagging metrics
        if 'pos_tagging' in all_metrics:
            report.append("## POS Tagging Performance")
            report.append("-" * 40)
            pos_metrics = all_metrics['pos_tagging']
            report.append(f"Accuracy: {pos_metrics['accuracy']:.4f}")
            report.append(f"Weighted F1: {pos_metrics['f1']:.4f}")
            report.append(f"Weighted Precision: {pos_metrics['precision']:.4f}")
            report.append(f"Weighted Recall: {pos_metrics['recall']:.4f}")
            report.append("")
            
            # Per-tag metrics
            report.append("### Per-Tag Performance")
            report.append("-" * 40)
            
            # Create DataFrame for better formatting
            per_tag_df = pd.DataFrame(pos_metrics['per_tag_metrics']).T
            per_tag_df = per_tag_df.round(4)
            report.append(per_tag_df.to_string())
            report.append("")
        
        # Tone processing metrics
        if 'tone_processing' in all_metrics:
            report.append("## Tone Processing Performance")
            report.append("-" * 40)
            tone_metrics = all_metrics['tone_processing']
            report.append(f"Exact Match Rate: {tone_metrics['exact_match_rate']:.4f}")
            report.append(f"Character Accuracy: {tone_metrics['character_accuracy']:.4f}")
            report.append(f"Word Accuracy: {tone_metrics['word_accuracy']:.4f}")
            report.append("")
        
        # Processing time
        if 'processing_time' in all_metrics:
            report.append("## Processing Time Analysis")
            report.append("-" * 40)
            time_metrics = all_metrics['processing_time']
            total_time = sum(time_metrics.values())
            report.append(f"Total Processing Time: {total_time:.4f} seconds")
            for step, time in time_metrics.items():
                report.append(f"{step}: {time:.4f}s ({time/total_time*100:.1f}%)")
            report.append("")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str
