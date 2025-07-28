
"""
Main script to run Vietnamese NLP pipeline
"""
import argparse
import logging
import time
from pathlib import Path

from src.pipeline import VietnameseNLPPipeline
from src.evaluator import PipelineEvaluator
from src.data_loader import VietnameseDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """Main execution function"""
    
    # Get the absolute path to the directory containing main.py
    script_dir = Path(__file__).resolve().parent
    
    # Make paths absolute
    config_path = script_dir / args.config
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    vlsp_path = script_dir / args.vlsp_path if args.vlsp_path else None

    # Initialize components
    logger.info("Initializing Vietnamese NLP Pipeline...")
    pipeline = VietnameseNLPPipeline(config_path)
    evaluator = PipelineEvaluator(output_dir)
    data_loader = VietnameseDatasetLoader(data_dir)
    
    # Load test samples
    test_samples = data_loader.create_test_samples()
    
    if args.mode == 'demo':
        # Run demo on test samples
        logger.info("Running pipeline demo...")
        
        for category, samples in test_samples.items():
            print(f"\n{'='*60}")
            print(f"Category: {category.upper()}")
            print('='*60)
            
            for text in samples[:3]:  # Show first 3 samples
                print(f"\nInput: {text}")
                result = pipeline.process(text, verbose=args.verbose)
                
                print(f"Tokens: {' | '.join(result.tokens)}")
                print(f"POS Tags: {result.pos_tags}")
                
                if category == 'missing_tones':
                    # Try tone restoration
                    from src.tone_processor import ToneProcessor
                    tp = ToneProcessor()
                    restored = tp.add_tones(text)
                    print(f"Restored: {restored}")
    
    elif args.mode == 'evaluate':
        # Run full evaluation
        logger.info("Running evaluation...")
        
        # Load datasets
        if args.dataset == 'student_feedback':
            data_splits = data_loader.load_student_feedback()
            # Run evaluation on test set
            test_data = data_splits['test']
            
            # Process and evaluate
            results = evaluate_on_dataset(pipeline, evaluator, test_data)
            
        elif args.dataset == 'vlsp2013':
            data_splits = data_loader.load_vlsp2013(vlsp_path)
            if 'test' not in data_splits:
                logger.error("VLSP 2013 test data not found. Please ensure it's downloaded and placed correctly.")
                return
            
            results = evaluate_vlsp_tokenization(pipeline, evaluator, data_splits['test'])
            
        # Generate report
        report = evaluator.generate_report(results, 
                                         save_path=output_dir / "evaluation_report.txt")
        print(report)
    
    elif args.mode == 'benchmark':
        # Benchmark different methods
        logger.info("Running benchmark comparison...")
        
        benchmark_results = run_benchmark_comparison(test_samples, output_dir)
        save_benchmark_results(benchmark_results, output_dir)


def evaluate_on_dataset(pipeline, evaluator, dataset):
    """
    Evaluate pipeline on a given dataset.
    Since the student_feedback dataset only has sentences and sentiment labels,
    we can't evaluate tokenization or POS tagging accuracy with it.
    Instead, we will run the pipeline on the sentences and gather processing time metrics.
    """
    logger.info(f"Running pipeline on {len(dataset)} sentences to collect metrics...")
    
    all_metrics = {}
    processing_times = {
        'preprocessing': [],
        'tokenization': [],
        'pos_tagging': [],
        'tone_processing': []
    }

    # The dataset is a pandas DataFrame, we iterate through the 'sentence' column
    for text in dataset['sentence']:
        try:
            result = pipeline.process(text)
            for component, time_val in result.processing_time.items():
                if component in processing_times:
                    processing_times[component].append(time_val)
        except Exception as e:
            logger.error(f"Error processing text: '{text}' - {e}")

    # Calculate total time for each component
    total_times = {k: sum(v) for k, v in processing_times.items()}
    all_metrics['processing_time'] = total_times
    
    logger.info("Evaluation processing finished.")
    return all_metrics

def evaluate_vlsp_tokenization(pipeline, evaluator, vlsp_test_data):
    """
    Evaluates tokenization performance on the VLSP 2013 dataset.
    """
    logger.info("Evaluating tokenization on VLSP 2013 dataset...")
    
    predicted_tokens = []
    ground_truth_tokens = []

    for line in vlsp_test_data:
        # VLSP 2013 format: tokens are separated by spaces, words by underscores
        # Example: "Tôi đi học" -> "Tôi đi học" (raw text)
        # Ground truth: "Tôi đi học" -> ["Tôi", "đi", "học"]
        # The VLSP data is already tokenized with spaces, so we need to split it.
        # However, the pipeline's tokenizer expects raw text.
        # So, we need to convert the VLSP format to raw text for the pipeline,
        # and then convert the VLSP format to a list of tokens for ground truth.

        # Convert VLSP format to raw text for pipeline input
        raw_text = line.replace("_", " ") # Remove underscores for raw text input

        # Get predicted tokens from the pipeline
        pipeline_result = pipeline.process(raw_text)
        predicted_tokens.append(pipeline_result.tokens)

        # Get ground truth tokens from VLSP format
        gt_tokens = line.split(" ")
        ground_truth_tokens.append(gt_tokens)
    
    tokenization_metrics = evaluator.evaluate_tokenization(predicted_tokens, ground_truth_tokens)
    
    all_metrics = {
        'tokenization': tokenization_metrics,
        'processing_time': {} # We don't track detailed processing time for this specific evaluation
    }
    
    logger.info("VLSP 2013 tokenization evaluation finished.")
    return all_metrics

def run_benchmark_comparison(test_samples, output_dir):
    """Compare different tokenization and POS tagging methods"""
    import pandas as pd
    
    methods = {
        'tokenizers': ['underthesea', 'pyvi', 'hybrid'],
        'pos_taggers': ['underthesea', 'pyvi']
    }
    
    results = []
    
    for tok_method in methods['tokenizers']:
        for pos_method in methods['pos_taggers']:
            logger.info(f"Testing {tok_method} tokenizer + {pos_method} POS tagger")
            
            # Create pipeline with specific methods
            pipeline = VietnameseNLPPipeline()
            pipeline.tokenizer.method = tok_method
            pipeline.pos_tagger.method = pos_method
            
            # Measure performance
            start_time = time.time()
            
            for category, samples in test_samples.items():
                for text in samples:
                    try:
                        result = pipeline.process(text)
                    except Exception as e:
                        logger.error(f"Error processing '{text}': {e}")
            
            elapsed_time = time.time() - start_time
            
            results.append({
                'tokenizer': tok_method,
                'pos_tagger': pos_method,
                'total_time': elapsed_time,
                'avg_time_per_sample': elapsed_time / sum(len(s) for s in test_samples.values())
            })
    
    return pd.DataFrame(results)

def save_benchmark_results(results_df, output_dir):
    """Save benchmark results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    results_df.to_csv(output_path / 'benchmark_results.csv', index=False)
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot processing time comparison
    results_pivot = results_df.pivot(index='tokenizer', 
                                     columns='pos_tagger', 
                                     values='avg_time_per_sample')
    results_pivot.plot(kind='bar', ax=ax)
    
    ax.set_title('Average Processing Time Comparison')
    ax.set_xlabel('Tokenizer Method')
    ax.set_ylabel('Time per Sample (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'benchmark_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese NLP Pipeline")
    
    parser.add_argument('--mode', choices=['demo', 'evaluate', 'benchmark'],
                       default='demo', help='Execution mode')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory')
    parser.add_argument('--output-dir', default='data/results',
                       help='Output directory for results')
    parser.add_argument('--dataset', choices=['student_feedback', 'vlsp2013', 'social_media'],
                       help='Dataset to evaluate on')
    parser.add_argument('--vlsp-path', help='Path to VLSP data')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    main(args)
