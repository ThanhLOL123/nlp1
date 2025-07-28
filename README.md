# Vietnamese NLP Pipeline

A comprehensive, modular pipeline for Vietnamese Natural Language Processing that provides state-of-the-art text processing capabilities with multiple backend support and extensive evaluation tools.

## 🚀 Features

### Core NLP Components
- **Advanced Tokenization**: Multiple algorithms (Underthesea, PyVi, Hybrid) for robust Vietnamese word segmentation
- **POS Tagging**: Part-of-speech tagging with standardized universal tags and multiple backend support
- **Tone Processing**: Complete Vietnamese tone analysis, removal, and restoration capabilities
- **Text Preprocessing**: Intelligent normalization for social media, email, URL, and general text cleaning

### Pipeline Architecture
- **Modular Design**: Easily swappable components for different use cases
- **Configuration-Driven**: YAML-based configuration for flexible pipeline customization  
- **Multiple Backends**: Support for Underthesea, PyVi, and Transformer-based models
- **Performance Monitoring**: Built-in timing and resource usage tracking

### Evaluation & Benchmarking
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall for all components
- **Dataset Support**: VLSP 2013, VLSP 2016, UIT Vietnamese Students' Feedback
- **Comparative Analysis**: Side-by-side performance comparison of different methods
- **Visualization**: Performance charts and confusion matrices

### Data Processing
- **Multiple Dataset Loaders**: Built-in support for major Vietnamese NLP datasets
- **Test Sample Generation**: Automatic creation of test cases across different text categories
- **Batch Processing**: Efficient processing of large text corpora

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vietnamese-nlp-pipeline.git
   cd vietnamese-nlp-pipeline
   ```

2. **Create and activate a virtual environment:**
   
   **Windows:**
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets (optional):**
   ```bash
   python scripts/download_data.py
   ```
   > Note: Some datasets like VLSP require manual download due to licensing requirements

## 🎯 Quick Start

### Pipeline API Usage

```python
from src.pipeline import VietnameseNLPPipeline

# Initialize with configuration
pipeline = VietnameseNLPPipeline('configs/config.yaml')

# Process Vietnamese text
text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
result = pipeline.process(text, verbose=True)

# Access results
print("Original:", result.original_text)
print("Tokens:", result.tokens)
print("POS Tags:", result.pos_tags)
print("Tone Info:", result.tone_info)
print("Processing Time:", result.processing_time)
```

### Command Line Interface

**Demo Mode** - Test on sample sentences:
```bash
python main.py --mode demo
```

**Example Output:**
```
============================================================
Category: SIMPLE
============================================================

Input: Tôi đi học
Tokens: Tôi | đi | học
POS Tags: [('Tôi', 'P'), ('đi', 'V'), ('học', 'V')]

Input: Hôm nay trời đẹp
Tokens: Hôm nay | trời | đẹp
POS Tags: [('Hôm nay', 'N'), ('trời', 'N'), ('đẹp', 'A')]

============================================================
Category: COMPLEX
============================================================

Input: Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò
Tokens: Chàng | trai | 9X | Quảng Trị | khởi nghiệp | từ | nấm | sò
POS Tags: [('Chàng', 'Nc'), ('trai', 'N'), ('9X', 'N'), ('Quảng Trị', 'Np'), ('khởi nghiệp', 'V'), ('từ', 'E'), ('nấm', 'N'), ('sò', 'M')]

============================================================
Category: SOCIAL_MEDIA
============================================================

Input: e ơi cho a hỏi cái này bao nhieu tien vay
Tokens: e | ơi | cho | a | hỏi | cái | này | bao | nhieu | tien | vay
POS Tags: [('e', 'V'), ('ơi', 'I'), ('cho', 'E'), ('a', 'N'), ('hỏi', 'V'), ('cái', 'Nc'), ('này', 'P'), ('bao', 'N'), ('nhieu', 'N'), ('tien', 'V'), ('vay', 'V')]

============================================================
Category: MISSING_TONES
============================================================

Input: toi muon hoc tieng viet
Tokens: toi | muon | hoc | tieng | viet
POS Tags: [('toi', 'N'), ('muon', 'V'), ('hoc', 'N'), ('tieng', 'N'), ('viet', 'M')]
Restored: Tôi muốn học tiếng Việt

============================================================
Category: MIXED_LANGUAGE
============================================================

Input: Tôi đang học Machine Learning
Tokens: Tôi | đang | học | Machine | Learning
POS Tags: [('Tôi', 'P'), ('đang', 'R'), ('học', 'V'), ('Machine', 'Np'), ('Learning', 'Np')]
```

**Evaluation Mode** - Evaluate on specific datasets:
```bash
python main.py --mode evaluate --dataset vlsp2013 --vlsp-path ./data/raw/vlsp2013
```

**Benchmark Mode** - Compare different methods:
```bash
python main.py --mode benchmark
```

### Jupyter Notebook

Explore the interactive demo:
```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

## ⚙️ Configuration

Configure the pipeline behavior through `configs/config.yaml`:

```yaml
pipeline:
  tokenizer:
    method: "hybrid"  # underthesea, pyvi, vncorenlp, hybrid
    
  pos_tagger:
    method: "underthesea"  # underthesea, pyvi, transformer
    
  tone_processor:
    remove_method: "pyvi"
    add_method: "pyvi"
    analyze_patterns: true

data:
  vlsp2013_path: "data/raw/vlsp2013"
  vlsp2016_path: "data/raw/vlsp2016"
  
evaluation:
  metrics: [accuracy, f1_score, precision, recall]
  save_confusion_matrix: true
```

## 🧪 Component Usage

### Individual Components

**Tokenizer:**
```python
from src.tokenizer import VietnameseTokenizer

tokenizer = VietnameseTokenizer('hybrid')
tokens = tokenizer.tokenize("Việt Nam tươi đẹp")
```

**POS Tagger:**
```python
from src.pos_tagger import VietnamesePOSTagger

tagger = VietnamesePOSTagger('underthesea')
pos_tags = tagger.tag(['Việt', 'Nam', 'tươi', 'đẹp'])
```

**Tone Processor:**
```python
from src.tone_processor import ToneProcessor

processor = ToneProcessor()
tone_info = processor.analyze_text("Xin chào")
no_tones = processor.remove_tones("Xin chào")
```

## 📊 Supported Datasets

- **VLSP 2013**: Vietnamese word segmentation dataset
- **VLSP 2016**: Extended Vietnamese NLP tasks  
- **UIT Vietnamese Students' Feedback**: Sentiment analysis dataset (auto-downloaded)
- **Custom datasets**: Easy integration through the data loader API

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📈 Performance

The pipeline provides detailed performance metrics:
- **Processing Speed**: Component-level timing analysis
- **Memory Usage**: Resource consumption monitoring  
- **Accuracy Metrics**: F1-score, precision, recall for all tasks
- **Comparative Analysis**: Side-by-side method comparison

## ⚠️ Current Limitations

### Technical Limitations
- **Language Support**: Currently focused only on Vietnamese language processing
- **Model Dependencies**: Relies on external libraries (Underthesea, PyVi) for core functionality
- **Memory Usage**: Large transformer models may require significant RAM (8GB+ recommended)
- **Processing Speed**: Sequential processing may be slow for large document collections

### Dataset Limitations
- **VLSP Datasets**: Require manual download due to licensing restrictions
- **Domain Specificity**: Best performance on formal Vietnamese text; informal/social media text may have reduced accuracy
- **Text Length**: Very long documents (>10k characters) may experience performance degradation

### Feature Limitations
- **Real-time Processing**: Not optimized for real-time streaming applications
- **GPU Acceleration**: Limited GPU support for transformer-based components
- **Custom Models**: No built-in training capabilities for domain-specific models
- **Language Detection**: No automatic Vietnamese text detection

## 🚀 Future Features

### Planned Enhancements

#### **Version 2.0 - Advanced Processing**
- **🔥 Neural Models**: Integration of Vietnamese BERT, PhoBERT, and other transformer models
- **⚡ GPU Acceleration**: CUDA support for faster processing with transformer models
- **🌊 Streaming API**: Real-time text processing capabilities
- **📱 Mobile Support**: Lightweight models for mobile and edge deployment

#### **Version 2.2 - Advanced Features**
- **🧠 Named Entity Recognition**: Person, location, organization identification
- **💭 Sentiment Analysis**: Emotion and opinion detection
- **📊 Text Classification**: Document categorization and topic modeling
- **🔗 Dependency Parsing**: Syntactic relationship analysis

#### **Version 2.3 - Enterprise Features**
- **☁️ Cloud API**: RESTful API for cloud deployment
- **📈 Analytics Dashboard**: Web-based performance monitoring
- **🔧 Auto-tuning**: Automatic hyperparameter optimization
- **📦 Docker Support**: Containerized deployment options

### Research & Development
- **📚 Custom Training**: Framework for training domain-specific models
- **🔬 Active Learning**: Semi-supervised learning for data annotation
- **🎨 Data Augmentation**: Automatic Vietnamese text augmentation techniques
- **🔍 Explainable AI**: Model interpretability and explanation features

### Community Features
- **👥 Model Hub**: Community-contributed models and datasets
- **📖 Documentation**: Interactive tutorials and API documentation
- **🎓 Educational**: Vietnamese NLP learning resources and examples
- **🏆 Benchmarks**: Standardized evaluation on Vietnamese NLP tasks

## 📁 Project Structure

```
vietnamese-nlp-pipeline/
├── 📁 configs/                    # Configuration files
│   └── config.yaml               # Main pipeline configuration
├── 📁 data/                      # Data storage
│   ├── 📁 raw/                   # Original datasets
│   ├── 📁 processed/             # Processed datasets  
│   └── 📁 results/               # Evaluation results and visualizations
│       ├── benchmark_results.csv
│       ├── benchmark_comparison.png
│       └── evaluation_report.txt
├── 📁 notebooks/                 # Jupyter notebooks
│   └── 01_pipeline_demo.ipynb   # Interactive pipeline demonstration
├── 📁 scripts/                   # Utility scripts
│   └── download_data.py          # Dataset download helper
├── 📁 src/                       # Core source code
│   ├── pipeline.py               # Main pipeline orchestrator
│   ├── tokenizer.py              # Vietnamese tokenization
│   ├── pos_tagger.py             # Part-of-speech tagging
│   ├── tone_processor.py         # Vietnamese tone processing
│   ├── preprocessor.py           # Text preprocessing utilities
│   ├── evaluator.py              # Performance evaluation
│   └── data_loader.py            # Dataset loading utilities
├── 📁 tests/                     # Unit and integration tests
│   └── test_pipeline.py          # Pipeline test cases
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Advanced Usage

### Custom Model Integration

```python
# Add custom tokenizer
from src.tokenizer import VietnameseTokenizer

class CustomTokenizer(VietnameseTokenizer):
    def custom_tokenize(self, text):
        # Your custom tokenization logic
        return tokens

# Use in pipeline
pipeline.tokenizer = CustomTokenizer('custom')
```

### Batch Processing

```python
texts = ["Text 1", "Text 2", "Text 3"]
results = []

for text in texts:
    result = pipeline.process(text)
    results.append(result)
    
# Analyze batch results
processing_times = [r.processing_time for r in results]
```

### Performance Analysis

```python
from src.evaluator import PipelineEvaluator

evaluator = PipelineEvaluator('data/results')

# Evaluate tokenization
predictions = [result.tokens for result in results]
ground_truth = [...]  # Your ground truth data

metrics = evaluator.evaluate_tokenization(predictions, ground_truth)
print(f"Tokenization F1: {metrics['f1']:.3f}")
```

## 🔧 Dependencies

### Core Libraries
- **underthesea** (6.7.0): Vietnamese NLP toolkit
- **pyvi** (0.1.1): Vietnamese text processing  
- **transformers** (4.36.0): Transformer-based models
- **torch** (2.1.0): Deep learning framework

### Data & Analysis
- **pandas** (2.1.4): Data manipulation
- **numpy** (1.24.3): Numerical computing
- **scikit-learn** (1.3.2): Machine learning metrics
- **matplotlib** (3.8.2): Plotting and visualization
- **seaborn** (0.13.0): Statistical visualization

### Development
- **pytest** (7.4.3): Testing framework
- **jupyter** (1.0.0): Interactive notebooks
- **pyyaml** (6.0.1): Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Underthesea](https://github.com/undertheseanlp/underthesea): Vietnamese NLP Toolkit
- [PyVi](https://github.com/trungtv/pyvi): Vietnamese tokenizer
- [VLSP Shared Tasks](https://vlsp.org.vn/): Vietnamese Language and Speech Processing
- [UIT-ViSFD](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback): Vietnamese Students' Feedback Dataset

