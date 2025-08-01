# Email Classification Project

A machine learning project for classifying emails using various ML algorithms including CatBoost, LightGBM, and Sentence Transformers.

## Project Overview

This project implements an email classification system that can automatically categorize emails based on their content. The system uses advanced NLP techniques and ensemble learning methods to achieve high accuracy.

## Features

- Multiple ML algorithms (CatBoost, LightGBM, Scikit-learn)
- NLP preprocessing with NLTK
- Sentence embeddings with Sentence Transformers
- Hyperparameter optimization with Optuna
- FastAPI web service for model serving
- Comprehensive evaluation metrics and visualization

## Project Structure

```
toy_mail_clasification/
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── pyproject.toml                         # Project configuration
├── .gitignore                             # Git ignore patterns
├── train.py                               # Model training script
├── evaluate.py                            # Model evaluation script
├── src/                                   # Source code
│   ├── __init__.py
│   ├── data/                              # Data processing modules
│   │   ├── __init__.py
│   │   └── preprocessing.py               # Data preprocessing utilities
│   ├── models/                            # Model definitions
│   │   ├── __init__.py
│   │   ├── base_model.py                  # Base model interface
│   │   └── ensemble.py                    # Ensemble model implementations
│   ├── features/                          # Feature engineering
│   │   ├── __init__.py
│   │   └── text_features.py               # Text feature extraction
│   ├── evaluation/                        # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py                     # Evaluation metrics
│   └── api/                               # FastAPI service
│       ├── __init__.py
│       ├── main.py                        # FastAPI app
│       └── schemas.py                     # Pydantic models
├── tests/                                 # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── data/                                  # Data directory
│   └── dataset_procesado-gama-mayo-v4.csv # Training dataset
├── models/                                # Saved models
├── notebooks/                             # Jupyter notebooks for exploration
└── docs/                                  # Additional documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd toy_mail_clasification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python train.py --config config/train_config.yaml
```

### Evaluating a Model

```bash
python evaluate.py --model-path models/best_model.pkl --data-path data/test_data.csv
```

### Running the API Service

```bash
uvicorn src.api.main:app --reload
```

## Configuration

The project uses YAML configuration files for easy parameter management. See `config/` directory for examples.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for any API changes
- Use type hints throughout the codebase
- Ensure all tests pass before submitting PRs

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries include:

- **CatBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting framework
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **Sentence Transformers**: Sentence embeddings
- **FastAPI**: Web framework for API
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the repository.
