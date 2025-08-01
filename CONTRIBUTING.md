# Contributing to Email Classification Project

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repository and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

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

3. Install development dependencies:
```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Run tests to ensure everything is working:
```bash
pytest tests/ -v
```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Before submitting a PR, ensure your code passes all checks:

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

## Testing

We use pytest for testing. Please ensure:

- All new code has corresponding tests
- Tests are meaningful and cover edge cases
- All tests pass before submitting PR

Run tests with:
```bash
pytest tests/ -v --cov=src
```

## Adding New Features

When adding new features:

1. Create a new branch for your feature
2. Add comprehensive tests
3. Update documentation if needed
4. Ensure all existing tests still pass
5. Add type hints to new functions
6. Follow the existing code structure

## Project Structure

```
src/
├── data/           # Data processing utilities
├── features/       # Feature engineering
├── models/         # Model implementations
├── evaluation/     # Evaluation metrics and plots
└── api/           # FastAPI service

tests/             # Test suite
config/            # Configuration files
docs/              # Documentation
```

## Documentation

We use docstrings for code documentation. Please:

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include parameter types and return types
- Provide examples where helpful

Example:
```python
def process_email(text: str, clean: bool = True) -> str:
    """
    Process email text for classification.
    
    Args:
        text: Raw email text to process
        clean: Whether to apply text cleaning
        
    Returns:
        Processed email text
        
    Example:
        >>> process_email("Hello World!")
        "hello world"
    """
```

## Reporting Issues

We use GitHub issues to track bugs. Report a bug by opening a new issue with:

- A quick summary and/or background
- Steps to reproduce (be specific!)
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

## Feature Requests

We welcome feature requests! Please:

- Explain the motivation for the feature
- Describe the proposed solution
- Consider alternatives you've explored
- Provide examples if applicable

## Code of Conduct

Be respectful and inclusive. We want this to be a welcoming environment for everyone.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.
