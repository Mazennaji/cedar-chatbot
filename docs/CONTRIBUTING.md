# 🤝 Contributing to Cedar Chatbot

Thank you for your interest in contributing to Cedar Chatbot!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cedar-chatbot.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Install dev dependencies: `make dev`

## Development Workflow

### Code Style

We use **Black** for formatting and **Ruff** for linting:

```bash
make format   # Auto-format
make lint     # Check for issues
```

### Testing

Always add tests for new features:

```bash
make test           # Run all tests
make test-fast      # Skip slow tests
```

Aim for >80% code coverage.

### Commit Messages

Follow conventional commits:

```
feat: add Egyptian Arabic dialect support
fix: normalize "2" correctly in word-initial position
docs: update API reference for feedback endpoint
test: add edge cases for sentiment analyzer
refactor: simplify memory sliding window logic
```

## Areas for Contribution

### High Priority
- **Lebanese dialogue dataset** — collect and curate training data
- **Arabic dialect detection** — distinguish Lebanese from Egyptian, Gulf, etc.
- **Model fine-tuning** — improve response quality on Arabic/Lebanese

### Medium Priority
- Telegram / WhatsApp bot integration
- PostgreSQL + Redis production backend
- Response caching for common queries
- Model quantization (ONNX / TensorRT)

### Good First Issues
- Add more Arabizi phrase mappings
- Improve sentiment lexicon coverage
- Add more test cases
- Documentation improvements

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure code is formatted: `make format`
3. Update documentation if needed
4. Write a clear PR description
5. Request review

## Code of Conduct

Be respectful, inclusive, and constructive. We're building something for the Lebanese community — let's make it welcoming for everyone.