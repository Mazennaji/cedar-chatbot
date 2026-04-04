"""Cedar Chatbot — Package Setup."""

from setuptools import setup, find_packages

setup(
    name="cedar-chatbot",
    version="1.0.0",
    description="A trilingual (EN/AR/LB) context-aware chatbot with RLHF",
    author="Cedar Team",
    author_email="cedar@example.com",
    url="https://github.com/Mazennaji/cedar-chatbot",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "django>=5.0",
        "streamlit>=1.29.0",
        "langdetect>=1.0.9",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)