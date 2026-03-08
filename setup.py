from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tft_pytorch",
    version="0.1.0",
    author="",
    description="PyTorch implementation of the Temporal Fusion Transformer (TFT) for time-series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "joblib>=1.1",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
