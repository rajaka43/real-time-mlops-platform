from setuptools import setup, find_packages

setup(
    name="mlops-platform",
    version="1.0.0",
    description="Production-grade MLOps Prediction Platform",
    author="MLOps Intern",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "pydantic==2.5.3",
        "scikit-learn==1.4.0",
        "numpy==1.26.3",
        "pandas==2.1.4",
        "scipy==1.11.4",
        "optuna==3.5.0",
        "mlflow==2.10.0",
        "python-dotenv==1.0.0",
        "httpx==0.26.0",
    ],
)
