from setuptools import setup
from setuptools import find_packages


setup(
    name="syfertext",
    author="Alan Aboudib",
    author_email="agabudeeb@gmail.com",
    description="A privacy preserving NLP framework",
    url="https://github.com/OpenMined/SyferText",
    keywords="nlp smpc secure multi-party computation federated learning deep learning artificial intelligence secure model sharin natural language processing spacy spaCy",
    classifier=["Programming Language :: Python :: 3.6", "Operating System :: OS Independent"],
    license="Apache-2.0",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mmh3==2.5.1",
        "syft @ git+https://github.com/OpenMined/PySyft@1eb369ae3a1865789f5809bec59f066ac1cbe58d",
        "requests==2.22.0",
    ],
    extras_require={
        "test": [
            "black>=19.10b0",
            "pytest>=5.4.3",
            "pytest-black>=0.3.10",
            "jupyter>=1.0.0",
            "papermill>=2.1.2",
        ]
    },
)
