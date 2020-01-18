from setuptools import setup
from setuptools import find_packages


setup(
    name = 'syfertext',
    author = 'Alan Aboudib',
    author_email = 'agabudeeb@gmail.com',
    description = 'A privacy preserving NLP framework',
    url = 'https://github.com/OpenMined/SyferText',
    keywords = 'nlp smpc secure multi-party computation federated learning deep learning artificial intelligence secure model sharin natural language processing spacy spaCy',
    classifier = ['Programming Language :: Python :: 3.6', 'Operating System :: OS Independent'],
    license = 'Apache-2.0',
    version = '0.1.0.dev1',
    packages = find_packages(),
    install_requires = [
        'tqdm==4.36.1',
        'mmh3==2.5.1',
        'syft==0.2.1a1',
        'requests==2.22.0'
    ],

    dependency_links = [
        'git+https://github.com/OpenMined@1bc16b402dc04912406295e0a6feec0092573235#egg=syft-0.2.1a1'
    ]
)
