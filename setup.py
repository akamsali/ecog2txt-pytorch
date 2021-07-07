import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecog2txt-pytorch",
    version="0.0.1",
    author="Akshita Ramya",
    author_email="akshitakamsali@gmail.com",
    description="Code for decoding speech as text from neural data using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akshitark/ecog2txt-pytorch",
    packages=setuptools.find_packages(),
    package_data={
        'ecog2txt_pytorch': [
            'conf/block_breakdowns.json',
            'conf/mocha-1_word_sequence.yaml'
            'conf/vocab.mocha-timit.1806',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
        'tikzplotlib', 'torch', 'hickle',
        'python_speech_features', 'pyyaml', 'protobuf>=3.7',
        'tfrecord', 'tfrecord_lite', 'wandb',
    ],
)
