from glob import glob
from os.path import basename
from os.path import splitext

import setuptools

setuptools.setup(
    name="dsconcept",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    url="https://developer.nasa.gov/DataSquad/classifier_scripts",
    author="Anthony Buonomo",
    author_email="anthony.r.buonomo@nasa.gov",
    description="Scripts for processing, topic modeling, and creating classifiers for STI concepts.",
    long_description=open("README.md").read(),
    license="MIT",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    install_requires=[
        "scikit-learn>=0.21.3",
        "spacy>=2.2.3",
        "numpy>=1.17.4",
        "pandas>=0.25.3",
        "pyLDAvis>=2.1.2",
        "textacy==0.9.1",
        "boto3>=1.7.46",
        "dask>=2.8.1",
        "PyYAML>=5.1.2",
        "h5py>=2.10.0",
        "tqdm>=4.39.0",
    ],
    classifiers=[
        "Development Status :: 2 - Beta",
        "Programming Language :: Python :: 3.6",
    ],
)
