import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dasp",
    version="0.0.2",
    author="Marco Ancona",
    author_email="marco.ancona@inf.ethz.ch",
    description="Implementation of Deep Approximate Shapley Propagation for Tensorflow/Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcoancona/DASP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "lpdn @ git+https://github.com/marcoancona/lpdn@master"
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)