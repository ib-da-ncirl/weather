import os
import setuptools

_here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="weather",
    version='0.0.1',
    author="Ian Buttimer",
    author_email="author@example.com",
    description="Weather Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ib-da-ncirl/yelp_analysis",
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
      'setuptools>=47.3.1',
      'pandas>=1.0.5',
      'requests>=2.24.0'
    ],
    dependency_links=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
