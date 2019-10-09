"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it - make sure you use python 3.x).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

"""
from io import open
from setuptools import find_packages, setup

setup(
    name="transformers",
    version="2.1.0",
    author="Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Google AI Language Team Authors, Open AI team Authors, Facebook AI Authors, Carnegie Mellon University Authors",
    author_email="thomas@huggingface.co",
    description="State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='NLP deep learning transformer pytorch tensorflow BERT GPT GPT-2 google openai CMU',
    license='Apache',
    url="https://github.com/huggingface/transformers",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'regex',
                      'sentencepiece',
                      'sacremoses'],
    entry_points={
      'console_scripts': [
        "transformers=transformers.__main__:main",
      ]
    },
    # python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
