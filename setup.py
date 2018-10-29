from setuptools import setup


setup(
  name='nnsum-data',
  version='1.0',
  description='Data preprocessing and downloading for nnsum library.',
  author='Chris Kedzie',
  author_email='kedzie@cs.columbia.edu',
  packages=[],
  dependency_links = [
    'git+https://github.com/kedz/rouge_papier.git#egg=rouge_papier',
    'git+https://github.com/kedz/duc_preprocess.git#egg=duc_preprocess-v1.0'],
  install_requires = ["rouge_papier", "duc_preprocess", "ujson", "requests",
                      "beautifulsoup4", "lxml"],
)
