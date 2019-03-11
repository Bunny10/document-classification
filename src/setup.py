from setuptools import setup, find_packages, dist

requirements = [package for package,version in
                  [line.split ("==") for line in
                   open("requirements.txt").read().split("\n") if line]]

setup(
    name="document_classification",
    version="0.0.1",
    description="Document Classification Service.",
    url="https://github.com/GokuMohandas/document-classification",
    author="Goku Mohandas",
    author_email="goku@ciitizen.com",
    packages=find_packages(),
    setup_requires=['Flask==1.0.2'],
    install_requires=requirements,
    python_requires=">=3.6",
    test_suite="tests",
)