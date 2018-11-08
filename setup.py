import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoGrad",
    version="0.0.1",
    author="cs207-project-group4",
    author_email="N/A",
    description="Automatic Differentiation Library",
    long_description=long_description,
    url="https://github.com/cs207-project-group4/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
