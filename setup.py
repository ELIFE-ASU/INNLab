import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="INNLab",
    version="0.5.0",
    author="Yanbo Zhang",
    author_email="zhangybspm@gmail.com",
    description="A package for invertible neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhangyanbo/INNLab",
    project_urls={
        "Bug Tracker": "https://github.com/Zhangyanbo/INNLab/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=['INN', 'INN.ResFlow'],#setuptools.find_packages(where="src/INN"),
    python_requires=">=3.6",
)