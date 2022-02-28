import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="INNLab",
    version="0.5.3",
    author="Yanbo Zhang",
    author_email="Zhang.Yanbo@asu.edu",
    description="A package for invertible neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhangyanbo/INNLab",
    project_urls={
        "Bug Tracker": "https://github.com/ELIFE-ASU/INNLab/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['INN',
              'INN.ResFlow',
              'INN.CouplingModels.NICEModel',
              'INN.CouplingModels.RealNVP',
              'INN.CouplingModels',
              'INN.Linears'],
    python_requires=">=3.6",
    test_suite='test',
)