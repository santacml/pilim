import setuptools

# with open("pilimit_lib\README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

long_description = "test"

setuptools.setup(
    name="pilimit",
    version="0.1.0",
    author="Greg Yang, Michael Santacroce, Edward J. Hu",
    author_email="michael.santacroce@microsoft.com",
    description="PyTorch implementation of the Pi-Limit library, which can be used to create trainable, deep, infinite-width, feature learning neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santacml/pilim",
    packages=setuptools.find_packages(),  #include="pilimit_lib", exclude="pilimit_orig"
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    'numpy',
    'torch>=1.12.0',
    'torchvision',
    'opt-einsum'
],
)