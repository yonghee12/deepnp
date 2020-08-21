import setuptools

setuptools.setup(
    name="deepnp",
    version="0.1.0",
    license='MIT',
    author="Yonghee Cheon",
    author_email="yonghee.cheon@gmail.com",
    description="Building deep learning models with numpy only",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yonghee12/deepnp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', "cupy"],
)
