from setuptools import setup, find_packages

setup(
    name="pydeepity",
    version="1.0.0",
    author="Ra4ster",
    description="A high-performance Predictive Coding Network C++ engine",
    packages=find_packages(),

    package_data={
        "pydeepity": ["*.so", "*.pyd", "*.dylib"],
    },
    include_package_data=True,
    python_requires=">=3.8",
)
