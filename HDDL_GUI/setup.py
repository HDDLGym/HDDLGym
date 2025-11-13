from setuptools import setup, find_packages

setup(
    name="hddl_gui",
    version="0.1.0",
    description="HDDL_GUI to understand the hierarchical structures of domains, problems, and resulting plans.",
    author="Ngoc La",
    packages=find_packages(),
    install_requires=[
        "flask",
        "matplotlib",
        "networkx",
        "graphviz",
        "python-dotenv",
        "anthropic"
    ],
    python_requires=">=3.8",
)
