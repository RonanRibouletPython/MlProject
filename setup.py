# Create the ML application as a package

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:

    # This function will return the list of packages required for the project
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[requirement.replace("\n", "") for requirement in requirements ]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name="MlProject",
    version="0.0.1",
    author="Ronan",
    author_email="ronan.riboulet@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)