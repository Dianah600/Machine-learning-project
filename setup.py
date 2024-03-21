from setuptools import find_packages, setup
from typing import List



HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) ->List[str]:
    """
    This will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements        

setup(
    name = 'Machine-learning-project',
    version = '0.0.1',
    author = 'Dianah',
    author_email = 'dianah.namutosi10@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)