from setuptools import setup, find_packages

setup(
    name='wizard',
    version='0.0.1',
    description='Wizard Gym Environment',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4,<=0.15.7',
        'numpy>=1.13.0'
    ]
)


