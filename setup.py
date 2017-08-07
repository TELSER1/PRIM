from setuptools import setup,find_packages

setup(
    name='prim',
    packages=find_packages(),
    version='0.1',
    setup_requires=['setuptools>=18.0', 'numpy'],
    install_requires=['numpy']
    )
