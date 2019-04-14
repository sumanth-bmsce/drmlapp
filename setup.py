from setuptools import find_packages, setup

requirements = ['flask==1.0.2', 'numpy==1.16.2', 'scipy==1.0.0', 'tensorflow==1.13.1', 'keras==2.2.4', 'werkzeug==0.15.1','pillow==6.0.0']
setup(
    name = "drmlapp",
    version = "1.0.6",
    long_description= "Python Web Application for DR Diagnosis",
    author = "Sumanth Simha, Nagaraj G",
    author_email = "ssimha152@gmail.com",
    python_requires = ">=3.6.0",
    packages = ["app", "app.templates"],
    install_requires= requirements
)