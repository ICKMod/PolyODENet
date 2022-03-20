from setuptools import setup

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]

setup(
    name='PolyODENet',
    version='0.1',
    packages=['Source'],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Talin Avanesian, Xiaohui Qu, Huub Van Dam, Qin Wu',
    author_email='qinwu@bnl.gov',
    description='Artificial Intelligence Catalysis Kinetics',
    python_requires='>=3.7',
    install_requires=required_list,
    entry_points={
        "console_scripts": [
            "train_poly = Source.main:main"
        ]
    }
)
