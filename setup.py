from setuptools import setup, find_packages
from os.path import join, dirname, abspath

long_description = open('README.md').read()


def read_requirements(basename):
    reqs_file = join(dirname(abspath(__file__)), basename)
    with open(reqs_file) as f:
        return [req.strip() for req in f.readlines()]


def main():
    reqs = read_requirements('requirements.txt')
    test_reqs = read_requirements('requirements_test.txt')

    setup(
        name='alea',
        version='0.1.1',
        description="Alea",
        long_description=long_description,
        long_description_content_type='text/markdown',
        author="Mitchell Sayer & David Hacker",
        author_email="",
        packages=find_packages(include='alea.*'),
        license='Apache 2.0',
        download_url='',
        include_package_data=True,
        zip_safe=True,
        url="",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Framework :: ',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python',
        ],
        keywords=[],
        install_requires=reqs,
        extras_require={
            'test': test_reqs,
        },
    )


if __name__ == '__main__':
    main()
