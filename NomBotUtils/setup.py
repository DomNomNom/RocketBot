import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


setuptools.setup(
    name='nombotutils',
    packages=setuptools.find_packages(),
    install_requires=[
        'rlbot',
        'rlbottraining>=0.3.0',
        'numpy',
    ],
    python_requires='>=3.7.0',
    version='0.1.1',
    description='Shared code for Dom\'s bots.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='DomNomNom',
    author_email='dominikschmid93@gmail.com',
    url='https://github.com/DomNomNom/RocketBot',
    keywords=['rocket-league', 'bot'],
    license='MIT License',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={},
    package_data={},
)
