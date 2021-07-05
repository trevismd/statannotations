from distutils.core import setup
from setuptools import find_packages
import re

with open("README.md", "r") as f:
    long_description = f.read()

VERSIONFILE = "statannotations/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M)
if match:
    version = match.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


def remove_dev_only(list_of_dependencies):
    return [dep for dep in list_of_dependencies
            if dep.strip() not in ("statsmodels", "packaging")]


setup(
    name="statannotations",
    version=version,
    maintainer="Florian Charlier",
    maintainer_email="trevis@cascliniques.be",
    description=("add statistical significance or custom annotations on "
                 "seaborn plots. Based on statannot 0.2.3"),
    license="MIT License",
    license_file="LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trevismd/statannotations",
    packages=find_packages(exclude=("tests", "usage", "docs", "build", "dist")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=remove_dev_only(open("requirements.txt").readlines()),
    python_requires='>=3.6',
)
