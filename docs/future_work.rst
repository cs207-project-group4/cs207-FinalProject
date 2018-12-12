.. _installation:

Installation
=============

autograd can be install though pypi and from our GitHub repository. The recommended way to install autograd is through pypi.


Pypi installation
------------------

Pypi installation::

    pip install dragongrad

GitHub Installation
-------------------

1. Create a virtual environment::

    cd my_directory
    virtualenv my_env

2. Activate the virtual environment::

    source my_env/bin/activate

3. Download Package from GutHub (or clone) and Unzip::

    unzip cs207-FinalProject-master.zip

4. Install Dependencies using Pip::

    pip install -r cs207-FinalProject-master/requirements.txt

5. Install autograd -- this step is **Very Important**::

    cd cs207-FinalProject-master
    python3 setup.py install


Requirements
------------

autograd works with `Python3
<https://docs.python.org/3/>`_.

Both installation methods will install the correct version of Numpy, It is recommended install this software in a virtual environment.

Dependencies
- `Numpy
<http://www.numpy.org/>`_.
