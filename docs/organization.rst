Software Organization
=====================

The `autograd` package organized into various modules. Our basic directory structure will look as follows::

    cs207-FinalProject/
        autograd/
            __init__.py
            blocks/
                __init__.py
                block.py
                expo.py
                hyperbolic.py
                operations.py
                trigo.py
            tests/
                __init__.py
                test_basic.py
                test_autograd.py
                ...
            config.py
            node.py
            utils.py
            variable.py
            optimize.py
        docs/
            dev_milestones/
                milestone1.md
                milestone2.md
            ...
        README.md
        requirements.txt
        setup.py
        Demo_Notebook.ipynb


The autograd package is organized into a few key modules:

- ``block.py``: objects implementing the core computational units of the graph, namely ``data_fn`` (*f(x)*) and ``gradient_fn`` (*f'(x)*).

- Within the blocks submodule, there additional block operations - categorized by operation type.

- ``variable.py``: data structure containing the function value and gradient value

- ``utils.py``: general utility functions that are reused throughout the project

- ``optimize.py``: contains the optimizer classes and functions

- ``node.py``: contains the node class and computational graph class for reverse mode

- ``config.py`` : Stores all the nodes for reverse mode

- ``tests``: contain all the tests, divided by which module is being tested

- ``docs``: contains development milestones in a sub directory, also contains useful information about the project, hosted on `read the docs <https://autograd.readthedocs.io/en/latest/#>`_.
