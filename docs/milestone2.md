# Report for Milestone 2

# Introduction
AutoGrad is a forward mode **Automatic Differentiation** (**AD**) software library.

Differentiation is a fundamental mathematical operation that underpins much of science and engineering. Differentiation is used to describe how a function changes with respect to a specific variable. Differential equations are common throughout science and engineering; from modeling the evolution of bacteria to calculating rocket thrust over time to predictive machine learning algorithms, the ability to rapidly compute accurate differential equations is of great interest.

The symbolic derivative of a function is precise; however as the function of interest become more complex, the symbolic derivative becomes increasingly difficult to determine. Numeric methods can be used to compute the derivative of such functions. The finite difference approach uses the definition of a derivative to estimate the derivative of a function; however, it suffers from low accuracy and instability.

AD is able to compute an approximation of the derivative of a function, **without computing a symbolic expression** of the derivative and with **machine precision** accuracy.

AD has many applications across Science and Engineering, the most popular one these days being Deep Neural Networks. These models try to fit a function with >*10M* parameters to a dataset. To do so, they use Gradient Descent algorithms using gradients approximations provided by AD. Famous applications include **Alpha Go**, **Self-Driving Cars** and **Image Recognition**.

# How to use AutoGrad

### Installation

#### Requirements
AutoGrad works with [`Python3`](https://docs.python.org/3/)
#### Dependencies
* [`Numpy`](http://www.numpy.org/)

#### How to Install:
1. Create a virtual environment
```
cd my_directory
virtualenv my_env
```
2. Activate the virtual environment
```
source my_env/bin/activate
```
3. Install package - there are a few ways to do this
  1. Pip (Preferred)
```
pip install AutoGrad
```
  2. From the source
    1. Download Package From GutHub and Unzip
```
unzip cs207-FinalProject-master.zip
```
    2.  Install Dependencies
```
pip install -r cs207-FinalProject-master/requirements.txt
```
