# Report for Milestone 1

# Introduction
This project aims at building a software library for **Automatic Differentiation**, or **AD**.
The goal of AD is to provide an **accurate** estimate of the derivative of a given function. It combines the advantages of 
Numerical Methods such as Finite Differences, and the Symbolic Computing that builds a symbolic expression of the derivative. 
AD is able to compute an approximation of the derivative of a function, **without computing a symbolic expression** of the derivative and 
with an **accuracy of machine precision**.

AD has many applications accross Science and Engineering, the most popular one these days being Deep Neural Networks. These models 
try to fit a function with >10M parameters to a dataset. To do so, they use Gradient Descent algorithms using gradients approximations provided
by AD. Famous applications include **Alpha Go**, **Self-Driving Cars** and **Image Recognition**.

# Background

# How to Use FancyProjectName ?

```
You can include code-like markdown like this : 
function test() {
  console.log("notice the blank line before this function?");
}
```

# Software Organization

# Implementation
The core data structures are `Variables` and `Blocks`.

We are goin to consider that every function can be splitted into core components, each of which being called a `Block`. Thus, the application of a function is a mere composition of `Block` operations. The function
![comp-graph](img/basic_function.png)


The first core data structure is `Variable`. This object will flow through the several `Blocks`, storing the new values of the functions computed, as well as the gradient computed so far.
![comp-graph](img/Variable.png)
It contains two main attributes : data and gradient. In each block, the input Variable brings the info from the previous value functions and the previous gradients computed so far and propagates the data flow as well as the gradient flow.

Note that we are not doing in-place modification of the input `Variable` in each `Block` as we may need of this `Variable` later in the computation



# Additional Comments
