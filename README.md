# Bayesian Black Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This repository documents my work on the **Bayesian Black Box Optimisation (BBO) Capstone Project**, where the objective is to efficiently optimise a set of unknown functions using a limited number of queries.

The challenge involves **eight synthetic black-box functions** with dimensionalities ranging from **2 to 8 input variables**. The internal structure of each function is unknown, meaning the only information available is the output returned after submitting a query.

The goal of the project is to **iteratively propose new input query points that maximise the function outputs** while minimising the number of evaluations. This reflects many real-world machine learning and engineering problems where evaluating a system is **expensive, noisy, or time-consuming**, such as hyperparameter tuning, chemical experiments, or engineering design optimisation.

This project helps me strengthen my understanding of **Bayesian optimisation, surrogate modelling and exploration–exploitation trade-offs**, which are important techniques in modern machine learning workflows. These skills are particularly relevant for my work and career interests in **quantitative modelling, risk analytics and applied machine learning**.

---

## 2. Inputs and Outputs

Each function accepts a vector of input parameters and returns a single output value.

### Inputs
- Inputs are **continuous values between 0 and 1**.
- The dimensionality depends on the specific function.

| Function | Input Dimension |
|--------|----------------|
| Function 1 | 2 |
| Function 2 | 2 |
| Function 3 | 3 |
| Function 4 | 4 |
| Function 5 | 4 |
| Function 6 | 5 |
| Function 7 | 6 |
| Function 8 | 8 |

Queries must be submitted in the following format: x1-x2-x3-...-xn


Where each value:
- begins with `0`
- is expressed to **six decimal places**

Example (2D function): 0.421819-0.390736


### Output

The system returns a **single scalar response value**, representing the performance of the chosen inputs.

Examples of what this value may represent include:

- model performance score
- chemical process yield
- likelihood score
- optimisation objective

All tasks are framed as **maximisation problems**, meaning higher values represent better outcomes.

---

## 3. Challenge Objectives

The objective of the BBO capstone project is to **identify input combinations that maximise the output of each unknown function** using as few queries as possible.

Key challenges include:

- The **true functional form is unknown**
- Only **limited evaluations are allowed**
- Functions may contain **noise, multiple local optima or nonlinear relationships**
- Higher-dimensional functions significantly increase search complexity

Each week, a new query is submitted for every function, and the resulting outputs are added to the dataset. Over time, this iterative process allows the model to gradually learn more about the underlying response surface.

The main goal is therefore to **design a query strategy that balances exploration of uncertain regions and exploitation of promising areas**.

---

## 4. Technical Approach

My approach uses **surrogate modelling and Bayesian optimisation principles** to guide query selection.

### Surrogate Model

To approximate the unknown functions, I use **Gaussian Process regression**. This model is well-suited for black-box optimisation because it provides:

- a predicted mean (expected output)
- a predictive uncertainty estimate

These two components allow the algorithm to reason about where to sample next.

### Acquisition Strategy

To select new query points, I use an **Expected Improvement (EI)** acquisition function. EI evaluates candidate points based on:

- how much improvement they might provide over the current best value
- how uncertain the model prediction is in that region

This naturally balances:

- **exploitation** – sampling near predicted high-value regions
- **exploration** – sampling where model uncertainty is high

### Iterative Strategy

The optimisation process follows these steps:

1. Fit a Gaussian Process model to the existing observations.
2. Evaluate the acquisition function across candidate points.
3. Select the point with the highest acquisition value.
4. Submit the query and observe the new output.
5. Update the dataset and repeat the process.

### Evolution of Strategy

During the first rounds of the project, the focus was on **learning the overall structure of each function**, which required more exploration. As more data points become available, the strategy gradually shifts toward **exploitation of promising regions**.

For higher-dimensional functions (6D–8D), additional exploration remains important because large areas of the search space remain uncertain.

This repository will continue to evolve as new data becomes available and optimisation strategies are refined.



