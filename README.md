# MultiCS

Welcome to the Python package for multi-task compressive sensing (MultiCS)!  This packages provides various algorithms to solve multiple compressive sensing tasks in parallel.  This codebase accompanies the paper [An Efficient Algorithm for Clustered Multi-Task Compressive Sensing](https://arxiv.org/abs/2310.00420) by [Alexander Lin](https://sites.google.com/view/alexanderlin) and [Demba Ba](http://www.demba-ba.org/).

## Basic Usage
The main entry point into the codebase is through the `MultiTaskCompSens` object.  You can instantiate it as follows:
```python
from multics.model import MultiTaskCompSens

model = MultiTaskCompSens(mode="clustered", alg="em", num_clusters=2)
```
There are two main parameters for this object (see our paper for more details):
- `mode`: This determines the type of model to use.  Options include `separate` (i.e. not sharing any information between CS tasks), `joint` (i.e. sharing information between all CS tasks), and `clustered` (i.e. automatically learning and sharing information between clusters of tasks).  If the `clustered` option is used, you also need to specify an additional argument `num_clusters`.  
- `alg`: This determines the type of algorithm to use.  Options are `em` (i.e. the original expectation-maximization algorithm) and `cofem` (i.e. the acceelerated, covariance-free version of EM proposed in our paper).  If using `cofem`, there are also additional required parameters: `num_probes` and `cg_tol`.

After instantiating the object, the `model.fit` function can be used to run the inference algorithm and solve the CS tasks.  For a full example of how to use this function, see the `time.py` script file.  You can also use this script to reproduce the results in our paper.