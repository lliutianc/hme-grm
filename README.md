# Sparse HME on Graded Response Model

This repo implements a stochastic proximal gradient descent of the two-layer HME with graded response model at its leaf note. Sparse group lasso penalty is added for feature selection and sparsity.

For the results of the simulation study, refer to the `experiment/simulation-plot.ipynb` file, we included the best sHME we observed on 2/3-class datasets as pre-trained models.

For the results of the IMDB-50K study, refer to the `experiment/movie-review-xxx.ipynb` files, we didn't include any pre-trained sHME for these datasets, but the results should be relatively robust. 
