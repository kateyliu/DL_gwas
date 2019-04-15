# Genomic selection using deep learning and saliency map

We provide a deep-learning method to predict five 	quantitative traits (Yield, Protein, Oil, Moisture and Plant height) of SoyNAM dataset.
We also applied saliency map approach measure phenotype contribution for genome wide association study. 
The program is implemented using Keras2.0 and Tensorflow backend with python 2.7

### Prerequisites

Python packages are required,

```
numpy
pandas
tensorflow
keras
scipy
sklearn
matplotlib
```
## Running the program

The scripts train and test model with 10 fold cross validation and plot a comparison of genotype contribution using saliency map value and Wald test value.

* **polytest.txt** - *Genotype contribution using Wald test score. Run with SoyNAM R package.*
* **saliency_value.txt** - *Genotype contribution calculated using saliency map approach.*
* **height.py** - *Executive scripts.*
* **IMP_height.txt** - *Inputs of imputed genotype matrix.*
* **QA_height.txt** - *Inputs of quality assured non-imputed genotype matrix.*

```
cd HEIGHT
python height.py

```

## Authors

* **Yang Liu** - *University of Missouri, Columbia MO, USA*
* **Email** - *ylmk2@mail.missouri.edu* 
* **Email** - *yanglou1990@gmail.com*


## License
GNU v2.0

