# IWEM
This is a pytorch implementation of IWEM, which is a new and general approach to learn deep generative models, as described in the following paper:
* Dongha Kim, Jaesung Hwang, Yongdai Kim, On casting importance weighted autoencoder to an EM algorithm to learn deep generative models

## Requirements
The code is compatible with:
* `pytorch 0.4.0`

## Run the experiments
```bash
python MLE_Analysis.py --dataname 'mnist' --decoder_option 'cnn' --option 'iwem' --num_sam 15
```
You can change `dataname` to one of mnist, bimnist, omniglot, or caltech101. And The `decoder_option` can also be changed to one of mlp and cnn, and the `option` can also be changed to one of `iwae`, `iwem`, `iwem_woo`, and `iwem_woa`.
