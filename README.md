This repository is adapted from [MoPoE](https://github.com/thomassutter/MoPoE/tree/477a441ecb6c735a0b8af4d643fe3ac04c58171f), 
the official code for the ICLR 2021 paper ["Generalized Multimodal ELBO"](https://openreview.net/forum?id=5Y21V0RDBV) by Sutter T, Daunhawer I, and Vogt J.
We adapted the code for running experiments on the intended datasets.

# Usage

1. Prepare the data

```
curl -L -o tmp.zip https://www.dropbox.com/sh/lx8669lyok9ois6/AADmH2Q6T_iIlRg2Hp-R_Clca\?dl\=1
unzip tmp.zip
unzip data_mnistsvhntext.zip -d data/
```
(This comes with the other datasets. Remove them to save space. I will upload needed ones separately in the future.)

Now it is not necessary to put the data in the `data/` folder. Edit the data pat in `job_*`.

It is also not necessary to prepare the pretrained classifier models, which is included in the zip file in the drop box, if you do not need them.

2. Modify & run the script
See `job_*` and `main_*.py` for examples on how to train them.

`alphabet.json` and the related are reserved for future.
