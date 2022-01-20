
import argparse
from utils.BaseFlags import parser as parser

# DATASET NAME
parser.add_argument('--dataset', type=str, default='GM_MNIST', help="name of the dataset")

# DATA DEPENDENT
# to be set by experiments themselves
parser.add_argument('--style_m1_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_m2_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")
parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=20, help="number of pairs per sample")
parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")

# Gaussian mixture
parser.add_argument('--emb_dim', type=int, default=8, help="dimension of the embedded Gaussian mixtures")
parser.add_argument('--GM_sample_size', type=int, default=800, help="sample size of the Gaussian mixtures")
parser.add_argument('--GM_radius', type=int, default=3, help="radius of the circle along which the Gaussian mixtures align")
parser.add_argument('--GM_var', type=int, default=1/5, help="variance of the Gaussian mixtures")

# SAVE and LOAD
# to bet set by experiments themselves
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")

# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=1, help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=1, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=1, help="default weight divergence term prior")