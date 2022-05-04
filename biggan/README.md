# Vision-aided training for BigGAN (CIFAR)

This repo is implemented upon the [BigGAN-PyTorch repo](https://github.com/ajbrock/BigGAN-PyTorch) and [DiffAugment-biggan-cifar](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-biggan-cifar). 
The main dependencies are:

* PyTorch version >= 1.0.1. Code has been tested with PyTorch 1.8.0.
* python libraries: `pip install -r requirements.txt`


## Pre-Trained Models and Evaluation

To evaluate a model on CIFAR-10 or CIFAR-100, run the following command:

```bash
python eval.py --dataset=<dataset_name> --network=<model> --dataset_name=<clean_fid_dataset_name> --self_modulation <1/0>
```
Evaluaiton is done using [clean-fid](https://github.com/GaParmar/clean-fid). Here, `dataset` is `C10` (CIFAR-10) or `C100` (CIFAR-100); `model` is the path to generator weights (file named `G_ema_best.pth` in the `weights` folder), or a pre-trained model. `dataset_name` is `cifar10` or `cifar100`. For unconditional models `self_modulation` will be TRUE. Our prerained models can be downloaded from [here](https://www.cs.cmu.edu/~vision-aided-gan/models/)

<table>
  <tr>
    <td>BigGAN</td>
    <td colspan=4 align=center >Conditional</td>
    <td colspan=4 align=center>Unconditional</td>
  </tr>
  <tr>
    <td></td>
    <td colspan=2 align=center>CIFAR-10</td>
    <td colspan=2 align=center>CIFAR-100</td>
    <td colspan=2 align=center>CIFAR-10</td>
    <td colspan=2 align=center>CIFAR-100</td>
  </tr>
  <tr>
    <td></td>
    <td  >100 % </td>
    <td  >10 % </td>
    <td  >100 % </td>
    <td  >10 % </td>
    <td  >100 % </td>
    <td  >10 % </td>
    <td  >100 % </td>
    <td  >10 % </td>
  </tr>
  <tr>
    <td>DiffAugment</td>
    <td  >10.09 </td>
    <td  >27.81 </td>
    <td  > 13.60</td>
    <td  > 39.59 </td>
    <td  >15.23 </td>
    <td  >32.63</td>
    <td  >19.20 </td>
    <td  > 33.75</td>
  </tr>
  <tr>
    <td>Vison-aided (Ours)</td>
    <td  >8.75 </td>
    <td  >13.11 </td>
    <td  >10.88</td>
    <td  >15.71 </td>
    <td  >11.17 </td>
    <td  >16.34</td>
    <td  >14.10 </td>
    <td  > 19.13</td>
  </tr>
</table>

The evaluation results of the pre-trained models should be close to these numbers. Specify `--repeat=NUM_REPEATS` to compute means and standard deviations over multiple evaluation runs. A standard deviation of less than 1% relatively is expected.

## Training

We provide all the training scripts in the `scripts` folder. For e.g. to train our model on CIFAR-10 with 10% training data.

```bash
#install vision-aided-gan library if not already done
cd ..
pip install .
cd biggan

# collect FID statistics
CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset C10

# train
CUDA_VISIBLE_DEVICES=0 bash scripts/vision-aided-biggan-cifar10-0.1.sh
```
