# Game-theoretical Preference Optimization (GPO)


### *The Code of Paper "Toward Optimal LLM Alignments Using Two-Player Games". 👉 <a href="http://arxiv.org/abs/2406.10977" target="_blank">[Arvix Link]</a>*

## 🔩 Requirements & Setup

This reponsitory based on [ MOSS-RLHF]("https://github.com/OpenLMLab/MOSS-RLHF").

This repository works on Python 3.8 and PyTorch 1.13.1.

We recommend using the **conda** virtual environment to run the code.

#### Step 1: Create a new Python virtual environment

```bash
conda update conda -n base -c defaults
conda create -n rlhf python=3.8
conda activate rlhf
```

#### Step 2: Install PyTorch and TensorBoard

```bash
conda install pytorch==1.13.1 pytorch-cuda=11.7 tensorboard -c pytorch -c nvidia
```

#### Step 3: Install the remaining dependencies

```bash
conda install datasets accelerate safetensors chardet cchardet -c huggingface -c conda-forge
pip3 install transformers sentencepiece einops triton==1.0.0 rouge jionlp==1.4.14 nltk sacrebleu cpm_kernels

apt install libaio-dev
DS_BUILD_OPS=1 pip install deepspeed

pip3 install -r requirements.txt
```

## ✨ Start training your own model!

### Training GPO model

Run the command below.

```
# You need to use your own sft model currently.
bash train_gpo.sh
```

## Citation

```bibtex
@article{zheng2024toward,
  title={Toward Optimal LLM Alignments Using Two-Player Games},
  author={Zheng, Rui and Guo, Hongyi and Liu, Zhihan and Zhang, Xiaoying and Yao, Yuanshun and Xu, Xiaojun and Wang, Zhaoran and Xi, Zhiheng and Gui, Tao and Zhang, Qi and others},
  journal={arXiv preprint arXiv:2406.10977},
  year={2024}
}
```