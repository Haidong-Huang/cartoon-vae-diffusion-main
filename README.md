
## Overview

This repository is on using **variational autoencoders (VAE)** and **diffusion model (DDPM)** for cartoon image generation.  
Code is implemented using **Python 3.10** and tested on **Ubuntu** OS.  
[Cartoon Set](https://google.github.io/cartoonset/) (10k) is used for training both the VAE and diffusion models.  
More implementation details can be found at this [blog post](https://lihanlian.github.io/posts/blog8).

**VAE Result**

<p align="center">
  <img alt="VAE Sample Grid" src="runs/vae/sample_grid.png" width="50%" />
</p>
<p align="center">5x5 VAE sample grid</p>

**DDPM Result**

<p align="center">
  <img alt="DDPM Sample Grid" src="runs/ddpm/sample_grid_200.jpg" width="50%" />
</p>
<p align="center">5x5 DDPM sample grid</p>

---

## Run Locally (Windows)

Follow these steps to set up and run this project on a Windows machine.

### 1. Install Python

Make sure you have **Python 3.10 or newer** installed.  
You can download it from:

https://www.python.org/downloads/

Be sure to check **“Add Python to PATH”** during installation.

---

### 2. Create a virtual environment

Open **PowerShell** or **Command Prompt** and run:

```powershell
python -m venv env
```

### 3. Activate the virtual environment

```powershell
.\env\Scripts\Activate.ps1
```

If you see a permission error like cannot be loaded because running scripts is disabled, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the activation command above.

Command Prompt (cmd.exe)

```powershell
env\Scripts\activate.bat
```

### 4. Install project dependencies

Once the environment is active, install requirements:

```powershell
pip install -r requirements.txt
```

### 5. Run training scripts

You can now run the Python scripts from this repository.

for example
```powershell
python train_vae.py
```
or
```powershell
python train_ddpm.py
```

To train the convolutional neural network using the fast neural style transfer algorithm:

```powershell
python fast_nst_training.py
```

## References
 1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE Paper)
 2. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM Paper)
 3. [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg), [Diffusion Models | PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU&t=874s) (YouTube) and corresponding github repo [Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch)
 