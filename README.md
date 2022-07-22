#  To activate MPS on M1 MAC

### 1. Run these in Terminal
```shell
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

### 2. Create a conda environment :
```bash
mkdir pytorch
cd pytorch
```
### 3. Run these:

```shell
conda create --prefix ./env python=3.8
conda activate ./env
```

### 4.Install torch with MPS backend. Note that torch version should be >= 1.12.0
```bash
pip3 install torch torchvision torchaudio
```
### 5. install other DS libraries:
```bash
conda install jupyter pandas numpy matplotlib 
```
### 6. Run 
```shell
jupyter notebook
```
### 7. test the MPS backend with provided function.

```Python
def check_backend():
    global device
    print(f" Pytorch Version {torch.__version__}")
    print (f' MPS backend is bulit? {torch.backends.mps.is_built()}')
    print( f' MPS backend is available {torch.backends.mps.is_available()}')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f' Device is set to {device}')
    return 
```

### 8. I trained a simple CNN on EMNIST balanced dataset for example. please check the Pytorch_EMNIST notyebook.
