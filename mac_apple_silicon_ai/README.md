

```
$ cd ~/github.com/wdxxl/mac_apple_silicon_ai 
$ python3 -m venv .venv
$ source .venv/bin/activate # deactivate
$ pip install -U pip
```

```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

Install Torch:
```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

Install TensorFlow:
```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U tensorflow tensorflow-macos tensorflow-metal -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
pip freeze
pip freeze > requirements.txt
pip install -r requirements.txt
```