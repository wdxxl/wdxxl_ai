

```
$ cd ~/github.com/wdxxl/mac_apple_silicon_ai 
$ cd MoE
$ python3 -m venv .venv
$ source .venv/bin/activate # deactivate
$ pip install -U pip
```

```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U torch -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
python3 01_simple_model.py

Input: tensor([[1., 2., 3.],
        [4., 5., 6.]])
Output: tensor([[1.5244],
        [3.8805]], grad_fn=<AddmmBackward0>)
```
