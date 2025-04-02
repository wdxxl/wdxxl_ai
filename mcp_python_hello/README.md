
```
$ cd ~/github.com/wdxxl/mac_apple_silicon_ai 
$ cd mcp_python_hello 
$ python3 -m venv .venv
$ source .venv/bin/activate # deactivate
$ pip install -U pip
```


```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
pip --trusted-host pypi.python.org --trusted-host pypi.tuna.tsinghua.edu.cn install -U mcp mcp[cli] httpx beautifulsoup4 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

```
mcp version  
MCP version 1.6.0
```

```
 mcp dev mcp_server_demo.py

Need to install the following packages:
@modelcontextprotocol/inspector@0.7.0
```

http://localhost:5173