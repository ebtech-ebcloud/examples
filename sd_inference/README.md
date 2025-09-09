# inference
Gradio 是一个 Python 库，用来快速搭建网页界面，使写的 Python 函数可以交互。
最常用的场景是：

* 让别人通过网页测试你的 AI 模型（比如文本生成、图像生成、语音识别等）
* 自己调试模型，也可以直接用浏览器操作
* 完全不用写前端 HTML/JS，Gradio 内部实现

简单来说：写好函数 → Gradio 自动帮做网页界面 → 在浏览器里点按钮就能运行函数。

因此该推理实例以调用 AI-ModelScope/stable-diffusion-xl-base-1.0 文生图模型，并且使用 Gradio 进行本地浏览器部署。

## 创建 conda 环境
```
conda create -n sdbase python=3.10
conda activate sdbase
```

## 安装依赖
```
pip install -r re.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行
```
python app.py
```
> 该代码会自动下载 AI-ModelScope/stable-diffusion-xl-base-1.0 模型，需要等待几分钟

## 本地部署

在本机 cmd 或者 PowerShell 输入
```
ssh -p 3xxxx -L 8000:127.0.0.1:7860 root@ssh-cn-huabei1.ebcloud.com
 ```

> `-p` 是登录服务器用的端口，在开发机`远程连接`的地方获得端口号和密码（端口密码复制到本地即可显现）

> `-L` 把本地的 8000 端口转发到远端的 127.0.0.1:7860，这个端口和启动服务的端口保持一致即可

输入密码，等待本地连接成功。

打开网页输入网址：`http://127.0.0.1:8000`，进行实验。
