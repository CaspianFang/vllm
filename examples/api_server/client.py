import requests

data = {
    "prompt": "To be or not to be, ",
}

# 向127.0.0.1/generate发送post请求
response = requests.post("http://127.0.0.1:8000/generate", json=data).json()

# 返回结果
print(response)
