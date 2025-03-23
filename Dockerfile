# 使用指定的基础镜像
FROM crpi-v332mopkzpj010s0.cn-hangzhou.personal.cr.aliyuncs.com/woody1/ai:v1

# 设置工作目录
WORKDIR /ai/WatermarkRemover-AI

# 拉取最新代码
RUN git pull

# 安装 Flask
RUN pip install Flask

# 启动应用
CMD ["python", "web2.py"]