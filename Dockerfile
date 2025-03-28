# 使用指定的基础镜像
FROM crpi-v332mopkzpj010s0.cn-hangzhou.personal.cr.aliyuncs.com/woody1/ai:v1

# 设置工作目录1
WORKDIR /ai/WatermarkRemover-AI

# 拉取最新代码1
RUN git pull

# 安装依赖
RUN conda run -n py312aiwatermark pip install Flask

# 启动应用
CMD ["conda", "run", "-n", "py312aiwatermark", "python", "web2.py"]
