# DXF 处理工具

## 使用步骤

1. 把 `图例和流程图_仪表管件设备均为模块` 这个文件放在根目录下
2. 运行 `dxf_block_extractor.py`，提取出所有块
3. 运行 `Entity.py`，找到匹配的块和实体组。

## Docker 使用说明

### 构建 Docker 镜像

把 `图例和流程图_仪表管件设备均为模块` 这个文件放在根目录下

```bash
docker build -t dxf-image .
```


### 启动交互式 bash 会话

```bash
docker run -it dxf-image /bin/bash
```

### 在容器中运行相应代码



## 依赖项

- Python 3.11
- ezdxf
- matplotlib
- networkx
