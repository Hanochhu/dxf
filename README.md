# DXF 处理工具

```
src/
├── core/                 # 核心数据结构
│   └── data_structures.py
├── parsers/              # 文件解析模块
│   ├── parser_interface.py
│   ├── dxf_parser.py
│   └── step_parser.py
├── feature/              # 特征提取和识别
│   └── block_identifier.py
├── connection/           # 连接分析
│   └── connection_analyzer.py
├── graph/                # 图构建
│   └── cad_graph.py
├── query/                # 查询接口
│   └── cad_query.py
├── system/               # 系统整合
│   └── cad_analysis_system.py
└── examples/             # 使用示例
    ├── basic_usage.py
    └── advanced_usage.py
```

## 使用步骤

1. 把 `图例和流程图_仪表管件设备均为模块` 这个文件放在根目录下
2. 运行 `dxf_block_extractor.py`，提取出所有块
3. 运行 `Entity.py`，找到匹配的块和实体组。
4. 运行'block_to_block.py',得到相关信息


## Docker 使用说明

### 构建 Docker 镜像

```bash
docker build -t dxf-image .
```

### 运行容器并执行脚本

```bash
docker run dxf-image python dxf_block_extractor.py
docker run dxf-image python Entity.py
```

### 启动交互式 bash 会话

```bash
docker run -it dxf-image /bin/bash
```

### 在容器中运行测试

```bash
docker run dxf-image python test_entity.py
```

## 依赖项

- Python 3.11
- ezdxf
- matplotlib
- networkx
