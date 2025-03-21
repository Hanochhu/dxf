import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Set, Optional
import random

# --------- 1. 创建模拟数据 ---------

def generate_mock_cad_data():
    """生成模拟CAD图纸数据"""
    
    # 模拟块数据
    blocks = {
        "B1": {"id": "B1", "type": "block", "name": "Process", "position": (100, 200), "features": {"entity_count": 5, "aspect_ratio": 1.5}},
        "B2": {"id": "B2", "type": "block", "name": "Decision", "position": (300, 200), "features": {"entity_count": 6, "aspect_ratio": 1.0}},
        "B3": {"id": "B3", "type": "block", "name": "Input", "position": (500, 200), "features": {"entity_count": 4, "aspect_ratio": 2.0}},
        "B4": {"id": "B4", "type": "block", "name": "Output", "position": (300, 400), "features": {"entity_count": 3, "aspect_ratio": 2.0}},
        "B5": {"id": "B5", "type": "block", "name": "Terminal", "position": (500, 400), "features": {"entity_count": 5, "aspect_ratio": 1.2}},
        "B6": {"id": "B6", "type": "block", "name": "Subprocess", "position": (700, 400), "features": {"entity_count": 8, "aspect_ratio": 1.5}},
        "B7": {"id": "B7", "type": "block", "name": "DataStore", "position": (700, 200), "features": {"entity_count": 4, "aspect_ratio": 1.8}},
        "B8": {"id": "B8", "type": "block", "name": "Document", "position": (900, 300), "features": {"entity_count": 3, "aspect_ratio": 1.3}},
        "A1": {"id": "A1", "type": "arrow", "name": "Arrow1", "position": (200, 200), "features": {"entity_count": 3, "aspect_ratio": 3.0}},
        "A2": {"id": "A2", "type": "arrow", "name": "Arrow2", "position": (400, 200), "features": {"entity_count": 3, "aspect_ratio": 3.0}},
        "A3": {"id": "A3", "type": "arrow", "name": "Arrow3", "position": (300, 300), "features": {"entity_count": 3, "aspect_ratio": 3.0}},
    }
    
    # 模拟连接数据 - 包含已知方向和未知方向的边
    connections = [
        # 已知方向的边
        {"source_id": "B1", "target_id": "B2", "has_direction": True, "connection_type": "direct"},
        {"source_id": "B1", "target_id": "B4", "has_direction": True, "connection_type": "direct"},
        {"source_id": "B2", "target_id": "B3", "has_direction": True, "connection_type": "direct"},
        {"source_id": "B3", "target_id": "B7", "has_direction": True, "connection_type": "direct"},
        {"source_id": "B6", "target_id": "B8", "has_direction": True, "connection_type": "direct"},
        {"source_id": "B6", "target_id": "B1", "has_direction": True, "connection_type": "direct"},   # 形成环
        
        # 未知方向的边
        {"source_id": "B2", "target_id": "B4", "has_direction": False, "connection_type": "direct"},
        {"source_id": "B4", "target_id": "B5", "has_direction": False, "connection_type": "direct"},
        {"source_id": "B5", "target_id": "B6", "has_direction": False, "connection_type": "indirect"},
        {"source_id": "B7", "target_id": "B8", "has_direction": False, "connection_type": "direct"},
        {"source_id": "B3", "target_id": "B5", "has_direction": False, "connection_type": "direct"},
    ]
    
    return blocks, connections

# --------- 2. 构建 NetworkX 图 ---------

def build_graph(blocks, connections):
    """构建NetworkX图"""
    G = nx.DiGraph()
    
    # 添加节点
    for block_id, block_data in blocks.items():
        G.add_node(block_id, **block_data)
    
    # 添加边
    for conn in connections:
        G.add_edge(
            conn["source_id"], 
            conn["target_id"], 
            has_direction=conn["has_direction"],
            connection_type=conn["connection_type"]
        )
    
    return G

# --------- 3. 递归式出入度查询实现 ---------

class RecursiveQueryEngine:
    """实现递归式出入度查询"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def get_outgoing_recursive(self, block_id: str, max_depth: int = -1) -> Dict[str, List]:
        """递归获取出度连接"""
        if block_id not in self.graph:
            return {"connections": [], "nodes": []}
            
        visited_edges = set()  # 跟踪已访问的边
        visited_nodes = set()  # 跟踪已访问的节点
        results = {"connections": [], "nodes": []}
        
        self._get_outgoing_recursive(block_id, visited_edges, visited_nodes, results, 0, max_depth)
        
        return results
    
    def _get_outgoing_recursive(self, block_id: str, visited_edges: Set, visited_nodes: Set, 
                               results: Dict[str, List], depth: int, max_depth: int):
        """递归查询的内部实现"""
        # 添加当前节点
        if block_id not in visited_nodes:
            visited_nodes.add(block_id)
            results["nodes"].append({
                "id": block_id,
                "data": self.graph.nodes[block_id],
                "depth": depth
            })
        
        # 如果达到最大深度，停止递归
        if max_depth >= 0 and depth >= max_depth:
            return
        
        # 处理所有出边
        for _, target_id, edge_data in self.graph.out_edges(block_id, data=True):
            edge_key = (block_id, target_id)
            
            # 如果边未访问过
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                
                # 添加连接到结果
                results["connections"].append({
                    "source_id": block_id,
                    "target_id": target_id,
                    "data": edge_data,
                    "depth": depth
                })
                
                # 递归处理目标节点
                self._get_outgoing_recursive(target_id, visited_edges, visited_nodes, 
                                          results, depth + 1, max_depth)
    
    def get_incoming_recursive(self, block_id: str, max_depth: int = -1) -> Dict[str, List]:
        """递归获取入度连接"""
        if block_id not in self.graph:
            return {"connections": [], "nodes": []}
            
        visited_edges = set()  # 跟踪已访问的边
        visited_nodes = set()  # 跟踪已访问的节点
        results = {"connections": [], "nodes": []}
        
        self._get_incoming_recursive(block_id, visited_edges, visited_nodes, results, 0, max_depth)
        
        return results
    
    def _get_incoming_recursive(self, block_id: str, visited_edges: Set, visited_nodes: Set, 
                              results: Dict[str, List], depth: int, max_depth: int):
        """递归查询的内部实现"""
        # 添加当前节点
        if block_id not in visited_nodes:
            visited_nodes.add(block_id)
            results["nodes"].append({
                "id": block_id,
                "data": self.graph.nodes[block_id],
                "depth": depth
            })
        
        # 如果达到最大深度，停止递归
        if max_depth >= 0 and depth >= max_depth:
            return
        
        # 处理所有入边
        for source_id, _, edge_data in self.graph.in_edges(block_id, data=True):
            edge_key = (source_id, block_id)
            
            # 如果边未访问过
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                
                # 添加连接到结果
                results["connections"].append({
                    "source_id": source_id,
                    "target_id": block_id,
                    "data": edge_data,
                    "depth": depth
                })
                
                # 递归处理源节点
                self._get_incoming_recursive(source_id, visited_edges, visited_nodes, 
                                          results, depth + 1, max_depth)

# --------- 4. 迭代式实现（非递归实现递归效果）---------

class IterativeQueryEngine:
    """使用迭代方式实现递归式查询"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def get_outgoing_recursive(self, block_id: str, max_depth: int = -1) -> Dict[str, List]:
        """使用迭代方式实现递归出度查询"""
        if block_id not in self.graph:
            return {"connections": [], "nodes": []}
        
        results = {"connections": [], "nodes": []}
        visited_edges = set()
        visited_nodes = set()
        
        # 使用队列实现广度优先遍历
        queue = [(block_id, 0)]  # (节点ID, 深度)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            # 添加节点
            if current_id not in visited_nodes:
                visited_nodes.add(current_id)
                results["nodes"].append({
                    "id": current_id,
                    "data": self.graph.nodes[current_id],
                    "depth": depth
                })
            
            # 如果达到最大深度，不再处理出边
            if max_depth >= 0 and depth >= max_depth:
                continue
            
            # 处理所有出边
            for _, target_id, edge_data in self.graph.out_edges(current_id, data=True):
                edge_key = (current_id, target_id)
                
                # 如果边未访问过
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    
                    # 添加连接
                    results["connections"].append({
                        "source_id": current_id,
                        "target_id": target_id,
                        "data": edge_data,
                        "depth": depth
                    })
                    
                    # 将目标节点添加到队列
                    queue.append((target_id, depth + 1))
        
        return results
    
    def get_incoming_recursive(self, block_id: str, max_depth: int = -1) -> Dict[str, List]:
        """使用迭代方式实现递归入度查询"""
        if block_id not in self.graph:
            return {"connections": [], "nodes": []}
        
        results = {"connections": [], "nodes": []}
        visited_edges = set()
        visited_nodes = set()
        
        # 使用队列实现广度优先遍历
        queue = [(block_id, 0)]  # (节点ID, 深度)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            # 添加节点
            if current_id not in visited_nodes:
                visited_nodes.add(current_id)
                results["nodes"].append({
                    "id": current_id,
                    "data": self.graph.nodes[current_id],
                    "depth": depth
                })
            
            # 如果达到最大深度，不再处理入边
            if max_depth >= 0 and depth >= max_depth:
                continue
            
            # 处理所有入边
            for source_id, _, edge_data in self.graph.in_edges(current_id, data=True):
                edge_key = (source_id, current_id)
                
                # 如果边未访问过
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    
                    # 添加连接
                    results["connections"].append({
                        "source_id": source_id,
                        "target_id": current_id,
                        "data": edge_data,
                        "depth": depth
                    })
                    
                    # 将源节点添加到队列
                    queue.append((source_id, depth + 1))
        
        return results
    
    def has_connection_with_specific_block(self, source_id: str, target_block_ids: List[str], 
                                         recursive: bool = True, max_depth: int = -1) -> Dict:
        """
        检查块是否与指定目标块有连接
        
        Args:
            source_id: 源块ID
            target_block_ids: 目标块ID列表
            recursive: 是否递归查询
            max_depth: 递归最大深度
            
        Returns:
            包含查询结果的字典
        """
        if source_id not in self.graph:
            return {"has_connection": False, "paths": {}}
        
        results = {"has_connection": False, "paths": {}}
        
        if not recursive:
            # 非递归模式：直接检查相邻节点
            for target_id in target_block_ids:
                if self.graph.has_edge(source_id, target_id):
                    results["has_connection"] = True
                    results["paths"][target_id] = [source_id, target_id]
        else:
            # 递归模式：可以检查多层连接
            for target_id in target_block_ids:
                try:
                    if max_depth > 0:
                        # 使用广度优先搜索实现深度限制
                        path = self._bfs_with_depth(source_id, target_id, max_depth)
                        if path:
                            results["paths"][target_id] = path
                            results["has_connection"] = True
                        else:
                            results["paths"][target_id] = []
                    else:
                        # 不限深度的情况使用普通的最短路径算法
                        path = nx.shortest_path(self.graph, source_id, target_id)
                        results["paths"][target_id] = path
                        results["has_connection"] = True
                except nx.NetworkXNoPath:
                    results["paths"][target_id] = []
        
        return results

    def _bfs_with_depth(self, source: str, target: str, max_depth: int) -> List[str]:
        """
        使用广度优先搜索实现深度受限的路径查找
        
        Args:
            source: 起始节点
            target: 目标节点
            max_depth: 最大深度限制
            
        Returns:
            找到的路径，如果没有找到则返回空列表
        """
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            (vertex, path) = queue.pop(0)
            if len(path) > max_depth + 1:  # +1 是因为路径包含起始节点
                continue
            
            for neighbor in self.graph.neighbors(vertex):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []

# --------- 5. 可视化关系图 ---------

def visualize_graph(G, highlight_nodes=None, highlight_edges=None, title="CAD Relationship Graph"):
    """可视化关系图"""
    plt.figure(figsize=(12, 8))
    
    # 节点位置
    pos = {node: data["position"] for node, data in G.nodes(data=True)}
    
    # 节点颜色
    node_colors = []
    for node in G.nodes():
        if highlight_nodes and node in highlight_nodes:
            node_colors.append('red')
        elif G.nodes[node]["type"] == "arrow":
            node_colors.append('green')
        else:
            node_colors.append('skyblue')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
    
    # 分别绘制已知方向和未知方向的边
    known_edges = []
    unknown_edges = []
    highlighted_known_edges = []
    highlighted_unknown_edges = []
    
    for u, v, data in G.edges(data=True):
        if highlight_edges and (u, v) in highlight_edges:
            if data.get("has_direction", True):
                highlighted_known_edges.append((u, v))
            else:
                highlighted_unknown_edges.append((u, v))
        else:
            if data.get("has_direction", True):
                known_edges.append((u, v))
            else:
                unknown_edges.append((u, v))
    
    # 绘制已知方向的普通边
    nx.draw_networkx_edges(
        G, pos, edgelist=known_edges, 
        width=2, 
        edge_color='gray',
        style='solid',
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1"
    )
    
    # 绘制未知方向的普通边 (使用无箭头连线)
    for u, v in unknown_edges:
        connection_type = G[u][v].get("connection_type", "direct")
        style = 'dashed' if connection_type == "indirect" else 'solid'
        
        # 绘制无箭头连线表示未知方向
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            width=2, 
            edge_color='blue',
            style=style,
            arrows=False,
            connectionstyle="arc3,rad=0.1"
        )
    
    # 绘制高亮的已知方向边
    nx.draw_networkx_edges(
        G, pos, edgelist=highlighted_known_edges, 
        width=3, 
        edge_color='red',
        style='solid',
        arrows=True,
        arrowsize=25,
        connectionstyle="arc3,rad=0.1"
    )
    
    # 绘制高亮的未知方向边或推断出方向的边
    for u, v in highlighted_unknown_edges:
        connection_type = G[u][v].get("connection_type", "direct")
        style = 'dashed' if connection_type == "indirect" else 'solid'
        
        # 如果是推断出方向的边，使用紫色箭头
        if G[u][v].get("direction_confidence") is not None:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], 
                width=3, 
                edge_color='purple',
                style=style,
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1"
            )
        else:
            # 否则使用无箭头高亮连线
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], 
                width=3, 
                edge_color='purple',
                style=style,
                arrows=False,
                connectionstyle="arc3,rad=0.1"
            )
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt

# --------- 6. 执行演示 ---------

def run_demonstration():
    """运行演示程序"""
    print("=== NetworkX CAD 关系分析系统验证 ===")
    
    # 1. 生成模拟数据
    blocks, connections = generate_mock_cad_data()
    print(f"生成了 {len(blocks)} 个块和 {len(connections)} 个连接")
    
    # 2. 构建图
    G = build_graph(blocks, connections)
    print(f"构建图完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    
    # 统计已知方向和未知方向的边
    known_edges = sum(1 for _, _, data in G.edges(data=True) if data.get("has_direction", True))
    unknown_edges = G.number_of_edges() - known_edges
    print(f"其中已知方向的边: {known_edges}, 未知方向的边: {unknown_edges}")
    
    # 3. 创建查询引擎
    recursive_query = RecursiveQueryEngine(G)
    iterative_query = IterativeQueryEngine(G)
    
    # 4. 创建边方向推理引擎
    direction_inference = EdgeDirectionInference(G)
    
    # 5. 推断边方向
    print("\n=== 边方向推理 ===")
    inferred_directions = direction_inference.infer_edge_directions(confidence_threshold=0.6)
    
    # 打印推断结果
    print("\n推断结果:")
    for (u, v), confidence in inferred_directions.items():
        print(f"推断边 {u} → {v} 的方向置信度为 {confidence:.2f}")
    
    # 6. 应用推断的方向
    G_updated = direction_inference.apply_inferred_directions(inferred_directions)
    
    # 统计更新后的已知方向和未知方向的边
    known_edges_updated = sum(1 for _, _, data in G_updated.edges(data=True) if data.get("has_direction", True))
    unknown_edges_updated = G_updated.number_of_edges() - known_edges_updated
    print(f"\n应用推断后，已知方向的边: {known_edges_updated}, 未知方向的边: {unknown_edges_updated}")
    print(f"成功推断了 {known_edges_updated - known_edges} 条边的方向")
    
    # 7. 在同一个框中显示两个图
    plt.figure(figsize=(16, 8))  # 调整图的大小
    
    # 左侧显示原始图
    plt.subplot(1, 2, 1)
    pos = {node: data["position"] for node, data in G.nodes(data=True)}
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.8)  # 减小节点大小
    
    # 绘制已知方向的边
    known_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get("has_direction", True)]
    nx.draw_networkx_edges(
        G, pos, edgelist=known_edges, 
        width=1.5,  # 减小线宽
        edge_color='gray',
        style='solid',
        arrows=True,
        arrowsize=15,  # 减小箭头大小
        connectionstyle="arc3,rad=0.1"
    )
    
    # 绘制未知方向的边 (无箭头)
    unknown_edges = [(u, v) for u, v, data in G.edges(data=True) if not data.get("has_direction", True)]
    for u, v in unknown_edges:
        connection_type = G[u][v].get("connection_type", "direct")
        style = 'dashed' if connection_type == "indirect" else 'solid'
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            width=1.5,  # 减小线宽
            edge_color='blue',
            style=style,
            arrows=False,
            connectionstyle="arc3,rad=0.1"
        )
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')  # 减小字体大小
    plt.title("原始CAD关系图 (蓝色无箭头连线表示未知方向)")
    plt.axis('off')
    
    # 右侧显示更新后的图
    plt.subplot(1, 2, 2)
    
    # 绘制节点
    nx.draw_networkx_nodes(G_updated, pos, node_size=500, node_color='skyblue', alpha=0.8)  # 减小节点大小
    
    # 绘制原本就已知方向的边
    original_known_edges = [(u, v) for u, v, data in G_updated.edges(data=True) 
                           if data.get("has_direction", True) and data.get("direction_confidence") is None]
    nx.draw_networkx_edges(
        G_updated, pos, edgelist=original_known_edges, 
        width=1.5,  # 减小线宽
        edge_color='gray',
        style='solid',
        arrows=True,
        arrowsize=15,  # 减小箭头大小
        connectionstyle="arc3,rad=0.1"
    )
    
    # 绘制推断出方向的边 (紫色箭头)
    inferred_edges = [(u, v) for u, v, data in G_updated.edges(data=True) if data.get("direction_confidence") is not None]
    nx.draw_networkx_edges(
        G_updated, pos, edgelist=inferred_edges, 
        width=2.5,  # 加粗线宽以突出显示
        edge_color='purple',
        style='solid',
        arrows=True,
        arrowsize=20,  # 加大箭头以突出显示
        connectionstyle="arc3,rad=0.1"
    )
    
    # 绘制仍然未知方向的边 (无箭头)
    still_unknown_edges = [(u, v) for u, v, data in G_updated.edges(data=True) if not data.get("has_direction", True)]
    for u, v in still_unknown_edges:
        connection_type = G_updated[u][v].get("connection_type", "direct")
        style = 'dashed' if connection_type == "indirect" else 'solid'
        nx.draw_networkx_edges(
            G_updated, pos, edgelist=[(u, v)], 
            width=1.5,  # 减小线宽
            edge_color='blue',
            style=style,
            arrows=False,
            connectionstyle="arc3,rad=0.1"
        )
    
    # 绘制标签
    nx.draw_networkx_labels(G_updated, pos, font_size=9, font_family='sans-serif')  # 减小字体大小
    plt.title("应用边方向推理后的图 (紫色箭头表示推断的边)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("direction_inference_comparison.png", dpi=300, bbox_inches='tight')  # 保存高分辨率图像
    plt.show()
    
    # 8. 执行递归查询演示
    print("\n=== 递归式出度查询 ===")
    source_block = "B1"
    max_depth = 3
    
    print(f"从 {source_block} 开始递归查询出度连接 (最大深度: {max_depth})")
    
    # 使用更新后的图创建新的查询引擎
    updated_query = IterativeQueryEngine(G_updated)
    result = updated_query.get_outgoing_recursive(source_block, max_depth)
    
    print(f"找到 {len(result['connections'])} 个连接, {len(result['nodes'])} 个相关节点")
    
    # 打印连接情况
    for conn in result["connections"]:
        confidence = ""
        if G_updated[conn['source_id']][conn['target_id']].get('direction_confidence'):
            confidence = f" (置信度: {G_updated[conn['source_id']][conn['target_id']]['direction_confidence']:.2f})"
        print(f"深度 {conn['depth']}: {conn['source_id']} → {conn['target_id']}{confidence}")
    
    return G, G_updated, direction_inference

# --------- 7. 边方向推理 ---------

class EdgeDirectionInference:
    """根据已知方向的边推断未知方向的边"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        
    def infer_edge_directions(self, confidence_threshold: float = 0.4):
        """
        推断图中未知方向边的方向
        
        Args:
            confidence_threshold: 接受推断结果的置信度阈值
            
        Returns:
            字典，键为边(u,v)，值为该方向的置信度
        """
        # 创建一个新图，只包含已知方向的边
        known_graph = nx.DiGraph()
        
        # 未知方向的边列表
        unknown_edges = []
        
        # 分离已知方向和未知方向的边
        for u, v, data in self.graph.edges(data=True):
            if data.get("has_direction", True):
                known_graph.add_edge(u, v)
            else:
                unknown_edges.append((u, v))
        
        print(f"已知方向的边数量: {known_graph.number_of_edges()}")
        print(f"未知方向的边数量: {len(unknown_edges)}")
        
        # 推断结果
        inferred_directions = {}
        
        # 使用NetworkX的内置方法进行推断
        
        # 1. 使用PageRank算法确定节点的重要性
        try:
            pagerank = nx.pagerank(known_graph)
            
            # 对于未知方向的边，从PageRank值高的节点指向值低的节点
            for u, v in unknown_edges:
                if u in pagerank and v in pagerank:
                    # 如果u的PageRank值高于v，则方向为u->v
                    if pagerank[u] > pagerank[v]:
                        confidence = min(0.5 + (pagerank[u] - pagerank[v]) * 5, 0.9)  # 缩放到合理范围
                        inferred_directions[(u, v)] = confidence
                        print(f"基于PageRank推断: {u} → {v}, 置信度={confidence:.2f}")
                    # 如果v的PageRank值高于u，则方向为v->u
                    elif pagerank[v] > pagerank[u]:
                        confidence = min(0.5 + (pagerank[v] - pagerank[u]) * 5, 0.9)  # 缩放到合理范围
                        inferred_directions[(v, u)] = confidence
                        print(f"基于PageRank推断: {v} → {u}, 置信度={confidence:.2f}")
        except:
            print("PageRank算法失败，可能是图不连通")
        
        # 2. 使用最短路径分析
        for u, v in unknown_edges:
            if (u, v) in inferred_directions or (v, u) in inferred_directions:
                continue  # 已经推断过的边跳过
                
            # 计算所有节点对之间的最短路径
            u_to_v_paths = 0
            v_to_u_paths = 0
            
            # 检查是否存在通过其他路径从u到v的路径
            for source in known_graph.nodes():
                for target in known_graph.nodes():
                    if source == u and target == v:
                        continue  # 跳过直接边
                    
                    try:
                        paths = list(nx.all_simple_paths(known_graph, source, target, cutoff=4))
                        for path in paths:
                            if u in path and v in path and path.index(u) < path.index(v):
                                u_to_v_paths += 1
                            elif u in path and v in path and path.index(v) < path.index(u):
                                v_to_u_paths += 1
                    except:
                        pass
            
            # 如果存在明显的方向趋势
            total_paths = u_to_v_paths + v_to_u_paths
            if total_paths > 0:
                direction_ratio = abs(u_to_v_paths - v_to_u_paths) / total_paths
                if direction_ratio >= confidence_threshold:
                    if u_to_v_paths > v_to_u_paths:
                        confidence = min(0.5 + direction_ratio * 0.4, 0.9)
                        inferred_directions[(u, v)] = confidence
                        print(f"基于路径分析推断: {u} → {v}, 置信度={confidence:.2f}")
                    else:
                        confidence = min(0.5 + direction_ratio * 0.4, 0.9)
                        inferred_directions[(v, u)] = confidence
                        print(f"基于路径分析推断: {v} → {u}, 置信度={confidence:.2f}")
        
        # 3. 使用拓扑排序启发式
        try:
            # 尝试对已知图进行拓扑排序
            topo_order = list(nx.topological_sort(known_graph))
            
            # 对于未知方向的边，从拓扑顺序靠前的节点指向靠后的节点
            for u, v in unknown_edges:
                if (u, v) in inferred_directions or (v, u) in inferred_directions:
                    continue  # 已经推断过的边跳过
                    
                if u in topo_order and v in topo_order:
                    u_pos = topo_order.index(u)
                    v_pos = topo_order.index(v)
                    
                    # 位置差越大，置信度越高
                    pos_diff = abs(u_pos - v_pos) / len(topo_order)
                    
                    if u_pos < v_pos:  # u在拓扑排序中排在v前面
                        confidence = min(0.5 + pos_diff * 0.4, 0.9)
                        inferred_directions[(u, v)] = confidence
                        print(f"基于拓扑排序推断: {u} → {v}, 置信度={confidence:.2f}")
                    else:
                        confidence = min(0.5 + pos_diff * 0.4, 0.9)
                        inferred_directions[(v, u)] = confidence
                        print(f"基于拓扑排序推断: {v} → {u}, 置信度={confidence:.2f}")
        except:
            print("拓扑排序失败，可能是图中存在环")
        
        # 如果以上方法都没有推断出结果，使用自定义启发式方法
        if not inferred_directions:
            print("使用自定义启发式方法进行推断...")
            for u, v in unknown_edges:
                # 使用多种启发式方法推断方向
                path_score = self._infer_by_path_connectivity(known_graph, u, v)
                structure_score = self._infer_by_local_structure(known_graph, u, v)
                flow_score = self._infer_by_global_flow(known_graph, u, v)
                
                # 综合多种方法的结果
                combined_score = (path_score + structure_score + flow_score) / 3
                
                # 如果置信度超过阈值，接受该推断
                if abs(combined_score) >= confidence_threshold:
                    if combined_score > 0:
                        inferred_directions[(u, v)] = abs(combined_score)
                        print(f"自定义方法推断: {u} → {v}, 置信度={abs(combined_score):.2f}")
                    else:
                        inferred_directions[(v, u)] = abs(combined_score)
                        print(f"自定义方法推断: {v} → {u}, 置信度={abs(combined_score):.2f}")
        
        # 如果仍然没有推断出任何边的方向，强制推断一些
        if not inferred_directions and unknown_edges:
            print("强制推断一些边的方向...")
            for u, v in unknown_edges:
                # 简单地假设从字母序较小的节点指向较大的节点
                if u < v:
                    inferred_directions[(u, v)] = 0.6
                    print(f"强制推断: {u} → {v}, 置信度=0.6")
                else:
                    inferred_directions[(v, u)] = 0.6
                    print(f"强制推断: {v} → {u}, 置信度=0.6")
        
        return inferred_directions
    
    def _infer_by_path_connectivity(self, known_graph: nx.DiGraph, u: str, v: str) -> float:
        """
        基于路径连通性推断方向
        
        如果从u到v有多条路径，但从v到u没有，则倾向于u->v方向
        返回值在[-1,1]之间，正值表示u->v方向，负值表示v->u方向
        """
        # 检查u到v的路径
        u_to_v_paths = self._count_paths(known_graph, u, v)
        
        # 检查v到u的路径
        v_to_u_paths = self._count_paths(known_graph, v, u)
        
        # 计算方向得分
        total_paths = u_to_v_paths + v_to_u_paths
        if total_paths == 0:
            return 0
        
        return (u_to_v_paths - v_to_u_paths) / total_paths
    
    def _count_paths(self, graph: nx.DiGraph, source: str, target: str, max_paths: int = 10) -> int:
        """计算从source到target的路径数量，限制最大数量以提高效率"""
        try:
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
            return min(len(paths), max_paths)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0
    
    def _infer_by_local_structure(self, known_graph: nx.DiGraph, u: str, v: str) -> float:
        """
        基于局部结构推断方向
        
        分析u和v的入度和出度模式，推断可能的方向
        """
        # 如果节点不在已知图中，返回0
        if u not in known_graph or v not in known_graph:
            return 0
        
        # 获取u和v的入度和出度
        u_in = known_graph.in_degree(u)
        u_out = known_graph.out_degree(u)
        v_in = known_graph.in_degree(v)
        v_out = known_graph.out_degree(v)
        
        # 如果u主要是出度节点，v主要是入度节点，倾向于u->v
        if u_out > u_in and v_in > v_out:
            return 0.8
        # 如果v主要是出度节点，u主要是入度节点，倾向于v->u
        elif v_out > v_in and u_in > u_out:
            return -0.8
        
        # 计算更细致的得分
        u_ratio = u_out / max(u_in, 1)
        v_ratio = v_in / max(v_out, 1)
        
        # 归一化到[-1,1]
        score = (u_ratio - v_ratio) / (u_ratio + v_ratio) if (u_ratio + v_ratio) > 0 else 0
        return max(min(score, 1), -1)  # 限制在[-1,1]范围内
    
    def _infer_by_global_flow(self, known_graph: nx.DiGraph, u: str, v: str) -> float:
        """
        基于全局流向推断方向
        
        分析图的整体流向，如从源节点到汇节点的趋势
        """
        # 识别可能的源节点和汇节点
        sources = [node for node in known_graph.nodes() if known_graph.in_degree(node) == 0 and known_graph.out_degree(node) > 0]
        sinks = [node for node in known_graph.nodes() if known_graph.out_degree(node) == 0 and known_graph.in_degree(node) > 0]
        
        if not sources or not sinks:
            return 0
        
        # 计算u和v到源节点和汇节点的平均距离
        u_to_sources = self._average_distance(known_graph, u, sources)
        u_to_sinks = self._average_distance(known_graph, u, sinks)
        v_to_sources = self._average_distance(known_graph, v, sources)
        v_to_sinks = self._average_distance(known_graph, v, sinks)
        
        # 如果u更接近源节点，v更接近汇节点，倾向于u->v
        if u_to_sources < v_to_sources and u_to_sinks > v_to_sinks:
            return 0.7
        # 如果v更接近源节点，u更接近汇节点，倾向于v->u
        elif v_to_sources < u_to_sources and v_to_sinks > u_to_sinks:
            return -0.7
        
        # 计算更细致的得分
        source_diff = v_to_sources - u_to_sources
        sink_diff = u_to_sinks - v_to_sinks
        
        # 归一化到[-1,1]
        score = (source_diff + sink_diff) / 2
        return max(min(score / 5, 1), -1)  # 除以5进行缩放，并限制在[-1,1]范围内
    
    def _average_distance(self, graph: nx.DiGraph, node: str, target_nodes: List[str]) -> float:
        """计算一个节点到目标节点列表的平均距离"""
        if not target_nodes or node not in graph:
            return float('inf')
        
        distances = []
        for target in target_nodes:
            try:
                dist = nx.shortest_path_length(graph, node, target)
                distances.append(dist)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        
        return sum(distances) / len(distances) if distances else float('inf')
    
    def apply_inferred_directions(self, inferred_directions: Dict[Tuple[str, str], float]) -> nx.DiGraph:
        """
        将推断的方向应用到图中
        
        Args:
            inferred_directions: 推断的方向字典
            
        Returns:
            更新后的图
        """
        # 创建原图的副本
        updated_graph = self.graph.copy()
        
        # 应用推断的方向
        for (u, v), confidence in inferred_directions.items():
            # 检查边是否存在（可能是u,v或v,u）
            if updated_graph.has_edge(u, v):
                # 更新边属性
                updated_graph[u][v]['has_direction'] = True
                updated_graph[u][v]['direction_confidence'] = confidence
            elif updated_graph.has_edge(v, u):
                # 如果边是反向存储的，需要反转边
                edge_data = updated_graph[v][u].copy()
                updated_graph.remove_edge(v, u)
                edge_data['has_direction'] = True
                edge_data['direction_confidence'] = confidence
                updated_graph.add_edge(u, v, **edge_data)
        
        return updated_graph

# 如果作为脚本运行，执行演示
if __name__ == "__main__":
    G, G_updated, direction_inference = run_demonstration()

