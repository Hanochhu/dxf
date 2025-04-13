"""
图构建模块
提供图表示的构建和操作功能
"""

import json
import os
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Any, Union

from src.core.data_structures import Block, Connection


class CADGraph:
    """
    CAD图形结构类

    本类提供图表示的构建和操作功能，整合了以下核心功能：
    1. 图构建与管理：从块和连接构建有向图表示
    2. 路径查找：查找节点间的路径和所有可能路径
    3. 环路检测：识别图中的环路结构
    4. 连通性分析：分析图的连通性和关键节点
    5. 数据导入导出：支持JSON格式的导入导出
    6. 社区检测：识别图中的社区结构

    注意: 此类整合了原PathFinder类的功能，统一提供图分析相关操作。
    """

    def __init__(self):
        """初始化CAD图形"""
        self.graph = nx.DiGraph()
        self.blocks = {}  # 块ID到块对象的映射
        self.connections = {}  # 连接ID到连接对象的映射

    def build_from_blocks_connections(
        self, blocks: List[Block], connections: List[Connection]
    ):
        """
        从块和连接构建图

        Args:
            blocks: 块列表
            connections: 连接列表
        """
        # 清空现有图
        self.graph = nx.DiGraph()
        self.blocks = {}
        self.connections = {}

        # 添加块作为节点
        for block in blocks:
            self.add_block(block)

        # 添加连接作为边
        for connection in connections:
            self.add_connection(connection)

    def add_block(self, block: Block):
        """
        将块作为节点添加到图中

        Args:
            block: 块对象
        """
        self.blocks[block.id] = block

        # 添加节点属性
        attrs = {
            "name": block.name,
            "entity_count": len(block.entities),
            "is_arrow": block.is_arrow,
        }

        # 添加边界框信息（如果有）
        if block.bounding_box:
            attrs.update(
                {
                    "bbox_min_x": block.bounding_box.min_point.x,
                    "bbox_min_y": block.bounding_box.min_point.y,
                    "bbox_max_x": block.bounding_box.max_point.x,
                    "bbox_max_y": block.bounding_box.max_point.y,
                    "bbox_width": block.bounding_box.width,
                    "bbox_height": block.bounding_box.height,
                    "bbox_aspect_ratio": block.bounding_box.aspect_ratio,
                }
            )

        # 添加中心点信息（如果有）
        if block.center:
            attrs.update(
                {
                    "center_x": block.center.x,
                    "center_y": block.center.y,
                    "center_z": block.center.z,
                }
            )

        # 添加引用信息（如果有）
        if block.reference:
            attrs.update(
                {
                    "ref_name": block.reference.name,
                    "ref_position_x": block.reference.position.x,
                    "ref_position_y": block.reference.position.y,
                    "ref_position_z": block.reference.position.z,
                    "ref_rotation": block.reference.rotation,
                    "ref_scale": block.reference.scale,
                }
            )

        self.graph.add_node(block.id, **attrs)

    def add_connection(self, connection: Connection):
        """
        将连接作为边添加到图中

        Args:
            connection: 连接对象
        """
        self.connections[connection.id] = connection

        # 根据连接属性添加方向
        source_id = connection.source_block.id
        target_id = connection.target_block.id

        # 构建边属性
        edge_attrs = {
            "connection": connection,
            "connection_id": connection.id,
            "explicit_direction": connection.has_explicit_direction,
            "connection_type": connection.connection_type,
            "segments": len(connection.path_segments),
        }

        # 添加方向信息
        if hasattr(connection, "direction"):
            direction = connection.direction
            edge_attrs.update(
                {
                    "direction_x": direction[0],
                    "direction_y": direction[1],
                    "direction_z": direction[2],
                }
            )

        self.graph.add_edge(source_id, target_id, **edge_attrs)

    def get_block(self, block_id: str) -> Optional[Block]:
        """
        获取指定ID的块

        Args:
            block_id: 块ID

        Returns:
            Optional[Block]: 块对象，如果不存在则返回None
        """
        return self.blocks.get(block_id)

    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """
        获取指定ID的连接

        Args:
            connection_id: 连接ID

        Returns:
            Optional[Connection]: 连接对象，如果不存在则返回None
        """
        return self.connections.get(connection_id)

    def get_predecessors(self, block_id: str) -> List[Block]:
        """
        获取指向指定块的所有块

        Args:
            block_id: 块ID

        Returns:
            List[Block]: 前驱块列表
        """
        if block_id not in self.graph:
            return []

        predecessors = list(self.graph.predecessors(block_id))
        return [
            self.blocks[pred_id] for pred_id in predecessors if pred_id in self.blocks
        ]

    def get_successors(self, block_id: str) -> List[Block]:
        """
        获取指定块指向的所有块

        Args:
            block_id: 块ID

        Returns:
            List[Block]: 后继块列表
        """
        if block_id not in self.graph:
            return []

        successors = list(self.graph.successors(block_id))
        return [
            self.blocks[succ_id] for succ_id in successors if succ_id in self.blocks
        ]

    def get_in_degree(self, block_id: str) -> int:
        """
        获取块的入度（指向该块的连接数）

        Args:
            block_id: 块ID

        Returns:
            int: 入度
        """
        if block_id not in self.graph:
            return 0

        return self.graph.in_degree(block_id)

    def get_out_degree(self, block_id: str) -> int:
        """
        获取块的出度（从该块出发的连接数）

        Args:
            block_id: 块ID

        Returns:
            int: 出度
        """
        if block_id not in self.graph:
            return 0

        return self.graph.out_degree(block_id)

    def has_connection(self, source_id: str, target_id: str) -> bool:
        """
        检查源块和目标块之间是否有直接连接

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            bool: 是否有连接
        """
        return self.graph.has_edge(source_id, target_id)

    def get_path(self, source_id: str, target_id: str) -> List[str]:
        """
        查找从源块到目标块的最短路径

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            List[str]: 路径中的块ID列表，如果没有路径则为空列表
        """
        try:
            if nx.has_path(self.graph, source_id, target_id):
                return nx.shortest_path(self.graph, source_id, target_id)
            else:
                return []
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []

    def get_all_paths(
        self, source_id: str, target_id: str, cutoff: int = None
    ) -> List[List[str]]:
        """
        查找从源块到目标块的所有路径

        Args:
            source_id: 源块ID
            target_id: 目标块ID
            cutoff: 最大路径长度

        Returns:
            List[List[str]]: 所有路径的列表
        """
        try:
            return list(
                nx.all_simple_paths(self.graph, source_id, target_id, cutoff=cutoff)
            )
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []

    def find_cycles(self) -> List[List[str]]:
        """
        查找图中的所有环路
        （原PathFinder类功能）

        Returns:
            List[List[str]]: 环路列表
        """
        try:
            return list(nx.simple_cycles(self.graph))
        except:
            # 如果图不支持查找环路（例如无向图），使用替代方法
            cycles = []
            for node in self.graph.nodes():
                try:
                    for cycle in nx.find_cycle(self.graph, source=node):
                        path = [node]
                        current = node
                        while True:
                            current = cycle[current]
                            if current == node:
                                break
                            path.append(current)
                        cycles.append(path)
                except:
                    pass
            return cycles

    def find_critical_nodes(self) -> List[str]:
        """
        查找图中的关键节点（删除后会增加连通分量数量的节点）
        （原PathFinder类功能）

        Returns:
            List[str]: 关键节点ID列表
        """
        try:
            return list(nx.articulation_points(self.graph.to_undirected()))
        except:
            # 简单版实现
            critical_nodes = []
            original_components = nx.number_connected_components(
                self.graph.to_undirected()
            )

            for node in self.graph.nodes():
                # 创建图的副本
                G_copy = self.graph.copy()

                # 移除当前节点
                G_copy.remove_node(node)

                # 检查连通分量是否增加
                new_components = nx.number_connected_components(G_copy.to_undirected())

                if new_components > original_components:
                    critical_nodes.append(node)

            return critical_nodes

    def analyze_connectivity(self) -> Dict:
        """
        分析图的连通性
        （原PathFinder类功能）

        Returns:
            Dict: 连通性分析结果
        """
        result = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
        }

        # 检查是否为有向图
        if isinstance(self.graph, nx.DiGraph):
            # 分析强连通分量
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            result["strongly_connected_components"] = len(strongly_connected)

            if strongly_connected:
                result["largest_strongly_connected_size"] = max(
                    len(c) for c in strongly_connected
                )

            # 分析弱连通分量
            weakly_connected = list(nx.weakly_connected_components(self.graph))
            result["weakly_connected_components"] = len(weakly_connected)

            if weakly_connected:
                result["largest_weakly_connected_size"] = max(
                    len(c) for c in weakly_connected
                )
        else:
            # 分析连通分量
            connected = list(nx.connected_components(self.graph))
            result["connected_components"] = len(connected)

            if connected:
                result["largest_connected_size"] = max(len(c) for c in connected)

        # 计算平均路径长度（如果图是连通的）
        try:
            result["average_shortest_path_length"] = nx.average_shortest_path_length(
                self.graph
            )
        except:
            # 图可能不是连通的
            result["average_shortest_path_length"] = None

        # 计算图密度
        result["density"] = nx.density(self.graph)

        return result

    def to_networkx(self) -> nx.DiGraph:
        """
        获取底层NetworkX图

        Returns:
            nx.DiGraph: NetworkX图对象
        """
        return self.graph

    def get_central_blocks(
        self, top_n: int = 5, method: str = "degree"
    ) -> List[Tuple[str, float]]:
        """
        获取图中最中心的块

        Args:
            top_n: 返回的块数量
            method: 中心性度量方法，可选 'degree', 'betweenness', 'closeness', 'eigenvector'

        Returns:
            List[Tuple[str, float]]: 块ID和中心性值的元组列表
        """
        centrality = {}

        try:
            if method == "degree":
                centrality = nx.degree_centrality(self.graph)
            elif method == "betweenness":
                centrality = nx.betweenness_centrality(self.graph)
            elif method == "closeness":
                centrality = nx.closeness_centrality(self.graph)
            elif method == "eigenvector":
                centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            else:
                # 默认使用度中心性
                centrality = nx.degree_centrality(self.graph)
        except:
            # 如果计算失败（例如图不是强连通的），退回到简单的度量
            centrality = {node: self.graph.degree(node) for node in self.graph.nodes()}

        # 排序并返回前N个
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def find_communities(self, method: str = "louvain") -> Dict[str, int]:
        """
        查找图中的社区（社团）

        Args:
            method: 社区检测方法，可选 'louvain', 'label_propagation', 'greedy_modularity'

        Returns:
            Dict[str, int]: 块ID到社区ID的映射
        """
        # 将有向图转换为无向图进行社区检测
        undirected = self.graph.to_undirected()

        try:
            if method == "louvain":
                # 尝试导入社区检测库
                try:
                    import community as community_louvain

                    communities = community_louvain.best_partition(undirected)
                except ImportError:
                    # 退回到标签传播
                    communities = self._label_propagation_communities(undirected)

            elif method == "label_propagation":
                communities = self._label_propagation_communities(undirected)

            elif method == "greedy_modularity":
                try:
                    communities = nx.community.greedy_modularity_communities(undirected)
                    # 转换为块ID到社区ID的映射
                    community_map = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            community_map[node] = i
                    communities = community_map
                except:
                    # 退回到标签传播
                    communities = self._label_propagation_communities(undirected)
            else:
                # 默认使用标签传播
                communities = self._label_propagation_communities(undirected)

        except:
            # 如果所有方法都失败，使用一个简单的基于连通分量的社区划分
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected)):
                for node in component:
                    communities[node] = i

        return communities

    def _label_propagation_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """
        使用标签传播算法进行社区检测

        Args:
            graph: 无向图

        Returns:
            Dict[str, int]: 节点ID到社区ID的映射
        """
        # 简单实现的标签传播
        import random

        # 初始化：每个节点属于自己的社区
        labels = {node: i for i, node in enumerate(graph.nodes())}

        # 最大迭代次数
        max_iterations = 10

        # 创建节点列表（便于随机化）
        nodes = list(graph.nodes())

        for _ in range(max_iterations):
            # 随机排序节点
            random.shuffle(nodes)

            # 跟踪是否有变化
            changes = 0

            for node in nodes:
                # 收集邻居的标签
                neighbor_labels = {}
                for neighbor in graph.neighbors(node):
                    label = labels[neighbor]
                    neighbor_labels[label] = neighbor_labels.get(label, 0) + 1

                if not neighbor_labels:
                    continue

                # 找到最常见的标签
                max_count = max(neighbor_labels.values())
                best_labels = [
                    label
                    for label, count in neighbor_labels.items()
                    if count == max_count
                ]

                # 随机选择一个最佳标签
                new_label = random.choice(best_labels)

                # 如果标签改变，记录变化
                if new_label != labels[node]:
                    labels[node] = new_label
                    changes += 1

            # 如果没有变化，则收敛
            if changes == 0:
                break

        return labels

    def export_to_json(self, file_path: str) -> bool:
        """
        将图结构导出为JSON

        Args:
            file_path: 文件路径

        Returns:
            bool: 操作是否成功
        """
        try:
            # 准备导出数据
            data = {"blocks": {}, "connections": {}}

            # 导出块信息
            for block_id, block in self.blocks.items():
                # 仅导出块的基本信息，而不是整个对象
                block_data = {
                    "id": block.id,
                    "name": block.name,
                    "entity_count": len(block.entities),
                    "is_arrow": block.is_arrow,
                }

                # 添加边界框信息（如果有）
                if block.bounding_box:
                    block_data["bounding_box"] = {
                        "min": [
                            block.bounding_box.min_point.x,
                            block.bounding_box.min_point.y,
                            block.bounding_box.min_point.z,
                        ],
                        "max": [
                            block.bounding_box.max_point.x,
                            block.bounding_box.max_point.y,
                            block.bounding_box.max_point.z,
                        ],
                        "width": block.bounding_box.width,
                        "height": block.bounding_box.height,
                        "aspect_ratio": block.bounding_box.aspect_ratio,
                    }

                data["blocks"][block_id] = block_data

            # 导出连接信息
            for conn_id, conn in self.connections.items():
                conn_data = {
                    "id": conn.id,
                    "source_block_id": conn.source_block.id,
                    "target_block_id": conn.target_block.id,
                    "has_explicit_direction": conn.has_explicit_direction,
                    "connection_type": conn.connection_type,
                    "segment_count": len(conn.path_segments),
                }

                # 添加路径段的基本信息
                conn_data["segments"] = []
                for segment in conn.path_segments:
                    segment_info = {
                        "id": segment.id,
                        "start": [
                            segment.start_point.x,
                            segment.start_point.y,
                            segment.start_point.z,
                        ],
                        "end": [
                            segment.end_point.x,
                            segment.end_point.y,
                            segment.end_point.z,
                        ],
                        "length": segment.get_length(),
                    }
                    conn_data["segments"].append(segment_info)

                data["connections"][conn_id] = conn_data

            # 导出图的拓扑信息
            data["graph"] = {
                "nodes": list(self.graph.nodes()),
                "edges": list(self.graph.edges()),
            }

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"导出图结构时出错: {e}")
            return False

    def import_from_json(self, file_path: str) -> bool:
        """
        从JSON导入图结构

        Args:
            file_path: 文件路径

        Returns:
            bool: 操作是否成功
        """
        try:
            # 读取文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 创建新图
            self.graph = nx.DiGraph()

            # 导入节点
            for node_id in data.get("graph", {}).get("nodes", []):
                block_data = data.get("blocks", {}).get(node_id, {"id": node_id})
                self.graph.add_node(node_id, **block_data)

            # 导入边
            for source, target in data.get("graph", {}).get("edges", []):
                conn_id = f"{source}_to_{target}"
                for conn_id_check, conn_data in data.get("connections", {}).items():
                    if (
                        conn_data.get("source_block_id") == source
                        and conn_data.get("target_block_id") == target
                    ):
                        conn_id = conn_id_check
                        break

                conn_data = data.get("connections", {}).get(conn_id, {})
                self.graph.add_edge(source, target, **conn_data)

            return True

        except Exception as e:
            print(f"导入图结构时出错: {e}")
            return False

    def get_subgraph(self, nodes: List[str]) -> "CADGraph":
        """
        获取包含指定节点的子图

        Args:
            nodes: 节点ID列表

        Returns:
            CADGraph: 子图
        """
        # 创建子图
        subgraph = CADGraph()

        # 获取NetworkX子图
        nx_subgraph = self.graph.subgraph(nodes)

        # 添加节点
        for node in nx_subgraph.nodes():
            if node in self.blocks:
                subgraph.add_block(self.blocks[node])

        # 添加边
        for source, target, data in nx_subgraph.edges(data=True):
            if "connection_id" in data and data["connection_id"] in self.connections:
                subgraph.add_connection(self.connections[data["connection_id"]])

        return subgraph

    def get_statistics(self) -> Dict:
        """
        获取图的统计信息
        整合了原来的统计功能和连通性分析功能

        Returns:
            Dict: 统计信息
        """
        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "arrow_count": sum(
                1
                for _, data in self.graph.nodes(data=True)
                if data.get("is_arrow", False)
            ),
            "directed_connection_count": sum(
                1
                for _, _, data in self.graph.edges(data=True)
                if data.get("explicit_direction", False)
            ),
            "indirect_connection_count": sum(
                1
                for _, _, data in self.graph.edges(data=True)
                if data.get("connection_type", "") == "indirect"
            ),
            "average_block_size": 0,
            "average_connection_length": 0,
        }

        # 计算平均块尺寸
        block_sizes = []
        for _, data in self.graph.nodes(data=True):
            if "bbox_width" in data and "bbox_height" in data:
                block_sizes.append(data["bbox_width"] * data["bbox_height"])

        if block_sizes:
            stats["average_block_size"] = sum(block_sizes) / len(block_sizes)

        # 计算平均连接长度
        segment_counts = []
        for _, _, data in self.graph.edges(data=True):
            if "segments" in data:
                segment_counts.append(data["segments"])

        if segment_counts:
            stats["average_connection_segments"] = sum(segment_counts) / len(
                segment_counts
            )

        # 计算图密度
        stats["graph_density"] = nx.density(self.graph)

        # 检查图是否连通
        if not nx.is_empty(self.graph):
            # 对于有向图
            if isinstance(self.graph, nx.DiGraph):
                stats["is_strongly_connected"] = nx.is_strongly_connected(self.graph)
                stats["is_weakly_connected"] = nx.is_weakly_connected(self.graph)
                stats["strongly_connected_components"] = (
                    nx.number_strongly_connected_components(self.graph)
                )
                stats["weakly_connected_components"] = (
                    nx.number_weakly_connected_components(self.graph)
                )
            # 对于无向图
            else:
                stats["is_connected"] = nx.is_connected(self.graph)
                stats["connected_components"] = nx.number_connected_components(
                    self.graph
                )

        # 获取更详细的连通性分析
        connectivity_analysis = self.analyze_connectivity()
        stats.update(connectivity_analysis)

        return stats

    def analyze_graph(self) -> Dict:
        """
        全面分析图结构的各个方面

        此方法整合了原ConnectionAnalyzer.analyze_connections_graph的功能，
        作为CADGraph类功能的一部分，以保持所有图分析操作在同一个类中。
        ConnectionAnalyzer类现在会调用此方法而不是自己实现图分析逻辑。

        Returns:
            Dict: 图分析结果，包括连通性、环路、关键节点、中心块和社区结构
        """
        # 分析结果
        results = {}

        # 连通性分析
        results["connectivity"] = self.analyze_connectivity()

        # 查找环路
        cycles = self.find_cycles()
        results["cycles"] = {"count": len(cycles), "cycles": cycles}

        # 查找关键节点
        critical_nodes = self.find_critical_nodes()
        results["critical_nodes"] = {
            "count": len(critical_nodes),
            "nodes": critical_nodes,
        }

        # 中心性分析
        central_blocks = self.get_central_blocks(top_n=5)
        results["central_blocks"] = {"blocks": central_blocks}

        # 社区检测
        communities = self.find_communities()
        community_counts = {}
        for _, community_id in communities.items():
            community_counts[community_id] = community_counts.get(community_id, 0) + 1

        results["communities"] = {
            "count": len(set(communities.values())),
            "distribution": community_counts,
        }

        return results


class GraphVisualizer:
    """图可视化工具"""

    def __init__(self, cad_graph: CADGraph):
        """
        初始化图可视化工具

        Args:
            cad_graph: CAD图对象
        """
        self.cad_graph = cad_graph

    def export_graphviz(self, file_path: str, format: str = "dot") -> bool:
        """
        导出Graphviz格式的图表示

        Args:
            file_path: 输出文件路径
            format: 输出格式，可选 'dot', 'gexf', 'gml'

        Returns:
            bool: 操作是否成功
        """
        try:
            graph = self.cad_graph.to_networkx()

            if format == "dot":
                # 使用pydot
                try:
                    from networkx.drawing.nx_pydot import write_dot

                    write_dot(graph, file_path)
                except ImportError:
                    # 如果pydot不可用，尝试使用nx自带的写入功能
                    nx.nx_agraph.write_dot(graph, file_path)

            elif format == "gexf":
                # GEXF (Graph Exchange XML Format)
                nx.write_gexf(graph, file_path)

            elif format == "gml":
                # GML (Graph Modeling Language)
                nx.write_gml(graph, file_path)

            else:
                # 默认使用GEXF
                nx.write_gexf(graph, file_path)

            return True

        except Exception as e:
            print(f"导出图形时出错: {e}")
            return False

    def generate_layout(
        self, algorithm: str = "spring"
    ) -> Dict[str, Tuple[float, float]]:
        """
        生成图的布局（节点位置）

        Args:
            algorithm: 布局算法，可选 'spring', 'circular', 'spectral', 'kamada_kawai'

        Returns:
            Dict[str, Tuple[float, float]]: 节点ID到位置坐标的映射
        """
        graph = self.cad_graph.to_networkx()

        try:
            if algorithm == "spring":
                return nx.spring_layout(graph)
            elif algorithm == "circular":
                return nx.circular_layout(graph)
            elif algorithm == "spectral":
                return nx.spectral_layout(graph)
            elif algorithm == "kamada_kawai":
                return nx.kamada_kawai_layout(graph)
            elif algorithm == "shell":
                return nx.shell_layout(graph)
            else:
                return nx.spring_layout(graph)
        except:
            # 如果算法失败，默认使用spring布局
            return nx.spring_layout(graph)

    def use_physical_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        使用块的物理位置作为节点布局

        Returns:
            Dict[str, Tuple[float, float]]: 节点ID到位置坐标的映射
        """
        positions = {}

        for node_id, node_data in self.cad_graph.graph.nodes(data=True):
            if "center_x" in node_data and "center_y" in node_data:
                positions[node_id] = (node_data["center_x"], node_data["center_y"])
            elif "ref_position_x" in node_data and "ref_position_y" in node_data:
                positions[node_id] = (
                    node_data["ref_position_x"],
                    node_data["ref_position_y"],
                )
            elif (
                "bbox_min_x" in node_data
                and "bbox_min_y" in node_data
                and "bbox_max_x" in node_data
                and "bbox_max_y" in node_data
            ):
                # 使用边界框中心点
                center_x = (node_data["bbox_min_x"] + node_data["bbox_max_x"]) / 2
                center_y = (node_data["bbox_min_y"] + node_data["bbox_max_y"]) / 2
                positions[node_id] = (center_x, center_y)

        # 对于没有位置信息的节点，使用spring布局
        missing_nodes = [
            node for node in self.cad_graph.graph.nodes() if node not in positions
        ]
        if missing_nodes:
            # 创建子图并计算布局
            subgraph = self.cad_graph.graph.subgraph(missing_nodes)
            subgraph_layout = nx.spring_layout(subgraph)

            # 合并布局
            positions.update(subgraph_layout)

        return positions

    def get_node_colors(
        self, attribute: str = "is_arrow", default_color: str = "#1f77b4"
    ) -> Dict[str, str]:
        """
        生成节点颜色映射

        Args:
            attribute: 用于颜色映射的节点属性
            default_color: 默认颜色

        Returns:
            Dict[str, str]: 节点ID到颜色的映射
        """
        colors = {}

        # 根据属性生成颜色
        for node_id, node_data in self.cad_graph.graph.nodes(data=True):
            if attribute in node_data:
                value = node_data[attribute]

                # 针对布尔类型属性
                if isinstance(value, bool):
                    colors[node_id] = "#ff7f0e" if value else "#1f77b4"

                # 针对数值类型属性（例如 'entity_count'）
                elif isinstance(value, (int, float)):
                    # 将值归一化到0-1范围
                    min_val = min(
                        node_data.get(attribute, 0)
                        for _, node_data in self.cad_graph.graph.nodes(data=True)
                        if attribute in node_data
                    )
                    max_val = max(
                        node_data.get(attribute, 0)
                        for _, node_data in self.cad_graph.graph.nodes(data=True)
                        if attribute in node_data
                    )

                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)

                        # 使用蓝到红的渐变色
                        red = int(255 * normalized)
                        blue = int(255 * (1 - normalized))
                        colors[node_id] = f"#{red:02x}00{blue:02x}"
                    else:
                        colors[node_id] = default_color

                # 针对字符串类型属性
                elif isinstance(value, str):
                    # 简单哈希为颜色
                    hash_val = hash(value) % 0xFFFFFF
                    colors[node_id] = f"#{hash_val:06x}"

                else:
                    colors[node_id] = default_color
            else:
                colors[node_id] = default_color

        return colors

    def get_edge_colors(
        self, attribute: str = "explicit_direction", default_color: str = "#000000"
    ) -> Dict[Tuple[str, str], str]:
        """
        生成边颜色映射

        Args:
            attribute: 用于颜色映射的边属性
            default_color: 默认颜色

        Returns:
            Dict[Tuple[str, str], str]: 边(源节点,目标节点)到颜色的映射
        """
        colors = {}

        # 根据属性生成颜色
        for source, target, edge_data in self.cad_graph.graph.edges(data=True):
            if attribute in edge_data:
                value = edge_data[attribute]

                # 针对布尔类型属性
                if isinstance(value, bool):
                    colors[(source, target)] = "#ff7f0e" if value else "#1f77b4"

                # 针对字符串类型属性（例如 'connection_type'）
                elif isinstance(value, str):
                    if value == "regular":
                        colors[(source, target)] = "#1f77b4"  # 蓝色
                    elif value == "indirect":
                        colors[(source, target)] = "#ff7f0e"  # 橙色
                    elif value == "special":
                        colors[(source, target)] = "#2ca02c"  # 绿色
                    else:
                        # 简单哈希为颜色
                        hash_val = hash(value) % 0xFFFFFF
                        colors[(source, target)] = f"#{hash_val:06x}"

                # 针对数值类型属性（例如 'segments'）
                elif isinstance(value, (int, float)):
                    # 将值归一化到0-1范围
                    min_val = min(
                        edge_data.get(attribute, 0)
                        for _, _, edge_data in self.cad_graph.graph.edges(data=True)
                        if attribute in edge_data
                    )
                    max_val = max(
                        edge_data.get(attribute, 0)
                        for _, _, edge_data in self.cad_graph.graph.edges(data=True)
                        if attribute in edge_data
                    )

                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)

                        # 使用从浅到深的蓝色渐变
                        intensity = int(255 * (0.3 + 0.7 * normalized))
                        colors[(source, target)] = f"#{0:02x}{intensity:02x}{255:02x}"
                    else:
                        colors[(source, target)] = default_color
            else:
                colors[(source, target)] = default_color

        return colors
