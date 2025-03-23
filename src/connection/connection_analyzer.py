"""
连接分析模块
提供块之间连接关系的分析功能
"""

import math
from typing import List, Dict, Tuple, Optional, Set, Any
import networkx as nx

from core.data_structures import (
    EntityType, Entity, Point, BoundingBox, Block, 
    LineEntity, Connection
)
from feature.block_identifier import BlockIdentifier


class ConnectionAnalyzer:
    """连接分析器"""
    
    def __init__(self, block_identifier: BlockIdentifier = None):
        """
        初始化连接分析器
        
        Args:
            block_identifier: 块识别器（可选）
        """
        self.block_identifier = block_identifier or BlockIdentifier()
        
        # 配置参数
        self.max_gap_distance = 10.0  # 最大间隙距离
        self.connection_angle_tolerance = 0.2  # 连接角度容差（弧度）
        self.block_connection_tolerance = 2.0  # 块连接容差
    
    def set_parameters(self, max_gap_distance: float = None, 
                      connection_angle_tolerance: float = None,
                      block_connection_tolerance: float = None):
        """
        设置分析参数
        
        Args:
            max_gap_distance: 最大间隙距离
            connection_angle_tolerance: 连接角度容差
            block_connection_tolerance: 块连接容差
        """
        if max_gap_distance is not None:
            self.max_gap_distance = max_gap_distance
        
        if connection_angle_tolerance is not None:
            self.connection_angle_tolerance = connection_angle_tolerance
        
        if block_connection_tolerance is not None:
            self.block_connection_tolerance = block_connection_tolerance
    
    def find_connections(self, blocks: List[Block], lines: List[LineEntity]) -> List[Connection]:
        """
        查找块之间的连接
        
        Args:
            blocks: 块列表
            lines: 线段列表
            
        Returns:
            List[Connection]: 连接列表
        """
        connections = []
        connection_id = 0
        
        # 首先查找直接连接
        for line in lines:
            source_block = self._find_connected_block(line.start_point, blocks)
            target_block = self._find_connected_block(line.end_point, blocks)
            
            if source_block and target_block and source_block.id != target_block.id:
                # 检查是否为已有连接的一部分
                existing_conn = self._find_existing_connection(connections, source_block, target_block)
                
                if existing_conn:
                    # 添加到现有连接
                    existing_conn.path_segments.append(line)
                else:
                    # 创建新连接
                    connection_id += 1
                    connections.append(Connection(
                        id=f"conn_{connection_id}",
                        source_block=source_block,
                        target_block=target_block,
                        path_segments=[line],
                        has_explicit_direction=False  # 后续确定
                    ))
        
        # 查找间接连接（有间隙的线段）
        indirect_connections = self._find_indirect_connections(blocks, lines, connections)
        connections.extend(indirect_connections)
        
        # 根据箭头块确定连接方向
        self._determine_connection_directions(connections, blocks)
        
        return connections
    
    def _find_connected_block(self, point: Point, blocks: List[Block]) -> Optional[Block]:
        """
        查找包含或非常接近点的块
        
        Args:
            point: 点
            blocks: 块列表
            
        Returns:
            Optional[Block]: 连接的块，如果没有找到则返回None
        """
        for block in blocks:
            if not block.bounding_box:
                continue
            
            # 检查点是否在块的边界框内
            if block.bounding_box.contains_point(point):
                return block
            
            # 检查点是否非常接近块的边界框
            extended_bbox = BoundingBox(
                Point(
                    block.bounding_box.min_point.x - self.block_connection_tolerance,
                    block.bounding_box.min_point.y - self.block_connection_tolerance,
                    block.bounding_box.min_point.z - self.block_connection_tolerance
                ),
                Point(
                    block.bounding_box.max_point.x + self.block_connection_tolerance,
                    block.bounding_box.max_point.y + self.block_connection_tolerance,
                    block.bounding_box.max_point.z + self.block_connection_tolerance
                )
            )
            
            if extended_bbox.contains_point(point):
                return block
        
        return None
    
    def _find_existing_connection(self, connections: List[Connection], 
                                 source: Block, target: Block) -> Optional[Connection]:
        """
        查找源块和目标块之间的已有连接
        
        Args:
            connections: 连接列表
            source: 源块
            target: 目标块
            
        Returns:
            Optional[Connection]: 现有连接，如果没有找到则返回None
        """
        for conn in connections:
            if ((conn.source_block.id == source.id and conn.target_block.id == target.id) or
                (conn.source_block.id == target.id and conn.target_block.id == source.id and not conn.has_explicit_direction)):
                return conn
        return None
    
    def _find_indirect_connections(self, blocks: List[Block], 
                                  lines: List[LineEntity], 
                                  direct_connections: List[Connection]) -> List[Connection]:
        """
        查找间接连接（带间隙或通过特殊符号）
        
        Args:
            blocks: 块列表
            lines: 线段列表
            direct_connections: 直接连接列表
            
        Returns:
            List[Connection]: 间接连接列表
        """
        indirect_connections = []
        
        # 分组未连接的线段
        connected_lines = set()
        for conn in direct_connections:
            for segment in conn.path_segments:
                connected_lines.add(segment.id)
        
        unconnected_lines = [line for line in lines if line.id not in connected_lines]
        
        # 尝试将小间隙的线段连接起来
        grouped_segments = self._group_aligned_segments(unconnected_lines)
        
        # 对于每组，尝试找到端点连接的块
        connection_id = len(direct_connections)
        for segment_group in grouped_segments:
            if len(segment_group) > 0:
                # 找到整个线段组的端点
                start_points = [segment.start_point for segment in segment_group]
                end_points = [segment.end_point for segment in segment_group]
                
                # 找到相距最远的两点
                max_distance = 0
                furthest_pair = (start_points[0], end_points[0])
                
                for start in start_points:
                    for end in end_points:
                        distance = start.distance_to(end)
                        if distance > max_distance:
                            max_distance = distance
                            furthest_pair = (start, end)
                
                # 检查这些端点是否连接到块
                source_block = self._find_connected_block(furthest_pair[0], blocks)
                target_block = self._find_connected_block(furthest_pair[1], blocks)
                
                if source_block and target_block and source_block.id != target_block.id:
                    connection_id += 1
                    indirect_connections.append(Connection(
                        id=f"conn_{connection_id}",
                        source_block=source_block,
                        target_block=target_block,
                        path_segments=segment_group,
                        has_explicit_direction=False,
                        connection_type="indirect"
                    ))
        
        return indirect_connections
    
    def _group_aligned_segments(self, lines: List[LineEntity]) -> List[List[LineEntity]]:
        """
        将看起来对齐或有小间隙连接的线段分组
        
        Args:
            lines: 线段列表
            
        Returns:
            List[List[LineEntity]]: 分组后的线段列表
        """
        if not lines:
            return []
        
        # 创建潜在连接的线段图
        segment_graph = nx.Graph()
        
        for i, line1 in enumerate(lines):
            segment_graph.add_node(i, line=line1)
            
            for j, line2 in enumerate(lines):
                if i != j:
                    # 检查line2是否可能与line1连接
                    if self._are_segments_connected(line1, line2):
                        segment_graph.add_edge(i, j)
        
        # 查找连通分量（连接线段组）
        groups = []
        for component in nx.connected_components(segment_graph):
            group = [segment_graph.nodes[i]['line'] for i in component]
            groups.append(group)
        
        return groups
    
    def _are_segments_connected(self, line1: LineEntity, line2: LineEntity) -> bool:
        """
        检查两条线段是否可能连接（对齐、小间隙）
        
        Args:
            line1: 第一条线段
            line2: 第二条线段
            
        Returns:
            bool: 是否可能连接
        """
        # 检查端点间距离
        distances = [
            (line1.start_point.distance_to(line2.start_point), (line1.start_point, line2.start_point)),
            (line1.start_point.distance_to(line2.end_point), (line1.start_point, line2.end_point)),
            (line1.end_point.distance_to(line2.start_point), (line1.end_point, line2.start_point)),
            (line1.end_point.distance_to(line2.end_point), (line1.end_point, line2.end_point))
        ]
        
        # 找到最近的端点
        closest = min(distances, key=lambda x: x[0])
        
        # 检查它们是否足够近
        if closest[0] > self.max_gap_distance:
            return False
        
        # 检查它们是否大致对齐（方向相似）
        dir1 = line1.get_direction()
        dir2 = line2.get_direction()
        
        # 方向相似度的点积
        dot_product = (dir1[0] * dir2[0] + dir1[1] * dir2[1] + dir1[2] * dir2[2])
        
        # 如果点积接近1或-1，则它们大致对齐
        return abs(abs(dot_product) - 1.0) <= self.connection_angle_tolerance
    
    def _determine_connection_directions(self, connections: List[Connection], blocks: List[Block]):
        """
        根据箭头块确定连接方向
        
        Args:
            connections: 连接列表
            blocks: 块列表
        """
        # 识别所有箭头块
        arrow_blocks = []
        for block in blocks:
            if self.block_identifier:
                if self.block_identifier.is_arrow_block(block):
                    arrow_blocks.append(block)
            elif block.is_arrow:
                arrow_blocks.append(block)
        
        for connection in connections:
            # 检查此连接上是否有箭头块
            for arrow in arrow_blocks:
                # 检查箭头是否在连接的任何线段上
                for segment in connection.path_segments:
                    if self._is_arrow_on_segment(arrow, segment):
                        # 根据箭头方向确定方向
                        arrow_dir = self._get_arrow_direction(arrow)
                        segment_dir = segment.get_direction()
                        
                        # 点积，查看它们是否指向相同方向
                        dot_product = (arrow_dir[0] * segment_dir[0] + 
                                     arrow_dir[1] * segment_dir[1] + 
                                     arrow_dir[2] * segment_dir[2])
                        
                        # 如果点积为正，它们指向相同方向
                        # 确保连接的源/目标与箭头方向匹配
                        if dot_product < 0:
                            # 如果不匹配，交换源和目标
                            connection.source_block, connection.target_block = connection.target_block, connection.source_block
                        
                        connection.has_explicit_direction = True
                        break
                
                if connection.has_explicit_direction:
                    break
        
        # 对于没有明确方向的连接，从已连接的有方向连接推断
        self._infer_connection_directions(connections)
    
    def _is_arrow_on_segment(self, arrow_block: Block, segment: LineEntity) -> bool:
        """
        检查箭头块是否在线段上
        
        Args:
            arrow_block: 箭头块
            segment: 线段
            
        Returns:
            bool: 箭头是否在线段上
        """
        if not arrow_block.bounding_box:
            return False
        
        # 检查箭头边界框是否与线相交
        # 首先，为线创建一个有厚度的边界框
        thickness = 1.0  # 根据需要调整
        
        line_min_x = min(segment.start_point.x, segment.end_point.x) - thickness
        line_min_y = min(segment.start_point.y, segment.end_point.y) - thickness
        line_min_z = min(segment.start_point.z, segment.end_point.z) - thickness
        
        line_max_x = max(segment.start_point.x, segment.end_point.x) + thickness
        line_max_y = max(segment.start_point.y, segment.end_point.y) + thickness
        line_max_z = max(segment.start_point.z, segment.end_point.z) + thickness
        
        line_bbox = BoundingBox(
            Point(line_min_x, line_min_y, line_min_z),
            Point(line_max_x, line_max_y, line_max_z)
        )
        
        # 检查箭头的边界框是否与线的边界框重叠
        if not arrow_block.bounding_box.overlaps(line_bbox):
            return False
        
        # 为了更精确，检查箭头的中心是否靠近线
        # 计算点到线的距离
        if not arrow_block.center:
            return False
        
        return self._point_line_distance(arrow_block.center, segment.start_point, segment.end_point) < 5.0
    
    def _point_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """
        计算点到线段的距离
        
        Args:
            point: 点
            line_start: 线段起点
            line_end: 线段终点
            
        Returns:
            float: 点到线段的距离
        """
        # 从线段起点到终点的向量
        line_vec = (
            line_end.x - line_start.x,
            line_end.y - line_start.y,
            line_end.z - line_start.z
        )
        
        # 从线段起点到点的向量
        point_vec = (
            point.x - line_start.x,
            point.y - line_start.y,
            point.z - line_start.z
        )
        
        # 线长度的平方
        line_length_sq = (line_vec[0]**2 + line_vec[1]**2 + line_vec[2]**2)
        
        # 边界情况：零长度线
        if line_length_sq == 0:
            return math.sqrt(point_vec[0]**2 + point_vec[1]**2 + point_vec[2]**2)
        
        # 计算投影因子
        t = max(0, min(1, (point_vec[0] * line_vec[0] + 
                        point_vec[1] * line_vec[1] + 
                        point_vec[2] * line_vec[2]) / line_length_sq))
        
        # 计算线上最近点
        closest_x = line_start.x + t * line_vec[0]
        closest_y = line_start.y + t * line_vec[1]
        closest_z = line_start.z + t * line_vec[2]
        
        # 计算距离
        return math.sqrt((point.x - closest_x)**2 + 
                       (point.y - closest_y)**2 + 
                       (point.z - closest_z)**2)
    
    def _get_arrow_direction(self, arrow_block: Block) -> Tuple[float, float, float]:
        """
        确定箭头块的指向方向
        
        Args:
            arrow_block: 箭头块
            
        Returns:
            Tuple[float, float, float]: 箭头方向向量
        """
        if not arrow_block.bounding_box:
            return (0, 0, 0)
        
        # 简化方法：使用箭头的边界框长宽比确定主要方向
        aspect = arrow_block.bounding_box.aspect_ratio
        
        if aspect > 1.0:  # 宽大于高
            # 根据实体分布确定是指向左还是右
            left_count = 0
            right_count = 0
            
            midpoint_x = (arrow_block.bounding_box.min_point.x + 
                        arrow_block.bounding_box.max_point.x) / 2
            
            for entity in arrow_block.entities:
                if isinstance(entity, LineEntity):
                    # 检查起点
                    if entity.start_point.x < midpoint_x:
                        left_count += 1
                    else:
                        right_count += 1
                    
                    # 检查终点
                    if entity.end_point.x < midpoint_x:
                        left_count += 1
                    else:
                        right_count += 1
            
            # 如果右侧端点更多，可能指向右侧
            if right_count > left_count:
                return (1.0, 0.0, 0.0)  # 右
            else:
                return (-1.0, 0.0, 0.0)  # 左
        else:  # 高大于宽
            # 确定是指向上还是下
            top_count = 0
            bottom_count = 0
            
            midpoint_y = (arrow_block.bounding_box.min_point.y + 
                        arrow_block.bounding_box.max_point.y) / 2
            
            for entity in arrow_block.entities:
                if isinstance(entity, LineEntity):
                    # 检查起点
                    if entity.start_point.y < midpoint_y:
                        bottom_count += 1
                    else:
                        top_count += 1
                    
                    # 检查终点
                    if entity.end_point.y < midpoint_y:
                        bottom_count += 1
                    else:
                        top_count += 1
            
            # 如果顶部端点更多，可能指向上
            if top_count > bottom_count:
                return (0.0, 1.0, 0.0)  # 上
            else:
                return (0.0, -1.0, 0.0)  # 下
    
    def _infer_connection_directions(self, connections: List[Connection]):
        """
        为没有明确方向的连接推断方向
        
        Args:
            connections: 连接列表
        """
        # 从有明确方向的连接构建有向图
        G = nx.DiGraph()
        
        # 添加所有块作为节点
        all_blocks = set()
        for conn in connections:
            all_blocks.add(conn.source_block.id)
            all_blocks.add(conn.target_block.id)
        
        for block_id in all_blocks:
            G.add_node(block_id)
        
        # 添加有明确方向的连接作为边
        for conn in connections:
            if conn.has_explicit_direction:
                G.add_edge(conn.source_block.id, conn.target_block.id, connection=conn)
        
        # 对每个无方向连接，尝试推断方向
        for conn in connections:
            if not conn.has_explicit_direction:
                source_id = conn.source_block.id
                target_id = conn.target_block.id
                
                # 检查是否有从源到目标或从目标到源的路径
                try:
                    if nx.has_path(G, source_id, target_id):
                        # 保持当前方向（源到目标）
                        conn.has_explicit_direction = True
                    elif nx.has_path(G, target_id, source_id):
                        # 反转方向（目标到源）
                        conn.source_block, conn.target_block = conn.target_block, conn.source_block
                        conn.has_explicit_direction = True
                except:
                    # 处理可能的NetworkX异常
                    pass


class ConnectionClassifier:
    """连接分类器"""
    
    def classify_connections(self, connections: List[Connection]) -> Dict[str, List[Connection]]:
        """
        对连接进行分类
        
        Args:
            connections: 连接列表
            
        Returns:
            Dict[str, List[Connection]]: 分类结果，键为分类名称，值为连接列表
        """
        classifications = {
            'direct': [],     # 直接连接
            'indirect': [],   # 间接连接
            'directed': [],   # 有方向的连接
            'undirected': [], # 无方向的连接
            'simple': [],     # 简单连接（单线段）
            'complex': [],    # 复杂连接（多线段）
        }
        
        for conn in connections:
            # 按连接类型分类
            if conn.connection_type == "indirect":
                classifications['indirect'].append(conn)
            else:
                classifications['direct'].append(conn)
            
            # 按方向性分类
            if conn.has_explicit_direction:
                classifications['directed'].append(conn)
            else:
                classifications['undirected'].append(conn)
            
            # 按复杂性分类
            if len(conn.path_segments) <= 1:
                classifications['simple'].append(conn)
            else:
                classifications['complex'].append(conn)
        
        return classifications
    
    def get_connection_stats(self, connections: List[Connection]) -> Dict:
        """
        获取连接统计信息
        
        Args:
            connections: 连接列表
            
        Returns:
            Dict: 统计信息
        """
        if not connections:
            return {
                "count": 0,
                "message": "No connections to analyze"
            }
        
        # 按类型统计
        type_counts = {}
        for conn in connections:
            conn_type = conn.connection_type
            type_counts[conn_type] = type_counts.get(conn_type, 0) + 1
        
        # 方向统计
        directed_count = sum(1 for conn in connections if conn.has_explicit_direction)
        undirected_count = len(connections) - directed_count
        
        # 路径段统计
        segment_counts = [len(conn.path_segments) for conn in connections]
        avg_segments = sum(segment_counts) / len(segment_counts) if segment_counts else 0
        max_segments = max(segment_counts) if segment_counts else 0
        
        # 计算连接长度
        lengths = []
        for conn in connections:
            total_length = sum(segment.get_length() for segment in conn.path_segments)
            lengths.append(total_length)
        
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0
        
        return {
            "count": len(connections),
            "type_counts": type_counts,
            "directed_count": directed_count,
            "undirected_count": undirected_count,
            "avg_segments": avg_segments,
            "max_segments": max_segments,
            "avg_length": avg_length,
            "max_length": max_length,
            "min_length": min_length
        }


class PathFinder:
    """路径查找器"""
    
    def __init__(self):
        """初始化路径查找器"""
        self.graph = nx.DiGraph()
    
    def build_graph(self, connections: List[Connection]):
        """
        构建连接图
        
        Args:
            connections: 连接列表
        """
        self.graph = nx.DiGraph()
        
        # 添加所有块作为节点
        blocks = set()
        for conn in connections:
            blocks.add(conn.source_block.id)
            blocks.add(conn.target_block.id)
        
        for block_id in blocks:
            self.graph.add_node(block_id)
        
        # 添加连接作为边
        for conn in connections:
            # 如果连接有明确方向，添加有向边
            if conn.has_explicit_direction:
                self.graph.add_edge(
                    conn.source_block.id,
                    conn.target_block.id,
                    weight=1.0,
                    connection=conn
                )
            else:
                # 否则添加双向边
                self.graph.add_edge(
                    conn.source_block.id,
                    conn.target_block.id,
                    weight=1.0,
                    connection=conn
                )
                self.graph.add_edge(
                    conn.target_block.id,
                    conn.source_block.id,
                    weight=1.0,
                    connection=conn
                )
    
    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """
        查找从源到目标的最短路径
        
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
        except nx.NetworkXError:
            return []
    
    def find_all_paths(self, source_id: str, target_id: str, cutoff: int = None) -> List[List[str]]:
        """
        查找从源到目标的所有简单路径
        
        Args:
            source_id: 源块ID
            target_id: 目标块ID
            cutoff: 最大路径长度
            
        Returns:
            List[List[str]]: 所有路径的列表
        """
        try:
            return list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=cutoff))
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return []
    
    def find_cycles(self) -> List[List[str]]:
        """
        查找图中的所有环路
        
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
    
    def analyze_connectivity(self) -> Dict:
        """
        分析图的连通性
        
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
                result["largest_strongly_connected_size"] = max(len(c) for c in strongly_connected)
            
            # 分析弱连通分量
            weakly_connected = list(nx.weakly_connected_components(self.graph))
            result["weakly_connected_components"] = len(weakly_connected)
            
            if weakly_connected:
                result["largest_weakly_connected_size"] = max(len(c) for c in weakly_connected)
        else:
            # 分析连通分量
            connected = list(nx.connected_components(self.graph))
            result["connected_components"] = len(connected)
            
            if connected:
                result["largest_connected_size"] = max(len(c) for c in connected)
        
        # 计算平均路径长度（如果图是连通的）
        try:
            result["average_shortest_path_length"] = nx.average_shortest_path_length(self.graph)
        except:
            # 图可能不是连通的
            result["average_shortest_path_length"] = None
        
        # 计算图密度
        result["density"] = nx.density(self.graph)
        
        return result
    
    def find_critical_nodes(self) -> List[str]:
        """
        查找图中的关键节点（删除后会增加连通分量数量的节点）
        
        Returns:
            List[str]: 关键节点ID列表
        """
        try:
            return list(nx.articulation_points(self.graph.to_undirected()))
        except:
            # 简单版实现
            critical_nodes = []
            original_components = nx.number_connected_components(self.graph.to_undirected())
            
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