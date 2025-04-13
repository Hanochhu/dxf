"""
查询接口模块
提供高级查询功能
"""

import re
from typing import List, Dict, Tuple, Optional, Set, Any, Union

from core.data_structures import Block, Connection
from graph.cad_graph import CADGraph
from feature.block_identifier import BlockIdentifier


class CADQueryInterface:
    """CAD图形查询接口"""

    def __init__(self, cad_graph: CADGraph, block_identifier: BlockIdentifier = None):
        """
        初始化查询接口

        Args:
            cad_graph: CAD图对象
            block_identifier: 块识别器（可选）
        """
        self.graph = cad_graph
        self.block_identifier = block_identifier or BlockIdentifier()

    def get_block_by_id(self, block_id: str) -> Optional[Block]:
        """
        根据ID获取块

        Args:
            block_id: 块ID

        Returns:
            Optional[Block]: 块对象，如果不存在则返回None
        """
        return self.graph.get_block(block_id)

    def get_blocks_by_name(self, name: str, exact_match: bool = False) -> List[Block]:
        """
        获取指定名称的所有块

        Args:
            name: 块名称
            exact_match: 是否精确匹配名称

        Returns:
            List[Block]: 块列表
        """
        if exact_match:
            return [
                block
                for block_id, block in self.graph.blocks.items()
                if block.name == name
            ]
        else:
            return [
                block
                for block_id, block in self.graph.blocks.items()
                if name.lower() in block.name.lower()
            ]

    def get_blocks_by_type(
        self, block_type: str, threshold: float = 0.7
    ) -> List[Block]:
        """
        获取指定类型的所有块（基于特征匹配）

        Args:
            block_type: 块类型名称
            threshold: 匹配阈值

        Returns:
            List[Block]: 匹配的块列表
        """
        if not self.block_identifier:
            return []

        matches = []

        for block_id, block in self.graph.blocks.items():
            block_matches = self.block_identifier.identify_block(block)
            for match_type, similarity in block_matches:
                if match_type == block_type and similarity >= threshold:
                    matches.append(block)
                    break

        return matches

    def has_in_connection_from(self, target_id: str, source_id: str) -> bool:
        """
        检查目标块是否有来自源块的入连接

        Args:
            target_id: 目标块ID
            source_id: 源块ID

        Returns:
            bool: 是否有连接
        """
        return self.graph.has_connection(source_id, target_id)

    def has_out_connection_to(self, source_id: str, target_id: str) -> bool:
        """
        检查源块是否有指向目标块的出连接

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            bool: 是否有连接
        """
        return self.graph.has_connection(source_id, target_id)

    def get_in_connections(self, block_id: str) -> List[Connection]:
        """
        获取块的所有入连接

        Args:
            block_id: 块ID

        Returns:
            List[Connection]: 连接列表
        """
        connections = []

        if block_id not in self.graph.graph:
            return []

        for pred_id in self.graph.graph.predecessors(block_id):
            edge_data = self.graph.graph.get_edge_data(pred_id, block_id)
            if edge_data and "connection_id" in edge_data:
                conn_id = edge_data["connection_id"]
                if conn_id in self.graph.connections:
                    connections.append(self.graph.connections[conn_id])

        return connections

    def get_out_connections(self, block_id: str) -> List[Connection]:
        """
        获取块的所有出连接

        Args:
            block_id: 块ID

        Returns:
            List[Connection]: 连接列表
        """
        connections = []

        if block_id not in self.graph.graph:
            return []

        for succ_id in self.graph.graph.successors(block_id):
            edge_data = self.graph.graph.get_edge_data(block_id, succ_id)
            if edge_data and "connection_id" in edge_data:
                conn_id = edge_data["connection_id"]
                if conn_id in self.graph.connections:
                    connections.append(self.graph.connections[conn_id])

        return connections

    def get_connected_blocks_of_type(
        self,
        block_id: str,
        block_type: str,
        direction: str = "both",
        threshold: float = 0.7,
    ) -> List[Block]:
        """
        获取与指定块连接的指定类型的所有块

        Args:
            block_id: 块ID
            block_type: 块类型
            direction: 连接方向，可选 "in", "out", "both"
            threshold: 匹配阈值

        Returns:
            List[Block]: 块列表
        """
        if block_id not in self.graph.graph:
            return []

        connected_blocks = []

        # 检查入连接
        if direction in ["in", "both"]:
            for pred_id in self.graph.graph.predecessors(block_id):
                if pred_id in self.graph.blocks:
                    pred_block = self.graph.blocks[pred_id]
                    matches = self.block_identifier.identify_block(pred_block)
                    for match_type, similarity in matches:
                        if match_type == block_type and similarity >= threshold:
                            connected_blocks.append(pred_block)
                            break

        # 检查出连接
        if direction in ["out", "both"]:
            for succ_id in self.graph.graph.successors(block_id):
                if succ_id in self.graph.blocks:
                    succ_block = self.graph.blocks[succ_id]
                    matches = self.block_identifier.identify_block(succ_block)
                    for match_type, similarity in matches:
                        if match_type == block_type and similarity >= threshold:
                            connected_blocks.append(succ_block)
                            break

        return connected_blocks

    def get_in_degree(self, block_id: str) -> int:
        """
        获取入度（入连接数）

        Args:
            block_id: 块ID

        Returns:
            int: 入度
        """
        return self.graph.get_in_degree(block_id)

    def get_out_degree(self, block_id: str) -> int:
        """
        获取出度（出连接数）

        Args:
            block_id: 块ID

        Returns:
            int: 出度
        """
        return self.graph.get_out_degree(block_id)

    def find_path(self, source_id: str, target_id: str) -> List[Block]:
        """
        查找从源块到目标块的路径

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            List[Block]: 路径中的块列表
        """
        path_ids = self.graph.get_path(source_id, target_id)
        return [
            self.graph.blocks[block_id]
            for block_id in path_ids
            if block_id in self.graph.blocks
        ]

    def find_all_paths(
        self, source_id: str, target_id: str, cutoff: int = None
    ) -> List[List[Block]]:
        """
        查找从源到目标的所有路径

        Args:
            source_id: 源块ID
            target_id: 目标块ID
            cutoff: 最大路径长度

        Returns:
            List[List[Block]]: 所有路径的列表
        """
        path_lists = self.graph.get_all_paths(source_id, target_id, cutoff)

        all_paths = []
        for path in path_lists:
            block_path = [
                self.graph.blocks[block_id]
                for block_id in path
                if block_id in self.graph.blocks
            ]
            all_paths.append(block_path)

        return all_paths

    def find_blocks_by_criteria(self, criteria: Dict) -> List[Block]:
        """
        根据条件查找块

        Args:
            criteria: 条件字典，例如 {'entity_count': ('>',5), 'is_arrow': True}

        Returns:
            List[Block]: 匹配的块列表
        """
        matches = []

        for block_id, block in self.graph.blocks.items():
            # 检查每个条件
            match = True

            for key, condition in criteria.items():
                # 获取块的属性值
                if key == "entity_count":
                    value = len(block.entities)
                elif key == "is_arrow":
                    value = block.is_arrow
                elif key == "name":
                    value = block.name
                elif key == "aspect_ratio" and block.bounding_box:
                    value = block.bounding_box.aspect_ratio
                elif key == "width" and block.bounding_box:
                    value = block.bounding_box.width
                elif key == "height" and block.bounding_box:
                    value = block.bounding_box.height
                elif key == "in_degree":
                    value = self.get_in_degree(block_id)
                elif key == "out_degree":
                    value = self.get_out_degree(block_id)
                else:
                    # 属性不存在或无法获取
                    match = False
                    break

                # 检查条件
                if isinstance(condition, tuple) and len(condition) == 2:
                    op, target = condition

                    if op == ">" and not (value > target):
                        match = False
                        break
                    elif op == ">=" and not (value >= target):
                        match = False
                        break
                    elif op == "<" and not (value < target):
                        match = False
                        break
                    elif op == "<=" and not (value <= target):
                        match = False
                        break
                    elif op == "==" and not (value == target):
                        match = False
                        break
                    elif op == "!=" and not (value != target):
                        match = False
                        break
                    elif op == "contains" and not (
                        target in value if isinstance(value, str) else False
                    ):
                        match = False
                        break
                    elif op == "startswith" and not (
                        value.startswith(target) if isinstance(value, str) else False
                    ):
                        match = False
                        break
                    elif op == "endswith" and not (
                        value.endswith(target) if isinstance(value, str) else False
                    ):
                        match = False
                        break
                else:
                    # 直接比较相等
                    if value != condition:
                        match = False
                        break

            if match:
                matches.append(block)

        return matches

    def query(self, query_str: str) -> Dict:
        """
        执行自然语言查询

        Args:
            query_str: 查询字符串

        Returns:
            Dict: 包含查询结果的字典
        """
        query_str = query_str.strip().lower()
        result = {"success": False, "message": "", "data": None}

        try:
            # 入度查询
            if re.search(
                r"(indegree|in degree|incoming|in connections) .* (block|node) ([a-zA-Z0-9_]+)",
                query_str,
            ):
                match = re.search(
                    r"(indegree|in degree|incoming|in connections) .* (block|node) ([a-zA-Z0-9_]+)",
                    query_str,
                )
                block_id = match.group(3)

                in_degree = self.get_in_degree(block_id)
                result = {
                    "success": True,
                    "message": f"块 {block_id} 有 {in_degree} 个入连接",
                    "data": in_degree,
                }

            # 出度查询
            elif re.search(
                r"(outdegree|out degree|outgoing|out connections) .* (block|node) ([a-zA-Z0-9_]+)",
                query_str,
            ):
                match = re.search(
                    r"(outdegree|out degree|outgoing|out connections) .* (block|node) ([a-zA-Z0-9_]+)",
                    query_str,
                )
                block_id = match.group(3)

                out_degree = self.get_out_degree(block_id)
                result = {
                    "success": True,
                    "message": f"块 {block_id} 有 {out_degree} 个出连接",
                    "data": out_degree,
                }

            # 连接查询
            elif re.search(
                r"(has connection|is connected|connection exists) .* (from|between) ([a-zA-Z0-9_]+) .* (to|and) ([a-zA-Z0-9_]+)",
                query_str,
            ):
                match = re.search(
                    r"(has connection|is connected|connection exists) .* (from|between) ([a-zA-Z0-9_]+) .* (to|and) ([a-zA-Z0-9_]+)",
                    query_str,
                )

                if match.group(2) == "from":
                    source_id = match.group(3)
                    target_id = match.group(5)
                else:  # between ... and ...
                    source_id = match.group(3)
                    target_id = match.group(5)

                has_connection = self.has_out_connection_to(source_id, target_id)
                result = {
                    "success": True,
                    "message": f"从 {source_id} 到 {target_id} 的连接: {has_connection}",
                    "data": has_connection,
                }

            # 路径查询
            elif re.search(
                r"(path|route|connection) .* (from|between) ([a-zA-Z0-9_]+) .* (to|and) ([a-zA-Z0-9_]+)",
                query_str,
            ):
                match = re.search(
                    r"(path|route|connection) .* (from|between) ([a-zA-Z0-9_]+) .* (to|and) ([a-zA-Z0-9_]+)",
                    query_str,
                )

                if match.group(2) == "from":
                    source_id = match.group(3)
                    target_id = match.group(5)
                else:  # between ... and ...
                    source_id = match.group(3)
                    target_id = match.group(5)

                path = self.find_path(source_id, target_id)

                if path:
                    path_ids = [block.id for block in path]
                    result = {
                        "success": True,
                        "message": f"找到从 {source_id} 到 {target_id} 的路径: {' -> '.join(path_ids)}",
                        "data": path,
                    }
                else:
                    result = {
                        "success": True,
                        "message": f"没有找到从 {source_id} 到 {target_id} 的路径",
                        "data": [],
                    }

            # 查找特定类型的块
            elif re.search(
                r"find .* (blocks?|nodes?) .* (type|kind|category) ([a-zA-Z0-9_]+)",
                query_str,
            ):
                match = re.search(
                    r"find .* (blocks?|nodes?) .* (type|kind|category) ([a-zA-Z0-9_]+)",
                    query_str,
                )
                block_type = match.group(3)

                blocks = self.get_blocks_by_type(block_type)

                if blocks:
                    result = {
                        "success": True,
                        "message": f"找到 {len(blocks)} 个类型为 {block_type} 的块",
                        "data": blocks,
                    }
                else:
                    result = {
                        "success": True,
                        "message": f"没有找到类型为 {block_type} 的块",
                        "data": [],
                    }

            # 如果以上模式都不匹配，返回错误消息
            else:
                result = {
                    "success": False,
                    "message": "无法识别的查询语句，请尝试其他格式",
                    "data": None,
                }

        except Exception as e:
            result = {
                "success": False,
                "message": f"查询处理出错: {str(e)}",
                "data": None,
            }

        return result


class CADQueryResults:
    """CAD查询结果类，用于格式化和显示查询结果"""

    @staticmethod
    def format_block_info(block: Block) -> Dict:
        """
        格式化块信息

        Args:
            block: 块对象

        Returns:
            Dict: 格式化的块信息
        """
        info = {
            "id": block.id,
            "name": block.name,
            "entity_count": len(block.entities),
            "is_arrow": block.is_arrow,
        }

        # 添加边界框信息（如果有）
        if block.bounding_box:
            info["bounding_box"] = {
                "min": (
                    block.bounding_box.min_point.x,
                    block.bounding_box.min_point.y,
                    block.bounding_box.min_point.z,
                ),
                "max": (
                    block.bounding_box.max_point.x,
                    block.bounding_box.max_point.y,
                    block.bounding_box.max_point.z,
                ),
                "width": block.bounding_box.width,
                "height": block.bounding_box.height,
                "aspect_ratio": block.bounding_box.aspect_ratio,
            }

            info["center"] = (
                block.bounding_box.center.x,
                block.bounding_box.center.y,
                block.bounding_box.center.z,
            )

        # 添加引用信息（如果有）
        if block.reference:
            info["reference"] = {
                "name": block.reference.name,
                "position": (
                    block.reference.position.x,
                    block.reference.position.y,
                    block.reference.position.z,
                ),
                "rotation": block.reference.rotation,
                "scale": block.reference.scale,
            }

            # 添加属性信息（如果有）
            if block.reference.attributes:
                info["attributes"] = [
                    {"tag": attr.tag, "value": attr.value}
                    for attr in block.reference.attributes
                ]

        return info

    @staticmethod
    def format_connection_info(connection: Connection) -> Dict:
        """
        格式化连接信息

        Args:
            connection: 连接对象

        Returns:
            Dict: 格式化的连接信息
        """
        info = {
            "id": connection.id,
            "source_block": connection.source_block.id,
            "target_block": connection.target_block.id,
            "has_explicit_direction": connection.has_explicit_direction,
            "connection_type": connection.connection_type,
            "segment_count": len(connection.path_segments),
        }

        # 添加路径段信息
        info["segments"] = [
            {
                "id": segment.id,
                "start_point": (
                    segment.start_point.x,
                    segment.start_point.y,
                    segment.start_point.z,
                ),
                "end_point": (
                    segment.end_point.x,
                    segment.end_point.y,
                    segment.end_point.z,
                ),
                "length": segment.get_length(),
            }
            for segment in connection.path_segments
        ]

        return info

    @staticmethod
    def format_query_results(results: Dict) -> Dict:
        """
        格式化查询结果

        Args:
            results: 查询结果

        Returns:
            Dict: 格式化的查询结果
        """
        formatted = {
            "success": results["success"],
            "message": results["message"],
            "data": None,
        }

        data = results.get("data")

        if data is not None:
            # 格式化块列表
            if isinstance(data, list) and data and isinstance(data[0], Block):
                formatted["data"] = [
                    CADQueryResults.format_block_info(block) for block in data
                ]

            # 格式化块
            elif isinstance(data, Block):
                formatted["data"] = CADQueryResults.format_block_info(data)

            # 格式化连接列表
            elif isinstance(data, list) and data and isinstance(data[0], Connection):
                formatted["data"] = [
                    CADQueryResults.format_connection_info(conn) for conn in data
                ]

            # 格式化连接
            elif isinstance(data, Connection):
                formatted["data"] = CADQueryResults.format_connection_info(data)

            # 格式化路径
            elif (
                isinstance(data, list)
                and data
                and isinstance(data[0], list)
                and data[0]
                and isinstance(data[0][0], Block)
            ):
                formatted["data"] = [
                    [CADQueryResults.format_block_info(block) for block in path]
                    for path in data
                ]

            # 其他类型直接保留
            else:
                formatted["data"] = data

        return formatted
