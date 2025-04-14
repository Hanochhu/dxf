"""
CAD分析系统主模块
整合各个模块功能，提供统一的接口
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid

from src.core.data_structures import Entity, Block, Connection, BlockFeature, LineEntity
from src.parsers.parser_interface import ParserRegistry, CADFileParser
from src.parsers.dxf_parser import DXFParser
from src.parsers.step_parser import STEPParser
from src.feature.block_identifier import BlockIdentifier, BlockFeatureExtractor
from src.connection.connection_analyzer import ConnectionAnalyzer
from src.graph.cad_graph import CADGraph
from src.query.cad_query import CADQueryInterface, CADQueryResults


class CADAnalysisSystem:
    """CAD分析系统主类"""

    def __init__(self):
        """初始化CAD分析系统"""
        # 初始化解析器注册中心
        self.parser_registry = ParserRegistry()

        # 注册所有可用解析器
        self._register_parsers()

        # 初始化核心组件
        self.block_identifier = BlockIdentifier()
        self.connection_analyzer = ConnectionAnalyzer(self.block_identifier)
        self.cad_graph = CADGraph()
        self.query_interface = CADQueryInterface(self.cad_graph, self.block_identifier)

        # 分析状态
        self.current_file = None
        self.entities = []
        self.blocks = []
        self.connections = []
        self.additional_info = {}

    def _register_parsers(self):
        """注册所有可用的解析器"""
        # 注册DXF解析器
        self.parser_registry.register_parser(DXFParser())

        # 注册STEP解析器
        self.parser_registry.register_parser(STEPParser())

    def analyze_file(self, file_path: str) -> bool:
        """
        分析CAD文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 分析是否成功
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        try:
            # 获取适用的解析器
            parser = self.parser_registry.get_parser(file_path)

            # 解析文件
            # parse_file 返回四元组: (entities, block_definitions, block_references, additional_info)
            self.entities, self.block_definitions, self.block_references, self.additional_info = parser.parse_file(
                file_path
            )

            # 提取线段用于连接分析
            lines = [
                entity for entity in self.entities if isinstance(entity, LineEntity)
            ]

            # 查找块之间的连接
            self.connections = self.connection_analyzer.find_connections(
                self.block_definitions, lines
            )

            # 构建图
            self.cad_graph.build_from_blocks_connections(self.block_definitions, self.connections)

            # 更新当前文件路径
            self.current_file = file_path

            return True

        except Exception as e:
            print(f"分析文件时出错: {e}")
            return False

    def load_block_templates(self, template_file: str) -> bool:
        """
        从文件加载块模板

        Args:
            template_file: 模板文件路径

        Returns:
            bool: 加载是否成功
        """
        return self.block_identifier.load_templates(template_file)

    def save_block_templates(self, template_file: str) -> bool:
        """
        保存块模板到文件

        Args:
            template_file: 模板文件路径

        Returns:
            bool: 保存是否成功
        """
        return self.block_identifier.save_templates(template_file)

    def add_block_template(self, name: str, template_file: str) -> bool:
        """
        从模板文件添加块模板

        Args:
            name: 模板名称
            template_file: 模板文件路径

        Returns:
            bool: 添加是否成功
        """
        try:
            if not os.path.exists(template_file):
                print(f"模板文件不存在: {template_file}")
                return False

            # 获取适用的解析器
            parser = self.parser_registry.get_parser(template_file)

            # 解析模板文件
            # parse_file 返回四元组: (entities, block_definitions, block_references, additional_info)
            _, block_definitions, _, _ = parser.parse_file(template_file)

            if not block_definitions:
                print("模板文件中没有找到块")
                return False

            # 使用第一个块作为模板
            self.block_identifier.add_block_template(name, block_definitions[0])
            return True

        except Exception as e:
            print(f"添加块模板时出错: {e}")
            return False

    def add_block_feature(self, feature: BlockFeature) -> bool:
        """
        添加块特征

        Args:
            feature: 块特征对象

        Returns:
            bool: 添加是否成功
        """
        try:
            self.block_identifier.add_block_feature(feature)
            return True
        except Exception as e:
            print(f"添加块特征时出错: {e}")
            return False

    def create_block_feature_from_block(
        self, block: Block, name: str, description: str = "", tolerance: float = 0.2
    ) -> Optional[BlockFeature]:
        """
        从块创建特征

        Args:
            block: 块对象
            name: 特征名称
            description: 特征描述
            tolerance: 特征容差

        Returns:
            Optional[BlockFeature]: 创建的特征对象，如果失败则返回None
        """
        try:
            feature = BlockFeature.from_sample_block(
                block, name, description, tolerance
            )
            return feature
        except Exception as e:
            print(f"从块创建特征时出错: {e}")
            return None

    def save_analysis(self, output_file: str) -> bool:
        """
        保存分析结果

        Args:
            output_file: 输出文件路径

        Returns:
            bool: 保存是否成功
        """
        return self.cad_graph.export_to_json(output_file)

    def load_analysis(self, input_file: str) -> bool:
        """
        加载分析结果

        Args:
            input_file: 输入文件路径

        Returns:
            bool: 加载是否成功
        """
        return self.cad_graph.import_from_json(input_file)

    def query(self, query_str: str) -> Dict:
        """
        执行查询

        Args:
            query_str: 查询字符串

        Returns:
            Dict: 查询结果
        """
        result = self.query_interface.query(query_str)
        return CADQueryResults.format_query_results(result)

    def get_blocks_by_type(self, block_type: str) -> List[Block]:
        """
        获取特定类型的所有块

        Args:
            block_type: 块类型

        Returns:
            List[Block]: 块列表
        """
        return self.query_interface.get_blocks_by_type(block_type)

    def has_connection(self, source_id: str, target_id: str) -> bool:
        """
        检查两个块之间是否有连接

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            bool: 是否有连接
        """
        return self.query_interface.has_out_connection_to(source_id, target_id)

    def get_in_degree(self, block_id: str) -> int:
        """
        获取块的入度

        Args:
            block_id: 块ID

        Returns:
            int: 入度
        """
        return self.query_interface.get_in_degree(block_id)

    def get_out_degree(self, block_id: str) -> int:
        """
        获取块的出度

        Args:
            block_id: 块ID

        Returns:
            int: 出度
        """
        return self.query_interface.get_out_degree(block_id)

    def check_block_has_path(self, source_id: str, target_id: str) -> bool:
        """
        检查是否存在从源块到目标块的路径

        Args:
            source_id: 源块ID
            target_id: 目标块ID

        Returns:
            bool: 是否存在路径
        """
        path = self.query_interface.find_path(source_id, target_id)
        return len(path) > 0

    def get_block_connections(self, block_id: str, direction: str = "both") -> Dict:
        """
        获取块的连接信息

        Args:
            block_id: 块ID
            direction: 连接方向，可选 "in", "out", "both"

        Returns:
            Dict: 连接信息
        """
        block = self.query_interface.get_block_by_id(block_id)
        if not block:
            return {"error": f"找不到块: {block_id}"}

        result = {
            "block_id": block_id,
            "block_name": block.name,
            "in_degree": self.get_in_degree(block_id),
            "out_degree": self.get_out_degree(block_id),
        }

        if direction in ["in", "both"]:
            in_connections = self.query_interface.get_in_connections(block_id)
            result["in_connections"] = [
                {
                    "from_block_id": conn.source_block.id,
                    "from_block_name": conn.source_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction,
                }
                for conn in in_connections
            ]

        if direction in ["out", "both"]:
            out_connections = self.query_interface.get_out_connections(block_id)
            result["out_connections"] = [
                {
                    "to_block_id": conn.target_block.id,
                    "to_block_name": conn.target_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction,
                }
                for conn in out_connections
            ]

        return result

    def set_connection_parameters(
        self,
        max_gap_distance: float = None,
        connection_angle_tolerance: float = None,
        block_connection_tolerance: float = None,
    ):
        """
        设置连接分析参数

        Args:
            max_gap_distance: 最大间隙距离
            connection_angle_tolerance: 连接角度容差
            block_connection_tolerance: 块连接容差
        """
        self.connection_analyzer.set_parameters(
            max_gap_distance, connection_angle_tolerance, block_connection_tolerance
        )

    def get_statistics(self) -> Dict:
        """
        获取分析统计信息

        Returns:
            Dict: 统计信息
        """
        # 获取图统计信息
        graph_stats = self.cad_graph.get_statistics()

        # 添加块和实体统计
        stats = {
            "file_path": self.current_file,
            "entity_count": len(self.entities),
            "block_count": len(self.blocks),
            "connection_count": len(self.connections),
            "arrow_count": sum(1 for block in self.blocks if block.is_arrow),
            "line_count": sum(
                1 for entity in self.entities if isinstance(entity, LineEntity)
            ),
            "graph_stats": graph_stats,
        }

        # 添加块类型分布
        if self.block_identifier:
            block_types = {}
            for block in self.blocks:
                matches = self.block_identifier.identify_block(block)
                if matches:
                    best_match = matches[0]
                    block_type = best_match[0]
                    block_types[block_type] = block_types.get(block_type, 0) + 1

            stats["block_type_distribution"] = block_types

        return stats

    def find_central_blocks(self, top_n: int = 5, method: str = "degree") -> List[Dict]:
        """
        查找图中最中心的块

        Args:
            top_n: 返回的块数量
            method: 中心性度量方法，可选 'degree', 'betweenness', 'closeness', 'eigenvector'

        Returns:
            List[Dict]: 中心块信息列表
        """
        central_blocks = self.cad_graph.get_central_blocks(top_n, method)

        result = []
        for block_id, centrality in central_blocks:
            block = self.cad_graph.get_block(block_id)
            if block:
                result.append(
                    {
                        "block_id": block_id,
                        "block_name": block.name,
                        "centrality": centrality,
                        "in_degree": self.get_in_degree(block_id),
                        "out_degree": self.get_out_degree(block_id),
                    }
                )

        return result

    def find_communities(self, method: str = "louvain") -> Dict[str, List[str]]:
        """
        查找图中的社区（社团）

        Args:
            method: 社区检测方法，可选 'louvain', 'label_propagation', 'greedy_modularity'

        Returns:
            Dict[str, List[str]]: 社区到块ID列表的映射
        """
        community_map = self.cad_graph.find_communities(method)

        # 按社区ID组织结果
        communities = {}
        for block_id, community_id in community_map.items():
            comm_key = str(community_id)
            if comm_key not in communities:
                communities[comm_key] = []
            communities[comm_key].append(block_id)

        return communities

    def export_graphviz(self, file_path: str, format: str = "dot") -> bool:
        """
        导出图的可视化表示

        Args:
            file_path: 输出文件路径
            format: 输出格式，可选 'dot', 'gexf', 'gml'

        Returns:
            bool: 导出是否成功
        """
        from graph.cad_graph import GraphVisualizer

        visualizer = GraphVisualizer(self.cad_graph)
        return visualizer.export_graphviz(file_path, format)


class CADAnalysisConfig:
    """CAD分析系统配置类"""

    def __init__(self):
        """初始化配置"""
        self.connection_params = {
            "max_gap_distance": 10.0,
            "connection_angle_tolerance": 0.2,
            "block_connection_tolerance": 2.0,
        }

        self.feature_params = {"default_tolerance": 0.2}

        self.analysis_params = {
            "auto_identify_arrows": True,
            "merge_indirect_connections": True,
        }

    def load_from_file(self, config_file: str) -> bool:
        """
        从文件加载配置

        Args:
            config_file: 配置文件路径

        Returns:
            bool: 加载是否成功
        """
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 更新连接参数
            if "connection_params" in config:
                self.connection_params.update(config["connection_params"])

            # 更新特征参数
            if "feature_params" in config:
                self.feature_params.update(config["feature_params"])

            # 更新分析参数
            if "analysis_params" in config:
                self.analysis_params.update(config["analysis_params"])

            return True

        except Exception as e:
            print(f"加载配置时出错: {e}")
            return False

    def save_to_file(self, config_file: str) -> bool:
        """
        保存配置到文件

        Args:
            config_file: 配置文件路径

        Returns:
            bool: 保存是否成功
        """
        try:
            config = {
                "connection_params": self.connection_params,
                "feature_params": self.feature_params,
                "analysis_params": self.analysis_params,
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            return True

        except Exception as e:
            print(f"保存配置时出错: {e}")
            return False

    def apply_to_system(self, system: CADAnalysisSystem):
        """
        将配置应用到系统

        Args:
            system: CAD分析系统
        """
        # 应用连接参数
        system.set_connection_parameters(
            max_gap_distance=self.connection_params.get("max_gap_distance"),
            connection_angle_tolerance=self.connection_params.get(
                "connection_angle_tolerance"
            ),
            block_connection_tolerance=self.connection_params.get(
                "block_connection_tolerance"
            ),
        )

        # 应用其他参数（如果需要）
