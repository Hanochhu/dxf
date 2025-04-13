"""
高级使用示例
演示CAD分析系统的高级功能和定制化应用
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import networkx as nx

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data_structures import BlockFeature, EntityType, Point, BoundingBox
from feature.block_identifier import BlockClusterAnalyzer, BlockLayoutAnalyzer
from connection.connection_analyzer import ConnectionClassifier, PathFinder
from graph.cad_graph import GraphVisualizer
from system.cad_analysis_system import CADAnalysisSystem


def analyze_block_clustering(system):
    """块聚类分析示例"""
    print("\n=== 块聚类分析 ===")

    # 创建聚类分析器
    cluster_analyzer = BlockClusterAnalyzer()

    # 对块进行聚类
    blocks = list(system.cad_graph.blocks.values())
    clusters = cluster_analyzer.cluster_blocks(blocks, n_clusters=0, method="kmeans")

    print(f"找到 {len(clusters)} 个聚类")

    # 分析每个聚类
    for i, (cluster_id, cluster_blocks) in enumerate(clusters.items(), 1):
        # 只显示前3个聚类的详情
        if i <= 3:
            print(f"\n聚类 {cluster_id}:")
            print(f"  块数量: {len(cluster_blocks)}")

            # 分析块类型分布
            block_types = {}
            for block in cluster_blocks:
                matches = system.block_identifier.identify_block(block)
                block_type = matches[0][0] if matches else "unknown"
                block_types[block_type] = block_types.get(block_type, 0) + 1

            print("  块类型分布:")
            for block_type, count in block_types.items():
                print(f"    {block_type}: {count}")

    return clusters


def analyze_spatial_layout(system):
    """空间布局分析示例"""
    print("\n=== 空间布局分析 ===")

    # 创建布局分析器
    layout_analyzer = BlockLayoutAnalyzer()

    # 分析块的空间分布
    blocks = list(system.cad_graph.blocks.values())
    distribution = layout_analyzer.analyze_spatial_distribution(blocks)

    if distribution.get("empty", False):
        print("没有足够的块进行空间分析")
        return

    # 输出空间分布信息
    bounds = distribution.get("bounds", {})
    print(
        f"边界范围: X: {bounds.get('min_x', 0):.2f} - {bounds.get('max_x', 0):.2f}, "
        f"Y: {bounds.get('min_y', 0):.2f} - {bounds.get('max_y', 0):.2f}"
    )
    print(f"平均密度: {distribution.get('avg_density', 0):.5f}")
    print(f"平均距离: {distribution.get('avg_distance', 0):.2f}")

    # 分析布局模式
    patterns = layout_analyzer.detect_patterns(blocks)

    if patterns:
        pattern_types = {}
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1

        print("\n检测到的布局模式:")
        for pattern_type, count in pattern_types.items():
            print(f"  {pattern_type}: {count}")

        # 详细分析前几个模式
        for i, pattern in enumerate(patterns[:3], 1):
            pattern_type = pattern.get("type", "unknown")
            blocks_count = len(pattern.get("blocks", []))
            print(f"\n模式 {i} ({pattern_type}):")
            print(f"  块数量: {blocks_count}")

            if pattern_type == "linear":
                print(f"  角度: {pattern.get('angle', 0):.2f}")
                print(f"  等间距: {pattern.get('is_equidistant', False)}")
            elif pattern_type == "grid":
                print(f"  行数: {pattern.get('rows', 0)}")
                print(f"  列数: {pattern.get('columns', 0)}")
                print(f"  填充率: {pattern.get('fill_rate', 0):.2f}")
            elif pattern_type == "radial":
                print(f"  平均距离: {pattern.get('avg_distance', 0):.2f}")
                print(f"  平均角度差: {pattern.get('avg_angle_diff', 0):.2f}")
    else:
        print("未检测到明显的布局模式")


def analyze_connection_patterns(system):
    """连接模式分析示例"""
    print("\n=== 连接模式分析 ===")

    # 创建连接分类器
    connection_classifier = ConnectionClassifier()

    # 对连接进行分类
    connections = list(system.cad_graph.connections.values())
    classifications = connection_classifier.classify_connections(connections)

    # 输出分类统计
    print("连接分类统计:")
    print(f"  直接连接: {len(classifications['direct'])}")
    print(f"  间接连接: {len(classifications['indirect'])}")
    print(f"  有方向连接: {len(classifications['directed'])}")
    print(f"  无方向连接: {len(classifications['undirected'])}")
    print(f"  简单连接: {len(classifications['simple'])}")
    print(f"  复杂连接: {len(classifications['complex'])}")

    # 获取连接统计信息
    stats = connection_classifier.get_connection_stats(connections)
    print(f"\n连接总数: {stats['count']}")
    print(f"平均段数: {stats.get('avg_segments', 0):.2f}")
    print(f"最大段数: {stats.get('max_segments', 0)}")
    print(f"平均长度: {stats.get('avg_length', 0):.2f}")
    print(f"最大长度: {stats.get('max_length', 0):.2f}")
    print(f"最小长度: {stats.get('min_length', 0):.2f}")


def analyze_paths_and_cycles(system):
    """路径和环路分析示例"""
    print("\n=== 路径和环路分析 ===")

    # 创建路径查找器
    path_finder = PathFinder()

    # 构建图
    connections = list(system.cad_graph.connections.values())
    path_finder.build_graph(connections)

    # 分析连通性
    connectivity = path_finder.analyze_connectivity()
    print("图连通性分析:")
    print(f"  节点数量: {connectivity.get('node_count', 0)}")
    print(f"  边数量: {connectivity.get('edge_count', 0)}")
    print(f"  图密度: {connectivity.get('density', 0):.5f}")

    if "strongly_connected_components" in connectivity:
        print(
            f"  强连通分量数量: {connectivity.get('strongly_connected_components', 0)}"
        )

        largest_scc = connectivity.get("largest_strongly_connected_size", 0)
        if largest_scc > 0:
            print(f"  最大强连通分量大小: {largest_scc}")

    if "weakly_connected_components" in connectivity:
        print(f"  弱连通分量数量: {connectivity.get('weakly_connected_components', 0)}")

        largest_wcc = connectivity.get("largest_weakly_connected_size", 0)
        if largest_wcc > 0:
            print(f"  最大弱连通分量大小: {largest_wcc}")

    # 查找环路
    cycles = path_finder.find_cycles()
    print(f"\n找到 {len(cycles)} 个环路")

    # 显示一些环路示例
    for i, cycle in enumerate(cycles[:3], 1):
        print(f"环路 {i}: {' -> '.join(cycle)}")

    # 查找关键节点
    critical_nodes = path_finder.find_critical_nodes()
    print(f"\n找到 {len(critical_nodes)} 个关键节点")
    for i, node in enumerate(critical_nodes[:5], 1):
        block = system.cad_graph.get_block(node)
        if block:
            print(f"关键节点 {i}: {block.name} (ID: {node})")


def visualize_graph(system):
    """图可视化示例"""
    print("\n=== 图可视化 ===")

    try:
        # 创建可视化器
        visualizer = GraphVisualizer(system.cad_graph)

        # 获取物理布局
        positions = visualizer.use_physical_positions()

        # 获取节点颜色（基于是否为箭头）
        node_colors = visualizer.get_node_colors(attribute="is_arrow")

        # 获取边颜色（基于是否有明确方向）
        edge_colors = visualizer.get_edge_colors(attribute="explicit_direction")

        # 使用NetworkX绘图
        G = system.cad_graph.to_networkx()

        plt.figure(figsize=(12, 10))

        # 转换颜色到列表格式
        node_color_list = [node_colors.get(node, "#1f77b4") for node in G.nodes()]

        # 绘制节点
        nx.draw_networkx_nodes(
            G, positions, node_size=300, node_color=node_color_list, alpha=0.8
        )

        # 绘制边
        for u, v, data in G.edges(data=True):
            # 确定边的样式
            if data.get("explicit_direction", False):
                # 有向边
                nx.draw_networkx_edges(
                    G,
                    positions,
                    edgelist=[(u, v)],
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=15,
                    edge_color=edge_colors.get((u, v), "k"),
                    width=2.0,
                    alpha=0.7,
                )
            else:
                # 无向边
                nx.draw_networkx_edges(
                    G,
                    positions,
                    edgelist=[(u, v)],
                    arrows=False,
                    edge_color=edge_colors.get((u, v), "k"),
                    style=(
                        "dashed"
                        if data.get("connection_type", "") == "indirect"
                        else "solid"
                    ),
                    width=1.5,
                    alpha=0.6,
                )

        # 获取块名称作为标签
        labels = {}
        for node in G.nodes():
            block = system.cad_graph.get_block(node)
            if block:
                # 使用截断的块名作为标签
                name = block.name
                if len(name) > 15:
                    name = name[:12] + "..."
                labels[node] = name

        # 绘制标签
        nx.draw_networkx_labels(
            G, positions, labels=labels, font_size=8, font_family="sans-serif"
        )

        plt.title("CAD图元素关系图")
        plt.axis("off")

        # 保存图像
        plt.savefig("cad_graph.png", dpi=300, bbox_inches="tight")
        print("关系图已保存为 cad_graph.png")

        # 尝试显示图像
        plt.show()

    except Exception as e:
        print(f"图可视化时出错: {e}")


def custom_block_analysis(system):
    """自定义块分析示例"""
    print("\n=== 自定义块分析 ===")

    # 查找具有特定特征的块
    # 例如：查找入度大于2且出度大于0的非箭头块
    criteria = {"in_degree": (">", 2), "out_degree": (">", 0), "is_arrow": False}

    blocks = system.query_interface.find_blocks_by_criteria(criteria)

    print(f"找到 {len(blocks)} 个满足条件的块")

    if blocks:
        print("\n块详情:")
        for i, block in enumerate(blocks[:5], 1):  # 只显示前5个
            block_id = block.id
            in_degree = system.get_in_degree(block_id)
            out_degree = system.get_out_degree(block_id)

            print(f"{i}. {block.name} (ID: {block_id})")
            print(f"   入度: {in_degree}, 出度: {out_degree}")

            # 分析这些块的连接
            if i <= 3:  # 只详细分析前3个
                in_connections = system.query_interface.get_in_connections(block_id)
                out_connections = system.query_interface.get_out_connections(block_id)

                # 查找最常连接的块类型
                connected_types = {}

                for conn in in_connections:
                    block_matches = system.block_identifier.identify_block(
                        conn.source_block
                    )
                    block_type = block_matches[0][0] if block_matches else "unknown"
                    connected_types[block_type] = connected_types.get(block_type, 0) + 1

                for conn in out_connections:
                    block_matches = system.block_identifier.identify_block(
                        conn.target_block
                    )
                    block_type = block_matches[0][0] if block_matches else "unknown"
                    connected_types[block_type] = connected_types.get(block_type, 0) + 1

                print("   连接的块类型:")
                for block_type, count in sorted(
                    connected_types.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"     {block_type}: {count}")


def main():
    """高级功能示例"""
    # 创建系统实例
    system = CADAnalysisSystem()
    print("CAD分析系统初始化完成")

    # 加载块特征模板（如果有）
    template_file = "block_templates.json"
    if os.path.exists(template_file):
        system.load_block_templates(template_file)
        print("已加载块特征模板")

    # 分析CAD文件
    file_path = input("请输入要分析的CAD文件路径: ").strip()

    if not file_path:
        # 使用默认示例文件
        file_path = "examples/example.dxf"

    if os.path.exists(file_path):
        print(f"\n正在分析文件: {file_path}")

        success = system.analyze_file(file_path)

        if success:
            print("文件分析成功！")

            # 执行高级分析
            analyze_block_clustering(system)
            analyze_spatial_layout(system)
            analyze_connection_patterns(system)
            analyze_paths_and_cycles(system)
            custom_block_analysis(system)

            # 可视化（最后执行，因为会显示图形）
            visualize_graph(system)
        else:
            print("文件分析失败")
    else:
        print(f"文件 {file_path} 不存在")


if __name__ == "__main__":
    main()
