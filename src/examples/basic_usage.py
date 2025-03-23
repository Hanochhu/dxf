"""
基本使用示例
演示CAD分析系统的基本功能
"""

import os
import sys
import json

print(sys.path)  # 在导入之前打印Python路径
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)  # 在导入之前打印Python路径

from core.data_structures import BlockFeature, EntityType
from system.cad_analysis_system import CADAnalysisSystem, CADAnalysisConfig


def main():
    """基本功能示例"""
    # 创建系统实例
    system = CADAnalysisSystem()
    print("CAD分析系统初始化完成")
    
    # 加载配置（如果有）
    config_file = "config.json"
    if os.path.exists(config_file):
        config = CADAnalysisConfig()
        if config.load_from_file(config_file):
            config.apply_to_system(system)
            print("已加载配置")
    
    # 加载块特征模板（如果有）
    template_file = "block_templates.json"
    if os.path.exists(template_file):
        if system.load_block_templates(template_file):
            print("已加载块特征模板")
    else:
        # 手动定义一些常见块特征
        valve_feature = BlockFeature(
            name="valve",
            description="阀门特征",
            entity_types={EntityType.CIRCLE, EntityType.LINE},
            min_entity_count=3,
            max_entity_count=10,
            min_aspect_ratio=0.8,
            max_aspect_ratio=1.2
        )
        system.add_block_feature(valve_feature)
        
        instrument_feature = BlockFeature(
            name="instrument",
            description="仪表特征",
            entity_types={EntityType.CIRCLE, EntityType.LINE, EntityType.TEXT},
            min_entity_count=5,
            max_entity_count=15,
            min_aspect_ratio=0.8,
            max_aspect_ratio=1.2
        )
        system.add_block_feature(instrument_feature)
        
        arrow_feature = BlockFeature(
            name="arrow",
            description="箭头特征",
            entity_types={EntityType.LINE, EntityType.POLYLINE},
            min_entity_count=1,
            max_entity_count=5,
            min_aspect_ratio=1.5,
            max_aspect_ratio=10.0
        )
        system.add_block_feature(arrow_feature)
        
        print("已手动添加常见块特征")
        
        # 保存特征模板供将来使用
        system.save_block_templates(template_file)
    
    # 分析DXF文件
    file_path = input("请输入要分析的CAD文件路径: ").strip()
    
    if not file_path:
        # 使用默认示例文件
        file_path = "examples/example.dxf"
    
    if os.path.exists(file_path):
        print(f"\n正在分析文件: {file_path}")
        
        success = system.analyze_file(file_path)
        
        if success:
            print("文件分析成功！")
            
            # 输出统计信息
            stats = system.get_statistics()
            print(f"\n统计信息:")
            print(f"实体数量: {stats['entity_count']}")
            print(f"块数量: {stats['block_count']}")
            print(f"连接数量: {stats['connection_count']}")
            print(f"箭头数量: {stats['arrow_count']}")
            
            # 查找特定类型的块
            block_types = ["valve", "instrument", "arrow"]
            for block_type in block_types:
                blocks = system.get_blocks_by_type(block_type)
                print(f"\n找到 {len(blocks)} 个 {block_type} 类型的块")
            
            # 分析中心块
            central_blocks = system.find_central_blocks(top_n=5)
            print("\n最中心的5个块:")
            for i, block_info in enumerate(central_blocks, 1):
                print(f"{i}. {block_info['block_name']} (ID: {block_info['block_id']})")
                print(f"   中心度: {block_info['centrality']:.4f}, 入度: {block_info['in_degree']}, 出度: {block_info['out_degree']}")
            
            # 检查连接关系
            if central_blocks and len(central_blocks) >= 2:
                block1_id = central_blocks[0]['block_id']
                block2_id = central_blocks[1]['block_id']
                
                # 检查连接
                has_connection = system.has_connection(block1_id, block2_id)
                print(f"\n块 {block1_id} 到 {block2_id} 的连接: {has_connection}")
                
                # 检查路径
                has_path = system.check_block_has_path(block1_id, block2_id)
                print(f"块 {block1_id} 到 {block2_id} 的路径: {has_path}")
                
                # 获取第一个块的连接信息
                connections = system.get_block_connections(block1_id)
                print(f"\n块 {block1_id} 的连接信息:")
                print(f"入度: {connections['in_degree']}")
                print(f"出度: {connections['out_degree']}")
                
                if 'in_connections' in connections and connections['in_connections']:
                    print("入连接:")
                    for conn in connections['in_connections'][:3]:  # 只显示前3个
                        print(f"  来自: {conn['from_block_name']} (ID: {conn['from_block_id']})")
                
                if 'out_connections' in connections and connections['out_connections']:
                    print("出连接:")
                    for conn in connections['out_connections'][:3]:  # 只显示前3个
                        print(f"  到: {conn['to_block_name']} (ID: {conn['to_block_id']})")
            
            # 查找社区
            communities = system.find_communities()
            print(f"\n找到 {len(communities)} 个社区")
            for i, (community_id, block_ids) in enumerate(communities.items(), 1):
                if i <= 3:  # 只显示前3个社区
                    print(f"社区 {community_id}: {len(block_ids)} 个块")
            
            # 保存分析结果
            output_file = "analysis_result.json"
            if system.save_analysis(output_file):
                print(f"\n分析结果已保存到: {output_file}")
            
            # 导出图形可视化
            try:
                graph_file = "graph.gexf"
                if system.export_graphviz(graph_file, "gexf"):
                    print(f"图形结构已导出到: {graph_file}")
            except Exception as e:
                print(f"导出图形时出错: {e}")
            
            # 自然语言查询示例
            print("\n自然语言查询示例:")
            queries = [
                "find blocks of type valve",
                "what is the indegree of block " + block1_id,
                "is there a connection from " + block1_id + " to " + block2_id,
                "find the path from " + block1_id + " to " + block2_id
            ]
            
            for query in queries:
                print(f"\n查询: {query}")
                result = system.query(query)
                print(f"结果: {result['message']}")
        else:
            print("文件分析失败")
    else:
        print(f"文件 {file_path} 不存在")


if __name__ == "__main__":
    main()