"""
DXF可视化示例
演示如何使用可视化模块渲染DXF文件内容
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(script_dir)  # src目录路径
project_root = os.path.dirname(src_path)  # 项目根目录
sys.path.append(project_root)

# 导入相关模块
from src.parsers.dxf_parser import DXFParser
from src.feature.block_identifier import BlockIdentifier
from src.connection.connection_analyzer import ConnectionAnalyzer
from src.visualization.dxf_visualizer import DXFVisualizer


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DXF可视化示例")
    parser.add_argument("dxf_file", type=str, help="要可视化的DXF文件路径")
    parser.add_argument(
        "--output", type=str, help="输出图像文件路径（可选）", default=None
    )
    parser.add_argument(
        "--highlight", type=str, help="高亮显示的块名称（可选）", default=None
    )
    parser.add_argument("--show-blocks", action="store_true", help="是否显示块边界")
    parser.add_argument(
        "--show-connections", action="store_true", help="是否显示连接关系"
    )
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    # 检查文件是否存在
    dxf_path = Path(args.dxf_file)
    if not dxf_path.exists():
        print(f"错误: 找不到文件 {args.dxf_file}")
        return 1

    # 解析DXF文件
    print(f"正在解析DXF文件: {args.dxf_file}...")
    parser = DXFParser()

    # 如果启用了调试模式
    if args.debug:
        parser.debug = True

    try:
        # parse_file 返回一个包含四个元素的元组: (entities, block_definitions, block_references, additional_info)
        parse_result = parser.parse_file(args.dxf_file)

        # 正确提取实体、块定义和块引用
        if isinstance(parse_result, tuple) and len(parse_result) >= 4:
            entities, block_definitions, block_references, additional_info = parse_result
            if args.debug:
                print(f"解析完成: 找到 {len(entities)} 个实体、{len(block_definitions)} 个块定义、{len(block_references)} 个块引用")
        else:
            # 兼容旧版本或其他返回类型
            if isinstance(parse_result, dict):
                entities = parse_result.get("entities", [])
                blocks = parse_result.get("blocks", [])
            else:
                entities = getattr(parse_result, "entities", [])
                blocks = getattr(parse_result, "blocks", [])
    except Exception as e:
        print(f"错误: 无法解析DXF文件: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    # 创建可视化器
    visualizer = DXFVisualizer()

    # 如果需要，识别块并分析连接
    connections = []
    if args.show_connections:
        print("正在分析连接关系...")
        # 识别块类型
        block_identifier = BlockIdentifier()
        for block in blocks:
            block_identifier.identify_block(block)

        # 分析连接
        connection_analyzer = ConnectionAnalyzer(block_identifier)
        # 找出所有线段
        lines = []
        for entity in entities:
            # 检查实体类型，兼容枚举和字符串类型
            entity_type = getattr(entity, "entity_type", None)
            if entity_type:
                if isinstance(entity_type, str) and entity_type.upper() == "LINE":
                    lines.append(entity)
                elif hasattr(entity_type, "name") and entity_type.name == "LINE":
                    lines.append(entity)

        # 分析连接关系
        connections = connection_analyzer.find_connections(blocks, lines)
        print(f"找到 {len(connections)} 个连接")

    # 高亮特定块
    highlight_block_ids = []
    if args.highlight:
        for block in blocks:
            if args.highlight.lower() in block.name.lower():
                highlight_block_ids.append(block.id)
                print(f"将高亮显示块: {block.name}")

    # 渲染DXF内容
    print("正在渲染DXF内容...")

    # 确定要渲染的内容
    render_blocks = blocks if args.show_blocks else None
    render_connections = connections if args.show_connections else None

    # 渲染
    visualizer.render_dxf(
        entities=entities, blocks=render_blocks, connections=render_connections
    )

    # 高亮指定块
    if highlight_block_ids and render_blocks:
        visualizer.render_blocks(
            blocks=[b for b in blocks if b.id in highlight_block_ids],
            highlight_ids=highlight_block_ids,
        )

    # 保存图像
    if args.output:
        visualizer.save_image(args.output)

    # 显示结果
    print("正在显示可视化结果...")
    visualizer.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
