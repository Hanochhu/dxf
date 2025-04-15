#!/usr/bin/env python
"""
DXF可视化工具
提供从命令行直接可视化DXF文件的功能
结合了dxf_visualization_example.py的示例功能和dxf_visualize_tool.py的完整参数控制

功能包括:
- 基本DXF文件可视化
- 块边界和内部结构显示
- 连接关系分析
- 图层过滤
- 自定义视图范围和显示样式
"""

import sys
import os
import argparse
from pathlib import Path
import traceback
import matplotlib

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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DXF文件可视化工具")

    # 必要参数
    parser.add_argument("dxf_file", type=str, help="要可视化的DXF文件路径")

    # 可选参数
    parser.add_argument(
        "-o", "--output", type=str, help="输出图像文件路径", default=None
    )
    parser.add_argument(
        "--no-display", action="store_true", help="不显示可视化结果，只保存图像"
    )
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式，显示详细信息"
    )
    parser.add_argument("--show-layers", action="store_true", help="显示图层信息")

    # 内容控制
    parser.add_argument(
        "--no-block_definitions", action="store_true", help="不显示块边界和内部结构"
    )
    parser.add_argument("--no-labels", action="store_true", help="不显示块名称标签",default=True)
    parser.add_argument(
        "--block-mode",
        type=str,
        choices=["boundary", "structure"],
        default="structure",
        help="块显示模式: boundary(只显示边界框) 或 structure(显示完整内部结构)",
    )
    parser.add_argument("--show-connections", action="store_true", help="显示连接关系")
    parser.add_argument(
        "--highlight", type=str, help="高亮显示指定名称的块", default=None
    )
    parser.add_argument(
        "--layer", type=str, help="仅显示指定的图层（多个图层用逗号分隔）", default=None
    )

    # 视图控制
    parser.add_argument("--no-grid", action="store_true", help="不显示网格")
    parser.add_argument("--no-axis", action="store_true", help="不显示坐标轴")
    parser.add_argument(
        "--figsize",
        type=str,
        help='图像尺寸，格式为"宽度,高度"，单位为英寸',
        default="12,8",
    )
    parser.add_argument("--dpi", type=int, help="图像分辨率", default=100)
    parser.add_argument(
        "--view-range",
        type=str,
        help='视图范围，格式为"min_x,min_y,max_x,max_y"',
        default=None,
    )
    parser.add_argument("--margin", type=float, help="视图边距", default=20)
    parser.add_argument(
        "--auto-scale", action="store_true", help="自动根据内容调整视图比例"
    )
    parser.add_argument("--label-size", type=float, help="块标签文字大小", default=9.0)

    # 颜色控制
    parser.add_argument("--bg-color", type=str, help="背景颜色", default="#f5f5f5")
    parser.add_argument("--line-color", type=str, help="线段颜色", default="#1f77b4")
    parser.add_argument(
        "--highlight-color", type=str, help="高亮颜色", default="#d62728"
    )
    parser.add_argument("--no-blocks", action="store_true", help="不渲染块定义")
    return parser.parse_args()


def main():
    """主函数
    处理命令行参数并执行DXF文件可视化
    结合了示例文件和工具文件的实现
    """
    # 解析命令行参数
    args = parse_args()

    # 检查文件是否存在
    dxf_path = Path(args.dxf_file)
    if not dxf_path.exists():
        print(f"错误：找不到文件 {args.dxf_file}")
        return 1

    # 解析DXF文件
    print(f"正在解析DXF文件: {args.dxf_file}")
    try:
        parser = DXFParser()
        # 解析前启用调试模式
        if args.debug:
            print(f"启用调试模式...")
            parser.debug = True

        # parse_file 返回一个包含四个元素的元组: (entities, block_definitions, block_references, additional_info)
        parse_result = parser.parse_file(args.dxf_file)

        # 正确提取实体、块定义和块引用
        if isinstance(parse_result, tuple) and len(parse_result) >= 4:
            entities, block_definitions, block_references, additional_info = (
                parse_result
            )
        else:
            # 兼容旧版本或其他返回类型
            if isinstance(parse_result, dict):
                entities = parse_result.get("entities", [])
                block_definitions = parse_result.get(
                    "block_definitions", parse_result.get("blocks", [])
                )
                block_references = parse_result.get("block_references", [])
            else:
                entities = getattr(parse_result, "entities", [])
                block_definitions = getattr(parse_result, "block_definitions", [])
                block_references = getattr(parse_result, "block_references", [])

        # 调试信息: 显示解析结果
        if args.debug:
            print(f"\n=== DXF解析结果摘要 ===")
            print(
                f"解析完成: 找到 {len(entities)} 个实体, {len(block_definitions)} 个块定义, {len(block_references)} 个块引用"
            )

            # 实体类型统计
            entity_types = {}
            for entity in entities:
                etype = getattr(entity, "entity_type", "UNKNOWN")
                if not isinstance(etype, str):
                    if hasattr(etype, "name"):
                        etype = etype.name
                    else:
                        etype = str(etype)
                entity_types[etype] = entity_types.get(etype, 0) + 1

            print("实体类型统计:", entity_types)

            # 显示块定义信息
            if block_definitions and len(block_definitions) > 0:
                print("\n块定义信息:")
                for i, block in enumerate(block_definitions[:5]):  # 只打印前5个
                    print(
                        f"  块定义 {i+1}: {block.name} (含 {len(block.entities) if hasattr(block, 'entities') else 0} 个实体)"
                    )
                if len(block_definitions) > 5:
                    print(f"  ...及其他 {len(block_definitions)-5} 个块定义")
                    
            # 显示块引用信息
            if block_references and len(block_references) > 0:
                print("\n块引用信息:")
                for i, block_ref in enumerate(block_references[:5]):  # 只打印前5个
                    print(
                        f"  块引用 {i+1}: {getattr(block_ref, 'name', 'N/A')} (位置: {getattr(block_ref, 'position', 'N/A')})"
                    )
                if len(block_references) > 5:
                    print(f"  ...及其他 {len(block_references)-5} 个块引用")

        # 显示图层信息
        if args.show_layers:
            layer_stats = {}
            for entity in entities:
                if hasattr(entity, "layer"):
                    layer = entity.layer
                    if layer not in layer_stats:
                        layer_stats[layer] = {"count": 0, "types": {}}

                    layer_stats[layer]["count"] += 1

                    etype = getattr(entity, "entity_type", "UNKNOWN")
                    if not isinstance(etype, str):
                        if hasattr(etype, "name"):
                            etype = etype.name
                        else:
                            etype = str(etype)

                    layer_stats[layer]["types"][etype] = (
                        layer_stats[layer]["types"].get(etype, 0) + 1
                    )

            print("\n=== 图层信息 ===")
            print(f"找到 {len(layer_stats)} 个图层")

            # 按实体数量排序图层
            sorted_layers = sorted(
                layer_stats.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for layer_name, stats in sorted_layers:
                print(f"图层: {layer_name} ({stats['count']} 个实体)")
                # 显示每个图层中的实体类型
                type_info = ", ".join(
                    [f"{etype}: {count}" for etype, count in stats["types"].items()]
                )
                print(f"  类型: {type_info}")

        # 如果指定了图层过滤，只保留指定图层的实体
        if args.layer:
            target_layers = args.layer.split(",")
            filtered_entities = []
            # 过滤主实体
            for entity in entities:
                if hasattr(entity, "layer") and entity.layer in target_layers:
                    filtered_entities.append(entity)

            # 同时过滤块内实体
            filtered_blocks = []
            for block in block_definitions:
                block_entities = []
                if hasattr(block, "entities"):
                    for entity in block.entities:
                        if hasattr(entity, "layer") and entity.layer in target_layers:
                            block_entities.append(entity)

                # 只保留包含指定图层实体的块
                if block_entities:
                    # 创建块的副本，只保留过滤后的实体
                    import copy

                    filtered_block = copy.copy(block)
                    filtered_block.entities = block_entities
                    filtered_blocks.append(filtered_block)

            if not filtered_entities and not filtered_blocks:
                print(f"警告: 未找到图层 '{args.layer}' 中的实体")
            else:
                print(
                    f"已过滤，仅显示图层 '{args.layer}' 中的 {len(filtered_entities)} 个实体和 {len(filtered_blocks)} 个块"
                )
                entities = filtered_entities
                block_definitions = filtered_blocks

    except Exception as e:
        print(f"错误: 无法解析DXF文件: {str(e)}")
        if args.debug:
            print("\n=== DXF解析错误 ===")
            print("\n=== 错误详情 ===")
            traceback.print_exc()
        return 1

    # 创建可视化器
    visualizer = DXFVisualizer()

    # 启用调试模式
    if args.debug:
        visualizer.set_debug_mode(True)

    # 应用可视化参数
    visualizer.show_grid = not args.no_grid
    visualizer.show_axis = not args.no_axis
    visualizer.margin = args.margin

    # 控制是否显示块标签
    visualizer.show_block_labels = not args.no_labels
    visualizer.block_label_size = args.label_size

    # 设置块显示模式
    visualizer.block_display_mode = args.block_mode

    # 设置颜色
    visualizer.set_colors(
        {
            "background": args.bg_color,
            "line": args.line_color,
            "highlight": args.highlight_color,
        }
    )

    # 准备图表尺寸
    try:
        width, height = map(float, args.figsize.split(","))
        figsize = (width, height)
    except:
        print("警告：无法解析图表尺寸，使用默认值(12,8)")
        figsize = (12, 8)

    # # 创建图表
    # visualizer.create_figure(figsize=figsize, dpi=args.dpi)

    # 如果需要，识别块并分析连接
    connections = []
    if args.show_connections:
        print("正在分析连接关系...")
        # 识别块类型
        block_identifier = BlockIdentifier()
        for block in block_definitions:
            block_identifier.identify_block(block)

        # 分析连接
        connection_analyzer = ConnectionAnalyzer(block_identifier)
        # 找出所有线段实体
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
        connections = connection_analyzer.find_connections(block_definitions, lines)
        print(f"找到 {len(connections)} 个连接")

    # 高亮特定块
    highlight_block_ids = []
    if args.highlight:
        for block in block_definitions:
            if args.highlight.lower() in block.name.lower():
                highlight_block_ids.append(block.id)
                print(f"将高亮显示块: {block.name}")

    # 确定是否显示块，默认显示，除非明确禁用
    if args.no_blocks:
        render_blocks = None
        render_block_definitions = None
    elif args.block_mode == "structure":
        render_blocks = block_references
        render_block_definitions = block_definitions
        # 确保每个块引用都能找到对应的块定义
        if render_blocks and render_block_definitions:
            block_def_map = {block.name: block for block in render_block_definitions if hasattr(block, 'name')}
            for block_ref in render_blocks:
                if hasattr(block_ref, 'name') and block_ref.name in block_def_map:
                    if not hasattr(block_ref, 'block') or block_ref.block is None:
                        block_ref.block = block_def_map[block_ref.name]
    else:
        render_blocks = block_references
        render_block_definitions = None

    # 设置调试模式
    if args.debug:
        visualizer.set_debug_mode(True)
        print(f"渲染模式: {'structure' if args.block_mode == 'structure' else 'boundary'}")
        print(f"块引用数量: {len(block_references) if block_references else 0}")
        print(f"块定义数量: {len(block_definitions) if block_definitions else 0}")

    # 渲染DXF内容
    print("正在渲染DXF内容...")

    # 确定要渲染的内容
    render_connections = connections if args.show_connections else None

    # 设置自定义视图范围
    focus_area = None
    if args.view_range:
        try:
            focus_area = tuple(map(float, args.view_range.split(",")))
            if len(focus_area) != 4:
                print(
                    "警告: 视图范围格式错误，应为'min_x,min_y,max_x,max_y'，使用自动范围"
                )
                focus_area = None
            else:
                print(f"使用自定义视图范围: {focus_area}")
        except:
            print("警告: 无法解析视图范围，使用自动范围")
    visualizer.render_dxf(
        entities=entities,
        blocks=render_blocks,
        connections=render_connections,
        focus_area=focus_area,
        block_definitions=render_block_definitions,
    )

    # 高亮指定块
    if highlight_block_ids and render_blocks:
        visualizer.render_blocks(
            blocks=[b for b in render_blocks if b.id in highlight_block_ids],
            highlight_ids=highlight_block_ids,
            block_definitions=render_block_definitions
        )

    # 保存图像
    if args.output:
        print(f"正在保存图像到: {args.output}")
        visualizer.save_image(args.output, dpi=args.dpi)

    # 显示结果
    if not args.no_display:
        print("正在显示可视化结果...")
        visualizer.show()
    else:
        visualizer.close()

    print("可视化完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
