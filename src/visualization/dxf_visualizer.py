"""
DXF文件可视化器
提供DXF文件的可视化和渲染功能
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from matplotlib.font_manager import FontProperties
import random

from src.core.data_structures import (
    Point,
    Entity,
    LineEntity,
    CircleEntity,
    ArcEntity,
    TextEntity,
    Block,
    EntityType,
    BlockReference,
)
from src.visualization.entity_renderer import EntityRenderer

# 添加中文字体支持
try:
    # 尝试使用系统中文字体
    font_path = "/System/Library/Fonts/PingFang.ttc"  # MacOS 中文字体
    chinese_font = FontProperties(fname=font_path)
except:
    try:
        # 尝试使用matplotlib内置的中文字体
        chinese_font = FontProperties(family="SimHei")  # 黑体
    except:
        chinese_font = FontProperties()  # 如果都失败，使用默认字体
        print("警告: 未能加载中文字体，中文显示可能有问题")


class DXFVisualizer:
    """
    DXF文件可视化器类

    提供以下功能：
    1. 直接渲染DXF文件中的实体
    2. 可视化块和其内部结构
    3. 高亮显示连接和关系
    4. 导出为图像或交互式查看
    """

    def __init__(self):
        """初始化DXF可视化器"""
        self.entity_renderer = EntityRenderer()
        self.fig = None
        self.ax = None

        # 可视化配置
        self.colors = {
            "background": "#f5f5f5",
            "grid": "#cccccc",
            "line": "#1f77b4",
            "circle": "#ff7f0e",
            "arc": "#2ca02c",
            "text": "#9467bd",
            "ellipse": "#8c564b",
            "block": "#e377c2",
            "connection": "#7f7f7f",
            "highlight": "#d62728",
        }

        self.line_styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-.",
        }

        # 视图设置
        self.margin = 20  # 边距
        self.scale_factor = 1.0  # 缩放因子
        self.show_grid = False  # 是否显示网格（默认关闭，避免多余虚线）
        self.show_axis = True  # 是否显示坐标轴
        self.interactive = True  # 是否启用交互式视图
        self.debug_mode = False  # 调试模式标志

        # 块显示设置
        self.show_block_labels = True  # 是否显示块标签
        self.block_label_size = 9.0  # 块标签字体大小
        self.block_display_mode = (
            "structure"  # 块显示模式: "boundary"(边界框) 或 "structure"(内部结构)
        )

    def set_debug_mode(self, debug: bool = True):
        """
        设置调试模式

        Args:
            debug: 是否启用调试模式
        """
        self.debug_mode = debug
        if debug:
            print("调试模式已启用 - 将显示更多信息")

    def set_colors(self, color_dict: Dict[str, str]):
        """
        设置可视化颜色

        Args:
            color_dict: 颜色字典，键为实体类型，值为颜色代码
        """
        self.colors.update(color_dict)

    def create_figure(self, figsize=(12, 8), dpi=100):
        """
        创建图形

        Args:
            figsize: 图形尺寸
            dpi: 分辨率
        """
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.colors["background"])

        # 应用网格和坐标轴设置
        if not self.show_axis:
            self.ax.set_axis_off()

        if self.show_grid:
            self.ax.grid(True, linestyle="--", color=self.colors["grid"], alpha=0.7)

    def render_entities(self, entities: List[Entity], highlight_ids: List[str] = None):
        """
        渲染实体列表

        Args:
            entities: 实体列表
            highlight_ids: 高亮实体ID列表
        """
        if self.fig is None or self.ax is None:
            self.create_figure()

        highlight_ids = highlight_ids or []

        # 打印调试信息
        if self.debug_mode:
            print(f"正在渲染 {len(entities)} 个实体")
            entity_types = {}
            for entity in entities:
                etype = getattr(entity, "entity_type", "UNKNOWN")
                if isinstance(etype, EntityType):
                    etype = etype.name
                entity_types[etype] = entity_types.get(etype, 0) + 1
            print(f"实体类型统计: {entity_types}")

        # 渲染每个实体
        for entity in entities:
            if isinstance(entity, Block):  # 跳过Block定义
                continue
            is_highlighted = entity.id in highlight_ids

            self.entity_renderer.render_entity(
                entity=entity,
                ax=self.ax,
                color=(
                    self.colors["highlight"]
                    if is_highlighted
                    else self._get_entity_color(entity)
                ),
                linewidth=1.5 if is_highlighted else 1.0,
                linestyle=self.line_styles["solid"],
                alpha=1.0,
            )

    def render_blocks(
        self,
        blocks: List[BlockReference],
        highlight_ids: List[str] = None,
        block_definitions: List[Block] = None,
    ):
        """
        渲染块列表

        Args:
            blocks: 块引用列表
            highlight_ids: 高亮块ID列表
            block_definitions: 块定义列表，用于在渲染块引用时提供块定义
        """
        if not blocks:
            if self.debug_mode:
                print("[警告] 没有块引用传入 render_blocks 方法")
            return
            
        # 输出传入的块引用信息，便于检查
        print("[块引用检查] 传入blocks参数类型: ", type(blocks))
        if blocks:
            for i, block in enumerate(blocks[:3]):  # 只显示前三个块，避免输出过多
                print(f"[块引用检查] 块{i+1}: 类型={type(block)}, 名称={getattr(block, 'name', None)}, ID={getattr(block, 'id', None)}, 边界={'有' if hasattr(block, 'block') and block.block else '无'}")
        
        if block_definitions:
            print(f"[块定义检查] 传入块定义数量: {len(block_definitions)}")
            # 显示几个块定义的名称，帮助诊断问题
            for i, block_def in enumerate(block_definitions[:3]):
                print(f"[块定义检查] 块定义{i+1}: 名称={getattr(block_def, 'name', None)}, ID={getattr(block_def, 'id', None)}, 实体数量={len(getattr(block_def, 'entities', []))}")
        
        # 首先尝试修复具有相同名称但不同大小写的块引用和块定义
        if block_definitions:
            # 创建不区分大小写的块定义映射
            case_insensitive_map = {}
            for block_def in block_definitions:
                if hasattr(block_def, 'name'):
                    name_lower = block_def.name.lower()
                    if name_lower not in case_insensitive_map:
                        case_insensitive_map[name_lower] = []
                    case_insensitive_map[name_lower].append(block_def)
                    
            # 处理没有关联块定义的块引用
            for block_ref in blocks:
                if not hasattr(block_ref, 'block') or block_ref.block is None:
                    if hasattr(block_ref, 'name'):
                        name_lower = block_ref.name.lower()
                        if name_lower in case_insensitive_map and case_insensitive_map[name_lower]:
                            block_ref.block = case_insensitive_map[name_lower][0]
                            if self.debug_mode:
                                print(f"[修复] 通过不区分大小写匹配为块引用 '{block_ref.name}' 找到块定义 '{block_ref.block.name}'")
        
        # 首先构建一个块定义字典，便于快速查找
        block_def_map = {}
        if block_definitions:
            block_def_map = {block.name: block for block in block_definitions if hasattr(block, 'name')}
            
        # 检查块引用与块定义的匹配情况
        if blocks and block_definitions:
            matched_count = 0
            for block_ref in blocks:
                if hasattr(block_ref, 'name') and block_ref.name in block_def_map:
                    matched_count += 1
                    # 预先关联块引用和块定义
                    if not hasattr(block_ref, 'block') or block_ref.block is None:
                        block_ref.block = block_def_map[block_ref.name]
            
            print(f"[块匹配检查] 成功匹配块定义的块引用数量: {matched_count}/{len(blocks)}")
            
        if self.fig is None or self.ax is None:
            self.create_figure()

        highlight_ids = highlight_ids or []

        # 打印调试信息
        if self.debug_mode:
            print(f"正在渲染 {len(blocks)} 个块")
            for i, block in enumerate(blocks[:5]):  # 只显示前5个块
                print(
                    f"块 {i+1}: 名称={block.name}, ID={block.id}, 实体数={len(block.entities) if hasattr(block, 'entities') else 0}"
                )
                if hasattr(block, "bounding_box") and block.bounding_box:
                    print(
                        f"  边界: 最小点=({block.bounding_box.min_point.x}, {block.bounding_box.min_point.y}), "
                        f"最大点=({block.bounding_box.max_point.x}, {block.bounding_box.max_point.y})"
                    )

            # 显示块的渲染模式
            print(f"块渲染模式: {self.block_display_mode}")

        # 生成块的随机颜色（但保持一致性）
        block_colors = {}
        random.seed(42)  # 使用固定种子，确保每次运行颜色一致

        # 为每个块分配一个唯一的颜色
        for block in blocks:
            if block.id not in block_colors:
                # 生成随机颜色，但避免与背景颜色太接近
                while True:
                    r = random.random() * 0.7 + 0.3  # 0.3-1.0 更明亮的颜色
                    g = random.random() * 0.7 + 0.3
                    b = random.random() * 0.7 + 0.3
                    # 确保颜色与背景有足够对比度
                    if r + g + b < 2.0:  # 避免颜色太浅
                        block_colors[block.id] = (r, g, b)
                        break

        # 渲染每个块
        for block in blocks:
            is_highlighted = block.id in highlight_ids

            # 确定块的颜色
            if is_highlighted:
                block_color = self.colors["highlight"]
            else:
                block_color = block_colors.get(block.id, self.colors["block"])

            # 边界模式: 只渲染边界框
            if self.block_display_mode == "boundary" and block.bounding_box:
                width = block.bounding_box.width
                height = block.bounding_box.height

                # 创建精确的边界框
                rect = patches.Rectangle(
                    (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                    width,
                    height,
                    linewidth=1.5 if is_highlighted else 1.0,
                    edgecolor=block_color,
                    facecolor="none",  # 无填充
                    alpha=1.0,  # 完全不透明
                    zorder=10,
                    linestyle="-",
                )
                self.ax.add_patch(rect)

            # 结构模式: 渲染块内部实体
            elif self.block_display_mode == "structure":
                # 结构模式：遍历块引用（BlockReference），查找块定义，做仿射变换后渲染
                from copy import deepcopy
                block_ref = block  # blocks 实际为 BlockReference
                if not hasattr(block_ref, "block") or block_ref.block is None:
                    # 如果块引用没有关联的块定义，尝试从传入的块定义列表中查找
                    if block_definitions and hasattr(block_ref, "name"):
                        if block_ref.name in block_def_map:
                            block_ref.block = block_def_map[block_ref.name]
                            if self.debug_mode:
                                print(f"[成功] 为块引用 '{block_ref.name}' 找到了对应的块定义")
                        else:
                            if self.debug_mode:
                                print(f"[警告] 块引用 '{block_ref.name}' 在块定义字典中未找到匹配项")
                            for block_def in block_definitions:
                                if block_def.name == block_ref.name:
                                    block_ref.block = block_def
                                    if self.debug_mode:
                                        print(f"[成功] 为块引用 '{block_ref.name}' 通过迭代找到了对应的块定义")
                                    break
                    
                    # 如果仍然没有找到块定义，则跳过此块
                    if not hasattr(block_ref, "block") or block_ref.block is None:
                        if self.debug_mode:
                            print(f"[警告] 块引用 {getattr(block_ref, 'name', None)} (ID: {getattr(block_ref, 'id', None)}) 没有关联的块定义，跳过渲染")
                        continue
                        
                block_def = block_ref.block
                
                # 检查块定义中的实体数量
                if not hasattr(block_def, "entities") or not block_def.entities:
                    if self.debug_mode:
                        print(f"[警告] 块定义 '{block_def.name}' 没有实体或实体列表为空")
                    continue
                
                if self.debug_mode and len(block_def.entities) > 0:
                    print(f"[信息] 块 '{block_def.name}' 包含 {len(block_def.entities)} 个实体")
                
                # 检查是否有缩放、旋转和位置信息 
                sx, sy, sz = getattr(block_ref, "scale", (1.0, 1.0, 1.0))
                angle_rad = math.radians(getattr(block_ref, "rotation", 0.0))
                
                # 确保位置属性存在
                if not hasattr(block_ref, "position") or block_ref.position is None:
                    if self.debug_mode:
                        print(f"[警告] 块引用 '{block_ref.name}' 缺少位置信息，将使用原点(0,0,0)")
                    dx, dy, dz = 0.0, 0.0, 0.0
                else:
                    dx, dy, dz = block_ref.position.x, block_ref.position.y, block_ref.position.z
                
                # 渲染块内的每个实体
                entity_count = 0
                transformed_entities = []
                
                # 先对所有实体应用变换，避免重复变换和引用问题
                for entity in block_def.entities:
                    entity_count += 1
                    ent = deepcopy(entity)
                    etype = getattr(ent, "entity_type", None)
                    
                    # 嵌套块引用需要特殊处理
                    if etype == EntityType.INSERT and hasattr(ent, "block") and ent.block is not None:
                        # 递归渲染嵌套块引用
                        nested_ref = deepcopy(ent)
                        # 应用变换
                        if hasattr(nested_ref, "position") and nested_ref.position:
                            nested_ref.position.x *= sx
                            nested_ref.position.y *= sy
                            x0, y0 = nested_ref.position.x, nested_ref.position.y
                            nested_ref.position.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                            nested_ref.position.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                            nested_ref.position.x += dx
                            nested_ref.position.y += dy
                        
                        # 累积旋转角度
                        if hasattr(nested_ref, "rotation"):
                            nested_ref.rotation = getattr(nested_ref, "rotation", 0.0) + getattr(block_ref, "rotation", 0.0)
                        else:
                            nested_ref.rotation = getattr(block_ref, "rotation", 0.0)
                            
                        # 累积缩放因子
                        if hasattr(nested_ref, "scale"):
                            nested_scale = nested_ref.scale
                            nested_ref.scale = (
                                nested_scale[0] * sx,
                                nested_scale[1] * sy,
                                nested_scale[2] * sz
                            )
                        else:
                            nested_ref.scale = (sx, sy, sz)
                            
                        if self.debug_mode:
                            print(f"[递归] 处理嵌套块引用: '{getattr(nested_ref, 'name', 'N/A')}', 位置=({nested_ref.position.x}, {nested_ref.position.y}), 旋转={getattr(nested_ref, 'rotation', 0.0)}")
                        
                        # 查找嵌套块的块定义
                        if (not hasattr(nested_ref, "block") or nested_ref.block is None) and block_definitions and hasattr(nested_ref, "name"):
                            for block_def in block_definitions:
                                if hasattr(block_def, "name") and block_def.name == nested_ref.name:
                                    nested_ref.block = block_def
                                    if self.debug_mode:
                                        print(f"[嵌套块] 为嵌套块 '{nested_ref.name}' 找到了对应的块定义")
                                    break
                        
                        # 递归调用
                        self.render_blocks([nested_ref], highlight_ids, block_definitions)
                        continue
                    # 处理其他实体类型
                    else:
                        # 应用通用坐标变换 (缩放、旋转、平移)
                        if etype != EntityType.INSERT:  # 避免重复处理嵌套块
                            self._apply_coordinate_transform(ent, sx, sy, angle_rad, dx, dy)
                            transformed_entities.append(ent)
                
                # 渲染所有已变换的实体
                for ent in transformed_entities:
                    try:
                        self.entity_renderer.render_entity(
                            entity=ent,
                            ax=self.ax,
                            color=block_color,
                            linewidth=1.5 if is_highlighted else 1.0,
                            linestyle="-",
                            alpha=1.0,
                            zorder=5,
                        )
                    except Exception as e:
                        if self.debug_mode:
                            print(f"渲染实体 {getattr(ent, 'id', None)} 时出错: {str(e)}")

                # 仍然显示边界框，但用虚线表示
                if hasattr(block, "bounding_box") and block.bounding_box and self._is_bounding_box_valid(block.bounding_box):
                    width = block.bounding_box.width
                    height = block.bounding_box.height

                    rect = patches.Rectangle(
                        (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                        width, height,
                        linewidth=0.8,
                        edgecolor=block_color,
                        facecolor='none',
                        alpha=0.5,
                        zorder=10,
                        linestyle='--'
                    )
                    self.ax.add_patch(rect)
                    
                # 显示块标签
                if self.show_block_labels and hasattr(block, "name") and block.name:
                    # 确定标签位置（在块边界框的左上角或中心位置）
                    if hasattr(block, "bounding_box") and block.bounding_box and self._is_bounding_box_valid(block.bounding_box):
                        label_x = block.bounding_box.min_point.x
                        label_y = block.bounding_box.max_point.y + 5  # 稍微偏上一点
                    elif hasattr(block, "position"):
                        label_x = block.position.x
                        label_y = block.position.y + 10
                    else:
                        # 默认位置
                        label_x, label_y = 0, 0
                    
                    # 添加带背景的标签
                    self.ax.text(
                        label_x, label_y,
                        block.name,
                        fontsize=self.block_label_size,
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'),
                        zorder=20,
                        ha='left', va='bottom'
                    )

    def render_connections(
        self, connections: List[Any], highlight_ids: List[str] = None
    ):
        """
        渲染连接

        Args:
            connections: 连接列表
            highlight_ids: 高亮连接ID列表
        """
        if self.fig is None or self.ax is None:
            self.create_figure()

        highlight_ids = highlight_ids or []

        # 打印调试信息
        if self.debug_mode:
            print(f"正在渲染 {len(connections)} 个连接")

        # 渲染每个连接
        for connection in connections:
            is_highlighted = connection.id in highlight_ids

            # 渲染连接中的路径段
            for segment in connection.path_segments:
                self.entity_renderer.render_entity(
                    entity=segment,
                    ax=self.ax,
                    color=(
                        self.colors["highlight"]
                        if is_highlighted
                        else self.colors["connection"]
                    ),
                    linewidth=2.0 if is_highlighted else 1.5,
                    linestyle=self.line_styles["solid"],
                    alpha=1.0,
                )

            # 添加方向标记（如果有明确方向）
            if connection.has_explicit_direction and connection.path_segments:
                segment = connection.path_segments[0]
                if hasattr(segment, "start_point") and hasattr(segment, "end_point"):
                    # 计算中点
                    mid_x = (segment.start_point.x + segment.end_point.x) / 2
                    mid_y = (segment.start_point.y + segment.end_point.y) / 2

                    # 计算方向
                    dx = segment.end_point.x - segment.start_point.x
                    dy = segment.end_point.y - segment.start_point.y
                    length = math.sqrt(dx * dx + dy * dy)

                    if length > 0:
                        # 归一化方向向量
                        dx, dy = dx / length, dy / length

                        # 添加箭头
                        self.ax.arrow(
                            mid_x - dx * 5,
                            mid_y - dy * 5,
                            dx * 10,
                            dy * 10,
                            head_width=5,
                            head_length=5,
                            fc=(
                                self.colors["highlight"]
                                if is_highlighted
                                else self.colors["connection"]
                            ),
                            ec=(
                                self.colors["highlight"]
                                if is_highlighted
                                else self.colors["connection"]
                            ),
                        )

    def render_dxf(
        self,
        entities: List[Entity],
        blocks: List[Block] = None,
        connections: List[Any] = None,
        focus_area: Tuple[float, float, float, float] = None,
        block_definitions: List[Block] = None,
    ):
        """
        渲染完整DXF内容

        Args:
            entities: 实体列表
            blocks: 块列表
            connections: 连接列表
            focus_area: 聚焦区域 (min_x, min_y, max_x, max_y)
        """
        # 创建新图形
        self.create_figure()

        # 打印调试信息
        if self.debug_mode:
            print("\n=== DXF内容摘要 ===")
            print(f"实体数量: {len(entities)}")
            print(f"块数量: {len(blocks) if blocks else 0}")
            print(f"连接数量: {len(connections) if connections else 0}")

            # 检查实体的边界框
            bounds_found = False
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")

            for entity in entities:
                if hasattr(entity, "bounding_box") and entity.bounding_box:
                    bounds_found = True
                    min_x = min(min_x, entity.bounding_box.min_point.x)
                    min_y = min(min_y, entity.bounding_box.min_point.y)
                    max_x = max(max_x, entity.bounding_box.max_point.x)
                    max_y = max(max_y, entity.bounding_box.max_point.y)

            if bounds_found:
                print(f"实体坐标范围: X=({min_x}, {max_x}), Y=({min_y}, {max_y})")
            else:
                print("警告: 未找到有效的实体边界")

        # 渲染基本实体
        if entities:
            print(f"正在渲染 {len(entities)} 个基本实体...")
            self.render_entities(entities)

        # 渲染块
        if blocks:
            print(f"正在渲染 {len(blocks)} 个块...")
            self.render_blocks(blocks, block_definitions=block_definitions)

        # 渲染连接
        if connections:
            print(f"正在渲染 {len(connections)} 个连接...")
            self.render_connections(connections)

        # 设置视图区域
        self._set_view_bounds(entities, blocks, focus_area)

        # # 不再添加图例
        # self._add_legend()

    def _set_view_bounds(
        self,
        entities: List[Entity],
        blocks: List[Block] = None,
        focus_area: Tuple[float, float, float, float] = None,
    ):
        """
        设置视图边界

        Args:
            entities: 实体列表
            blocks: 块列表
            focus_area: 聚焦区域 (min_x, min_y, max_x, max_y)
        """
        if focus_area:
            min_x, min_y, max_x, max_y = focus_area
        else:
            # 计算所有实体的边界
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")

            entity_count = 0
            # 考虑实体
            for entity in entities:
                if hasattr(entity, "bounding_box") and entity.bounding_box:
                    min_x = min(min_x, entity.bounding_box.min_point.x)
                    min_y = min(min_y, entity.bounding_box.min_point.y)
                    max_x = max(max_x, entity.bounding_box.max_point.x)
                    max_y = max(max_y, entity.bounding_box.max_point.y)
                    entity_count += 1
                elif hasattr(entity, "center") and hasattr(entity, "radius"):
                    # 处理圆形实体
                    min_x = min(min_x, entity.center.x - entity.radius)
                    min_y = min(min_y, entity.center.y - entity.radius)
                    max_x = max(max_x, entity.center.x + entity.radius)
                    max_y = max(max_y, entity.center.y + entity.radius)
                    entity_count += 1
                elif hasattr(entity, "start_point") and hasattr(entity, "end_point"):
                    # 处理线段实体
                    min_x = min(min_x, entity.start_point.x, entity.end_point.x)
                    min_y = min(min_y, entity.start_point.y, entity.end_point.y)
                    max_x = max(max_x, entity.start_point.x, entity.end_point.x)
                    max_y = max(max_y, entity.start_point.y, entity.end_point.y)
                    entity_count += 1
                elif hasattr(entity, "position"):
                    # 处理文本或点实体
                    min_x = min(min_x, entity.position.x)
                    min_y = min(min_y, entity.position.y)
                    max_x = max(max_x, entity.position.x)
                    max_y = max(max_y, entity.position.y)
                    entity_count += 1
                elif hasattr(entity, "vertices") and entity.vertices:
                    # 处理多段线或多边形
                    for vertex in entity.vertices:
                        min_x = min(min_x, vertex.x)
                        min_y = min(min_y, vertex.y)
                        max_x = max(max_x, vertex.x)
                        max_y = max(max_y, vertex.y)
                    entity_count += 1

            # 考虑块
            if blocks:
                for block in blocks:
                    if hasattr(block, "bounding_box") and block.bounding_box:
                        min_x = min(min_x, block.bounding_box.min_point.x)
                        min_y = min(min_y, block.bounding_box.min_point.y)
                        max_x = max(max_x, block.bounding_box.max_point.x)
                        max_y = max(max_y, block.bounding_box.max_point.y)
                        entity_count += 1
                    elif hasattr(block, "position"):
                        # 处理块引用
                        min_x = min(min_x, block.position.x)
                        min_y = min(min_y, block.position.y)
                        max_x = max(max_x, block.position.x)
                        max_y = max(max_y, block.position.y)
                        entity_count += 1
                    elif hasattr(block, "entities") and block.entities:
                        # 处理块中的实体
                        for entity in block.entities:
                            if hasattr(entity, "bounding_box") and entity.bounding_box:
                                min_x = min(min_x, entity.bounding_box.min_point.x)
                                min_y = min(min_y, entity.bounding_box.min_point.y)
                                max_x = max(max_x, entity.bounding_box.max_point.x)
                                max_y = max(max_y, entity.bounding_box.max_point.y)
                                entity_count += 1

            # 检查是否找到了有效边界
            if (
                min_x == float("inf")
                or min_y == float("inf")
                or max_x == float("-inf")
                or max_y == float("-inf")
                or entity_count == 0
            ):
                min_x, min_y = -100, -100
                max_x, max_y = 100, 100
                print(
                    "警告: 未找到有效的实体边界, 使用默认视图范围: (-100,-100) 到 (100,100)"
                )
            else:
                # 确保范围至少有一定大小
                width = max_x - min_x
                height = max_y - min_y

                # 对于很小的图形，增加一些边界
                if width < 1:
                    min_x -= 10
                    max_x += 10
                    width = 20

                if height < 1:
                    min_y -= 10
                    max_y += 10
                    height = 20

                # 调整为合适的纵横比
                aspect_ratio = width / height
                if aspect_ratio > 3:  # 太宽
                    new_height = width / 2
                    height_diff = new_height - height
                    min_y -= height_diff / 2
                    max_y += height_diff / 2
                elif aspect_ratio < 0.33:  # 太高
                    new_width = height / 2
                    width_diff = new_width - width
                    min_x -= width_diff / 2
                    max_x += width_diff / 2

                print(
                    f"设置视图范围: X=({min_x:.2f}, {max_x:.2f}), Y=({min_y:.2f}, {max_y:.2f})"
                )

        # 应用边距
        margin = self.margin
        width = max_x - min_x
        height = max_y - min_y
        # 确保边距比例与图形大小相关
        margin_x = width * 0.05 if self.margin < width * 0.05 else self.margin
        margin_y = height * 0.05 if self.margin < height * 0.05 else self.margin

        self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
        self.ax.set_ylim(min_y - margin_y, max_y + margin_y)

        # 设置正确的纵横比
        self.ax.set_aspect("equal", adjustable="box")

    def _add_legend(self):
        """添加图例"""
        legend_elements = [
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["line"],
                label="线段",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["circle"],
                label="圆",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["arc"],
                label="圆弧",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["ellipse"],
                label="椭圆",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["block"],
                label="块",
                linestyle="--",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["connection"],
                label="连接",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.colors["highlight"],
                label="高亮",
                linewidth=2.0,
            ),
        ]

        # 创建一个清晰的图例框
        legend = self.ax.legend(
            handles=legend_elements,
            loc="upper right",
            prop=chinese_font,
            framealpha=0.9,  # 增加不透明度
            edgecolor="gray",  # 添加边框
            title="图例",  # 添加标题
            title_fontproperties=chinese_font,  # 标题使用中文字体
        )

        # 设置图例标题的字体大小
        legend.get_title().set_fontsize("large")

        # 确保图例在其他元素上方
        legend.set_zorder(30)

    def _get_entity_color(self, entity: Entity) -> str:
        """
        根据实体类型获取颜色

        Args:
            entity: 实体对象

        Returns:
            str: 颜色代码
        """
        entity_type_map = {
            EntityType.LINE: "line",
            EntityType.CIRCLE: "circle",
            EntityType.ARC: "arc",
            EntityType.ELLIPSE: "ellipse",
            EntityType.TEXT: "text",
        }

        color_key = entity_type_map.get(entity.entity_type, "line")
        return self.colors.get(color_key, self.colors["line"])

    def save_image(self, filename: str, dpi: int = 300):
        """
        保存为图像文件

        Args:
            filename: 文件名
            dpi: 分辨率
        """
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            print(f"图像已保存为: {filename}")

    def show(self, block: bool = True):
        """
        显示可视化结果

        Args:
            block: 是否阻塞执行
        """
        if self.fig:
            plt.show(block=block)

    def close(self):
        """关闭图形"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _apply_coordinate_transform(self, entity, sx, sy, angle_rad, dx, dy):
        """
        对实体应用坐标变换（缩放、旋转、平移）
        
        Args:
            entity: 要变换的实体
            sx, sy: 缩放因子
            angle_rad: 旋转角度（弧度）
            dx, dy: 平移距离
        """
        # 获取实体类型以针对特殊类型应用特定变换
        etype = getattr(entity, "entity_type", None)
        
        # 根据实体类型应用特定变换
        if etype == EntityType.ARC:
            # 圆弧需要特殊处理：中心点变换 + 角度旋转
            if hasattr(entity, "center") and hasattr(entity, "radius"):
                # 中心点变换
                entity.center.x *= sx
                entity.center.y *= sy
                x0, y0 = entity.center.x, entity.center.y
                entity.center.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                entity.center.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                entity.center.x += dx
                entity.center.y += dy
                
                # 半径缩放 - 取平均缩放因子
                entity.radius *= (sx + sy) / 2.0
                
                # 起止角度旋转
                entity.start_angle = getattr(entity, "start_angle", 0.0) + math.degrees(angle_rad)
                entity.end_angle = getattr(entity, "end_angle", 0.0) + math.degrees(angle_rad)
                
            return
            
        elif etype == EntityType.ELLIPSE:
            # 椭圆需要特殊处理：中心点变换 + 主次轴变换
            if hasattr(entity, "center"):
                # 中心点变换
                entity.center.x *= sx
                entity.center.y *= sy
                x0, y0 = entity.center.x, entity.center.y
                entity.center.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                entity.center.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                entity.center.x += dx
                entity.center.y += dy
                
                # 主轴变换 (如果有)
                if hasattr(entity, "major_axis"):
                    entity.major_axis *= (sx + sy) / 2.0
                
                # 比率保持不变
                # 旋转角度调整
                if hasattr(entity, "rotation"):
                    entity.rotation = getattr(entity, "rotation", 0.0) + math.degrees(angle_rad)
                    
            return
            
        elif etype in [EntityType.TEXT, EntityType.MTEXT]:
            # 文本需要特殊处理：位置变换 + 旋转角度累加
            if hasattr(entity, "position"):
                # 位置变换
                entity.position.x *= sx
                entity.position.y *= sy
                x0, y0 = entity.position.x, entity.position.y
                entity.position.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                entity.position.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                entity.position.x += dx
                entity.position.y += dy
                
                # 旋转角度累加
                if hasattr(entity, "rotation"):
                    entity.rotation = getattr(entity, "rotation", 0.0) + math.degrees(angle_rad)
                
            return
            
        # 通用坐标点变换 - 对于其他实体类型
        for attr_name in dir(entity):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(entity, attr_name, None)
            
            # 变换点对象
            if attr and hasattr(attr, 'x') and hasattr(attr, 'y'):
                # 缩放
                attr.x *= sx
                attr.y *= sy
                
                # 旋转
                x0, y0 = attr.x, attr.y
                attr.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                attr.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                
                # 平移
                attr.x += dx
                attr.y += dy
                
            # 变换点列表（如多段线顶点）
            elif attr_name == 'vertices' and isinstance(attr, list):
                for vertex in attr:
                    if hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                        # 缩放
                        vertex.x *= sx
                        vertex.y *= sy
                        
                        # 旋转
                        x0, y0 = vertex.x, vertex.y
                        vertex.x = x0 * math.cos(angle_rad) - y0 * math.sin(angle_rad)
                        vertex.y = x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
                        
                        # 平移
                        vertex.x += dx
                        vertex.y += dy
            
            # 调整半径 (如圆)
            elif attr_name == 'radius' and isinstance(attr, (int, float)):
                entity.radius *= (sx + sy) / 2.0

    def _is_bounding_box_valid(self, bbox):
        """
        检查边界框是否有效
        
        Args:
            bbox: 边界框对象
            
        Returns:
            bool: 边界框是否有效
        """
        if not bbox:
            return False
        
        if not hasattr(bbox, "min_point") or not hasattr(bbox, "max_point"):
            return False
        
        if not bbox.min_point or not bbox.max_point:
            return False
        
        # 检查坐标是否是有效数值
        try:
            if (not isinstance(bbox.min_point.x, (int, float)) or 
                not isinstance(bbox.min_point.y, (int, float)) or
                not isinstance(bbox.max_point.x, (int, float)) or
                not isinstance(bbox.max_point.y, (int, float))):
                return False
            
            # 检查是否是无穷或NaN
            import math
            if (math.isinf(bbox.min_point.x) or math.isnan(bbox.min_point.x) or
                math.isinf(bbox.min_point.y) or math.isnan(bbox.min_point.y) or
                math.isinf(bbox.max_point.x) or math.isnan(bbox.max_point.x) or
                math.isinf(bbox.max_point.y) or math.isnan(bbox.max_point.y)):
                return False
            
            # 确保最小点小于最大点
            if bbox.min_point.x > bbox.max_point.x or bbox.min_point.y > bbox.max_point.y:
                return False
            
            return True
        except:
            return False
