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
import colorsys

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
from src.visualization.entity_style_manager import EntityStyleManager

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
        self.style_manager = EntityStyleManager()  # 使用专门的样式管理器
        self.fig = None
        self.ax = None

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
        # 将颜色设置委托给样式管理器
        self.style_manager.set_colors(color_dict)

    def create_figure(self, figsize=(12, 8), dpi=100):
        """
        创建图形

        Args:
            figsize: 图形尺寸
            dpi: 分辨率
        """
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.style_manager.colors["background"])

        # 应用网格和坐标轴设置
        if not self.show_axis:
            self.ax.set_axis_off()

        if self.show_grid:
            self.ax.grid(True, linestyle="--", color=self.style_manager.colors["grid"], alpha=0.7)

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
            self._print_entity_debug_info(entities)

        # 渲染每个实体
        for entity in entities:
            if isinstance(entity, Block):  # 跳过Block定义
                continue
                
            is_highlighted = entity.id in highlight_ids
            
            # 获取实体样式
            style = self.style_manager.get_entity_style(entity, is_highlighted)
            
            # 渲染实体
            self.entity_renderer.render_entity(entity=entity, ax=self.ax, **style)
    
    def _print_entity_debug_info(self, entities: List[Entity]):
        """打印实体调试信息"""
        print(f"正在渲染 {len(entities)} 个实体")
        entity_types = {}
        for entity in entities:
            etype = getattr(entity, "entity_type", "UNKNOWN")
            if isinstance(etype, EntityType):
                etype = etype.name
            entity_types[etype] = entity_types.get(etype, 0) + 1
        print(f"实体类型统计: {entity_types}")

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
        if self.debug_mode:
            print("[块引用检查] 传入blocks参数类型: ", type(blocks))
            if blocks:
                for i, block in enumerate(blocks[:3]):  # 只显示前三个块，避免输出过多
                    print(f"  块 {i+1}: {type(block)}, ID: {getattr(block, 'id', 'unknown')}")

        highlight_ids = highlight_ids or []
        
        # 创建随机色彩映射，为每个块分配一种颜色，避免相邻块颜色相似
        block_colors = {}
        
        # 使用golden ratio颜色生成法确保相邻块的颜色差异明显
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # 使用随机起始点
        
        # 为每个块生成唯一颜色
        for i, block in enumerate(blocks):
            if hasattr(block, "id") and block.id:
                h = (h + golden_ratio_conjugate) % 1.0
                # HSV to RGB: 饱和度和明度固定，只改变色相
                r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
                block_colors[block.id] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        # 渲染每个块
        for block in blocks:
            if not hasattr(block, "id") or not block.id:
                continue
                
            is_highlighted = block.id in highlight_ids
            
            # 获取块样式 - 使用样式管理器
            block_type = getattr(block, "entity_type", EntityType.INSERT)
            style = self.style_manager.get_entity_style(block, is_highlighted)
            
            # 如果块有自定义颜色，使用它
            if not is_highlighted and block.id in block_colors:
                style["color"] = block_colors[block.id]
            
            # 边界模式: 只渲染边界框
            if self.block_display_mode == "boundary" and hasattr(block, "bounding_box") and block.bounding_box:
                width = block.bounding_box.width
                height = block.bounding_box.height

                # 创建精确的边界框
                rect = patches.Rectangle(
                    (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                    width,
                    height,
                    linewidth=style["linewidth"],
                    edgecolor=style["color"],
                    facecolor="none",  # 无填充
                    alpha=style["alpha"],  # 完全不透明
                    zorder=style["zorder"],
                    linestyle=style["linestyle"],
                )
                self.ax.add_patch(rect)
                
                # 如果需要显示块标签
                if self.show_block_labels and hasattr(block, "name") and block.name:
                    self._add_block_label(block, style["color"])

            # 结构模式: 渲染块内部实体
            elif self.block_display_mode == "structure":
                # 寻找块定义
                block_def = None
                if block_definitions:
                    for b in block_definitions:
                        if b.name == block.name:
                            block_def = b
                            break
                
                # 没有找到块定义，或块定义没有实体
                if not block_def or not hasattr(block_def, "entities") or not block_def.entities:
                    # 只渲染边界框
                    if hasattr(block, "bounding_box") and block.bounding_box:
                        width = block.bounding_box.width
                        height = block.bounding_box.height
                        
                        rect = patches.Rectangle(
                            (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                            width,
                            height,
                            linewidth=style["linewidth"],
                            edgecolor=style["color"],
                            facecolor="none",
                            alpha=style["alpha"],
                            zorder=style["zorder"],
                        )
                        self.ax.add_patch(rect)
                    continue
                
                # 应用变换: 缩放、旋转和平移
                transformed_entities = []
                
                # 获取变换参数
                sx = getattr(block, "scale_x", 1.0)
                sy = getattr(block, "scale_y", 1.0)
                angle_deg = getattr(block, "rotation", 0.0)
                angle_rad = math.radians(angle_deg)
                
                if hasattr(block, "position") and block.position:
                    dx, dy = block.position.x, block.position.y
                else:
                    dx, dy = 0, 0
                
                # 变换每个实体
                for entity in block_def.entities:
                    # 创建深拷贝以避免修改原实体
                    import copy
                    entity_copy = copy.deepcopy(entity)
                    
                    # 应用坐标变换
                    self._apply_coordinate_transform(entity_copy, sx, sy, angle_rad, dx, dy)
                    
                    # 添加到变换后的实体列表
                    transformed_entities.append(entity_copy)
                
                # 渲染所有已变换的实体
                for ent in transformed_entities:
                    try:
                        # 获取实体的样式，但使用块的颜色
                        ent_style = self.style_manager.get_entity_style(ent, is_highlighted)
                        ent_style["color"] = style["color"]  # 使用块的颜色
                        
                        # 确保不覆盖原始实体的线型，只有在原始实体没有特定线型时才使用块的线型
                        if is_highlighted:
                            # 高亮状态使用实线
                            ent_style["linestyle"] = self.style_manager.line_styles["solid"]
                        # 否则保留原始线型
                        
                        self.entity_renderer.render_entity(
                            entity=ent,
                            ax=self.ax,
                            **ent_style
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
                        edgecolor=style["color"],
                        facecolor='none',
                        alpha=0.5,
                        zorder=10,
                        linestyle='--'
                    )
                    self.ax.add_patch(rect)
                
                # 如果需要显示块标签
                if self.show_block_labels and hasattr(block, "name") and block.name:
                    self._add_block_label(block, style["color"])
    
    def _add_block_label(self, block, color):
        """添加块标签"""
        if not hasattr(block, "bounding_box") or not block.bounding_box:
            return
            
        # 在块的中心添加标签
        center_x = (block.bounding_box.min_point.x + block.bounding_box.max_point.x) / 2
        center_y = (block.bounding_box.min_point.y + block.bounding_box.max_point.y) / 2
        
        self.ax.text(
            center_x, center_y,
            f"{block.name}",
            ha='center',
            va='center',
            fontsize=self.block_label_size,
            color=color,
            fontweight='bold',
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=color,
                alpha=0.7
            )
        )

    def render_connections(
        self, connections: List[Any], highlight_ids: List[str] = None
    ):
        """
        渲染实体之间的连接

        Args:
            connections: 连接列表
            highlight_ids: 高亮连接ID列表
        """
        if not connections:
            return

        highlight_ids = highlight_ids or []

        if self.debug_mode:
            print(f"渲染 {len(connections)} 个连接")

        # 渲染每个连接
        for conn in connections:
            if not hasattr(conn, "id") or not hasattr(conn, "source") or not hasattr(conn, "target"):
                continue

            # 判断是否高亮
            is_highlighted = conn.id in highlight_ids
            
            # 获取连接样式
            conn_style = self.style_manager.get_entity_style(conn, is_highlighted)
            conn_style["color"] = self.style_manager.colors["connection"]
            if is_highlighted:
                conn_style["color"] = self.style_manager.colors["highlight"]
            
            # 获取连接点坐标
            if hasattr(conn.source, "position") and hasattr(conn.target, "position"):
                start_x, start_y = conn.source.position.x, conn.source.position.y
                end_x, end_y = conn.target.position.x, conn.target.position.y
                
                # 绘制连接线
                self.ax.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color=conn_style["color"],
                    linewidth=conn_style["linewidth"],
                    linestyle=conn_style["linestyle"],
                    alpha=conn_style["alpha"],
                    zorder=conn_style["zorder"]
                )
                
                # 添加箭头指示方向
                if hasattr(conn, "direction") and conn.direction != "none":
                    # 计算方向向量
                    dx, dy = end_x - start_x, end_y - start_y
                    length = np.sqrt(dx * dx + dy * dy)
                    
                    if length > 0:
                        # 单位化
                        dx, dy = dx / length, dy / length
                        
                        # 确定箭头位置（根据连接方向）
                        if conn.direction == "forward" or conn.direction == "both":
                            # 在终点附近添加箭头
                            arrow_x = end_x - dx * 10  # 箭头稍微偏离终点
                            arrow_y = end_y - dy * 10
                            
                            self.ax.arrow(
                                arrow_x, arrow_y,
                                dx * 8, dy * 8,  # 箭头长度
                                head_width=5,
                                fc=conn_style["color"],
                                ec=conn_style["color"],
                            )
                            
                        if conn.direction == "backward" or conn.direction == "both":
                            # 在起点附近添加箭头
                            arrow_x = start_x + dx * 10  # 箭头稍微偏离起点
                            arrow_y = start_y + dy * 10
                            
                            self.ax.arrow(
                                arrow_x, arrow_y,
                                -dx * 8, -dy * 8,  # 反向箭头长度
                                head_width=5,
                                fc=conn_style["color"],
                                ec=conn_style["color"],
                            )
                
                # 可选: 在连接中点添加连接类型标签
                if hasattr(conn, "type") and conn.type:
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # 添加带背景的标签
                    self.ax.text(
                        mid_x, mid_y,
                        conn.type,
                        fontsize=8,
                        color=conn_style["color"],
                        ha='center',
                        va='center',
                        bbox=dict(
                            facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                            boxstyle='round,pad=0.2'
                        ),
                        zorder=10
                    )
                    
    def _get_entity_color(self, entity: Entity) -> str:
        """
        获取实体颜色 - 该方法将被弃用，使用style_manager.get_entity_style代替

        Args:
            entity: 实体对象

        Returns:
            颜色代码
        """
        # 实体类型到颜色键的映射
        entity_type_map = {
            EntityType.LINE: "line",
            EntityType.CIRCLE: "circle",
            EntityType.ARC: "arc",
            EntityType.ELLIPSE: "ellipse",
            EntityType.TEXT: "text",
            EntityType.MTEXT: "text",
            EntityType.POLYLINE: "line",
            EntityType.LWPOLYLINE: "line",
            EntityType.SPLINE: "line",
            EntityType.INSERT: "block",
        }
        
        if not hasattr(entity, "entity_type"):
            return self.style_manager.colors["unknown"]

        color_key = entity_type_map.get(entity.entity_type, "line")
        return self.style_manager.colors.get(color_key, self.style_manager.colors["line"])

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
                edgecolor=self.style_manager.colors["line"],
                label="线段",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["circle"],
                label="圆",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["arc"],
                label="圆弧",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["ellipse"],
                label="椭圆",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["block"],
                label="块",
                linestyle="--",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["connection"],
                label="连接",
                linewidth=1.5,
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=self.style_manager.colors["highlight"],
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

    def debug_linetype(self, entities=None, n_samples=3):
        """
        调试线型渲染问题
        
        Args:
            entities: 要调试的实体列表(为None时使用当前已加载的所有实体)
            n_samples: 每种类型分析的样本数量
        """
        if not self.debug_mode:
            print("需要先启用调试模式: set_debug_mode(True)")
            return
            
        print("\n=== 线型渲染调试 ===")
        
        # 如果没有提供实体列表，使用当前已加载的实体
        if entities is None:
            print("没有提供实体列表，无法进行分析")
            return
            
        # 按类型分组实体
        entity_types = {}
        for entity in entities:
            etype = getattr(entity, "entity_type", "Unknown")
            if isinstance(etype, EntityType):
                etype = etype.name
            if etype not in entity_types:
                entity_types[etype] = []
            entity_types[etype].append(entity)
        
        # 对每种类型的实体进行分析
        for etype, ents in entity_types.items():
            print(f"\n## 实体类型: {etype} (共 {len(ents)} 个)")
            # 分析样本
            for i, entity in enumerate(ents[:n_samples]):
                print(f"\n样本 {i+1}:")
                self.style_manager.debug_entity_linetype(entity)
                
                # 获取样式
                style = self.style_manager.get_entity_style(entity)
                print(f"获取的样式: {style}")
                
        print("\n=== 线型渲染调试结束 ===\n")
