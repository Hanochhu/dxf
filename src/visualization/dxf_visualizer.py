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
    Point, Entity, LineEntity, CircleEntity, ArcEntity, 
    TextEntity, Block, EntityType
)
from src.visualization.entity_renderer import EntityRenderer

# 添加中文字体支持
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/PingFang.ttc'  # MacOS 中文字体
    chinese_font = FontProperties(fname=font_path)
except:
    try:
        # 尝试使用matplotlib内置的中文字体
        chinese_font = FontProperties(family='SimHei')  # 黑体
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
            'background': '#f5f5f5',
            'grid': '#cccccc',
            'line': '#1f77b4',
            'circle': '#ff7f0e',
            'arc': '#2ca02c',
            'text': '#9467bd',
            'ellipse': '#8c564b',
            'block': '#e377c2',
            'connection': '#7f7f7f',
            'highlight': '#d62728'
        }
        
        self.line_styles = {
            'solid': '-',
            'dashed': '--',
            'dotted': ':',
            'dashdot': '-.'
        }
        
        # 视图设置
        self.margin = 20  # 边距
        self.scale_factor = 1.0  # 缩放因子
        self.show_grid = True  # 是否显示网格
        self.show_axis = True  # 是否显示坐标轴
        self.interactive = True  # 是否启用交互式视图
        self.debug_mode = False  # 调试模式标志
        
        # 块显示设置
        self.show_block_labels = True  # 是否显示块标签
        self.block_label_size = 9.0  # 块标签字体大小
        self.block_display_mode = "structure"  # 块显示模式: "boundary"(边界框) 或 "structure"(内部结构)
    
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
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(self.colors['background'])
        
        # 应用网格和坐标轴设置
        if not self.show_axis:
            self.ax.set_axis_off()
        
        if self.show_grid:
            self.ax.grid(True, linestyle='--', color=self.colors['grid'], alpha=0.7)
    
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
                etype = getattr(entity, 'entity_type', 'UNKNOWN')
                if isinstance(etype, EntityType):
                    etype = etype.name
                entity_types[etype] = entity_types.get(etype, 0) + 1
            print(f"实体类型统计: {entity_types}")
        
        # 渲染每个实体
        for entity in entities:
            is_highlighted = entity.id in highlight_ids
            
            self.entity_renderer.render_entity(
                entity=entity,
                ax=self.ax,
                color=self.colors['highlight'] if is_highlighted else self._get_entity_color(entity),
                linewidth=1.5 if is_highlighted else 1.0,
                linestyle=self.line_styles['solid'],
                alpha=1.0
            )
    
    def render_blocks(self, blocks: List[Block], highlight_ids: List[str] = None):
        """
        渲染块列表
        
        Args:
            blocks: 块列表
            highlight_ids: 高亮块ID列表
        """
        if self.fig is None or self.ax is None:
            self.create_figure()
        
        highlight_ids = highlight_ids or []
        
        # 打印调试信息
        if self.debug_mode:
            print(f"正在渲染 {len(blocks)} 个块")
            for i, block in enumerate(blocks[:5]):  # 只显示前5个块
                print(f"块 {i+1}: 名称={block.name}, ID={block.id}, 实体数={len(block.entities) if hasattr(block, 'entities') else 0}")
                if hasattr(block, 'bounding_box') and block.bounding_box:
                    print(f"  边界: 最小点=({block.bounding_box.min_point.x}, {block.bounding_box.min_point.y}), "
                          f"最大点=({block.bounding_box.max_point.x}, {block.bounding_box.max_point.y})")
            
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
                block_color = self.colors['highlight']
            else:
                block_color = block_colors.get(block.id, self.colors['block'])
            
            # 边界模式: 只渲染边界框
            if self.block_display_mode == "boundary" and block.bounding_box:
                width = block.bounding_box.width
                height = block.bounding_box.height
                
                # 创建精确的边界框
                rect = patches.Rectangle(
                    (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                    width, height,
                    linewidth=1.5 if is_highlighted else 1.0,
                    edgecolor=block_color,
                    facecolor='none',  # 无填充
                    alpha=1.0,  # 完全不透明
                    zorder=10,
                    linestyle='-'
                )
                self.ax.add_patch(rect)
            
            # 结构模式: 渲染块内部实体
            elif self.block_display_mode == "structure" and hasattr(block, 'entities'):
                # 渲染块内部所有实体
                for entity in block.entities:
                    # 跳过没有边界框的实体
                    if not hasattr(entity, 'bounding_box') or not entity.bounding_box:
                        if self.debug_mode:
                            print(f"跳过无边界框的实体: {entity.id} (类型: {getattr(entity, 'entity_type', 'UNKNOWN')})")
                        continue
                        
                    try:
                        self.entity_renderer.render_entity(
                            entity=entity,
                            ax=self.ax,
                            color=block_color,
                            linewidth=1.5 if is_highlighted else 1.0,
                            linestyle='-',
                            alpha=1.0,
                            zorder=5
                        )
                    except Exception as e:
                        if self.debug_mode:
                            print(f"渲染实体 {entity.id} 时出错: {str(e)}")
                
                # 仍然显示边界框，但用虚线表示
                if block.bounding_box:
                    width = block.bounding_box.width
                    height = block.bounding_box.height
                    
                    rect = patches.Rectangle(
                        (block.bounding_box.min_point.x, block.bounding_box.min_point.y),
                        width, height,
                        linewidth=1.0,
                        edgecolor=block_color,
                        facecolor='none',
                        alpha=0.7,
                        zorder=10,
                        linestyle='--'
                    )
                    self.ax.add_patch(rect)
    
    def render_connections(self, connections: List[Any], highlight_ids: List[str] = None):
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
                    color=self.colors['highlight'] if is_highlighted else self.colors['connection'],
                    linewidth=2.0 if is_highlighted else 1.5,
                    linestyle=self.line_styles['solid'],
                    alpha=1.0
                )
            
            # 添加方向标记（如果有明确方向）
            if connection.has_explicit_direction and connection.path_segments:
                segment = connection.path_segments[0]
                if hasattr(segment, 'start_point') and hasattr(segment, 'end_point'):
                    # 计算中点
                    mid_x = (segment.start_point.x + segment.end_point.x) / 2
                    mid_y = (segment.start_point.y + segment.end_point.y) / 2
                    
                    # 计算方向
                    dx = segment.end_point.x - segment.start_point.x
                    dy = segment.end_point.y - segment.start_point.y
                    length = math.sqrt(dx*dx + dy*dy)
                    
                    if length > 0:
                        # 归一化方向向量
                        dx, dy = dx/length, dy/length
                        
                        # 添加箭头
                        self.ax.arrow(
                            mid_x - dx * 5, mid_y - dy * 5,
                            dx * 10, dy * 10,
                            head_width=5, head_length=5,
                            fc=self.colors['highlight'] if is_highlighted else self.colors['connection'],
                            ec=self.colors['highlight'] if is_highlighted else self.colors['connection']
                        )
    
    def render_dxf(self, entities: List[Entity], blocks: List[Block] = None, 
                 connections: List[Any] = None, focus_area: Tuple[float, float, float, float] = None):
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
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for entity in entities:
                if hasattr(entity, 'bounding_box') and entity.bounding_box:
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
            self.render_blocks(blocks)
        
        # 渲染连接
        if connections:
            print(f"正在渲染 {len(connections)} 个连接...")
            self.render_connections(connections)
        
        # 设置视图区域
        self._set_view_bounds(entities, blocks, focus_area)
        
        # 添加图例
        self._add_legend()
    
    def _set_view_bounds(self, entities: List[Entity], blocks: List[Block] = None, 
                        focus_area: Tuple[float, float, float, float] = None):
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
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            entity_count = 0
            # 考虑实体
            for entity in entities:
                if hasattr(entity, 'bounding_box') and entity.bounding_box:
                    min_x = min(min_x, entity.bounding_box.min_point.x)
                    min_y = min(min_y, entity.bounding_box.min_point.y)
                    max_x = max(max_x, entity.bounding_box.max_point.x)
                    max_y = max(max_y, entity.bounding_box.max_point.y)
                    entity_count += 1
                elif hasattr(entity, 'center') and hasattr(entity, 'radius'):
                    # 处理圆形实体
                    min_x = min(min_x, entity.center.x - entity.radius)
                    min_y = min(min_y, entity.center.y - entity.radius)
                    max_x = max(max_x, entity.center.x + entity.radius)
                    max_y = max(max_y, entity.center.y + entity.radius)
                    entity_count += 1
                elif hasattr(entity, 'start_point') and hasattr(entity, 'end_point'):
                    # 处理线段实体
                    min_x = min(min_x, entity.start_point.x, entity.end_point.x)
                    min_y = min(min_y, entity.start_point.y, entity.end_point.y)
                    max_x = max(max_x, entity.start_point.x, entity.end_point.x)
                    max_y = max(max_y, entity.start_point.y, entity.end_point.y)
                    entity_count += 1
                elif hasattr(entity, 'position'):
                    # 处理文本或点实体
                    min_x = min(min_x, entity.position.x)
                    min_y = min(min_y, entity.position.y)
                    max_x = max(max_x, entity.position.x)
                    max_y = max(max_y, entity.position.y)
                    entity_count += 1
                elif hasattr(entity, 'vertices') and entity.vertices:
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
                    if hasattr(block, 'bounding_box') and block.bounding_box:
                        min_x = min(min_x, block.bounding_box.min_point.x)
                        min_y = min(min_y, block.bounding_box.min_point.y)
                        max_x = max(max_x, block.bounding_box.max_point.x)
                        max_y = max(max_y, block.bounding_box.max_point.y)
                        entity_count += 1
                    elif hasattr(block, 'position'):
                        # 处理块引用
                        min_x = min(min_x, block.position.x)
                        min_y = min(min_y, block.position.y)
                        max_x = max(max_x, block.position.x)
                        max_y = max(max_y, block.position.y)
                        entity_count += 1
                    elif hasattr(block, 'entities') and block.entities:
                        # 处理块中的实体
                        for entity in block.entities:
                            if hasattr(entity, 'bounding_box') and entity.bounding_box:
                                min_x = min(min_x, entity.bounding_box.min_point.x)
                                min_y = min(min_y, entity.bounding_box.min_point.y)
                                max_x = max(max_x, entity.bounding_box.max_point.x)
                                max_y = max(max_y, entity.bounding_box.max_point.y)
                                entity_count += 1
            
            # 检查是否找到了有效边界
            if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf') or entity_count == 0:
                min_x, min_y = -100, -100
                max_x, max_y = 100, 100
                print("警告: 未找到有效的实体边界, 使用默认视图范围: (-100,-100) 到 (100,100)")
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
                
                print(f"设置视图范围: X=({min_x:.2f}, {max_x:.2f}), Y=({min_y:.2f}, {max_y:.2f})")
        
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
        self.ax.set_aspect('equal', adjustable='box')
    
    def _add_legend(self):
        """添加图例"""
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor=self.colors['line'], label='线段', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['circle'], label='圆', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['arc'], label='圆弧', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['ellipse'], label='椭圆', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['block'], label='块', linestyle='--', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['connection'], label='连接', linewidth=1.5),
            patches.Patch(facecolor='none', edgecolor=self.colors['highlight'], label='高亮', linewidth=2.0),
        ]
        
        # 创建一个清晰的图例框
        legend = self.ax.legend(
            handles=legend_elements, 
            loc='upper right', 
            prop=chinese_font,
            framealpha=0.9,  # 增加不透明度
            edgecolor='gray',  # 添加边框
            title='图例',  # 添加标题
            title_fontproperties=chinese_font  # 标题使用中文字体
        )
        
        # 设置图例标题的字体大小
        legend.get_title().set_fontsize('large')
        
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
            EntityType.LINE: 'line',
            EntityType.CIRCLE: 'circle',
            EntityType.ARC: 'arc',
            EntityType.ELLIPSE: 'ellipse',
            EntityType.TEXT: 'text'
        }
        
        color_key = entity_type_map.get(entity.entity_type, 'line')
        return self.colors.get(color_key, self.colors['line'])
    
    def save_image(self, filename: str, dpi: int = 300):
        """
        保存为图像文件
        
        Args:
            filename: 文件名
            dpi: 分辨率
        """
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
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