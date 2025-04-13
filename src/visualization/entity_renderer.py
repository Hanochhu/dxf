"""
实体渲染器
提供各类DXF实体的渲染功能
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional, Any
from matplotlib.font_manager import FontProperties
import warnings

from src.core.data_structures import (
    Point,
    Entity,
    LineEntity,
    CircleEntity,
    ArcEntity,
    TextEntity,
    EntityType,
)

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

# 显示详细警告，但忽略字体相关警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing.*glyph.*")


class EntityRenderer:
    """
    实体渲染器类

    提供各类DXF实体的渲染方法
    """

    def __init__(self):
        """初始化实体渲染器"""
        # 渲染方法映射
        self.render_methods = {
            EntityType.LINE: self.render_line,
            EntityType.CIRCLE: self.render_circle,
            EntityType.ARC: self.render_arc,
            EntityType.ELLIPSE: self.render_ellipse,
            EntityType.TEXT: self.render_text,
            EntityType.MTEXT: self.render_text,  # MTEXT使用同样的文本渲染方法
            EntityType.POLYLINE: self.render_polyline,
            EntityType.LWPOLYLINE: self.render_polyline,  # 轻量级多段线使用相同的渲染方法
            EntityType.SPLINE: self.render_spline,
            EntityType.ARROW: self.render_arrow,  # 箭头类型
            EntityType.INSERT: self.render_insert,  # 块插入类型
            EntityType.UNKNOWN: self.render_unknown,
            EntityType.LEADER: self.render_leader,
            EntityType.SOLID: self.render_solid,
        }

    def render_entity(self, entity: Entity, ax: plt.Axes, **kwargs):
        """
        根据实体类型调用对应的渲染方法

        Args:
            entity: 实体对象
            ax: Matplotlib轴对象
            **kwargs: 传递给具体渲染方法的参数
        """
        # 获取实体类型
        entity_type = getattr(entity, "entity_type", None)

        # 检查是否有自定义的渲染方法
        render_method = None

        # 支持枚举类型
        if isinstance(entity_type, EntityType):
            render_method = self.render_methods.get(entity_type)

        # 支持字符串类型
        elif isinstance(entity_type, str):
            # 尝试将字符串转换为EntityType枚举
            try:
                entity_type = EntityType[entity_type.upper()]
                render_method = self.render_methods.get(entity_type)
            except (KeyError, ValueError):
                # 如果转换失败，尝试直接匹配字符串
                for enum_type, method in self.render_methods.items():
                    if enum_type.name.lower() == entity_type.lower():
                        render_method = method
                        break

        # 处理点类型的特殊情况
        if entity_type == "POINT" or (
            hasattr(entity_type, "value") and entity_type.value == "POINT"
        ):
            self.render_point(entity, ax, **kwargs)
        # 调用渲染方法
        elif render_method:
            render_method(entity, ax, **kwargs)
        else:
            print(f"警告: 不支持的实体类型 {entity_type}")

    def render_line(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#ff7f0e",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染直线

        Args:
            entity: 直线实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        if not hasattr(entity, "start_point") or not hasattr(entity, "end_point"):
            print(f"警告: 直线实体缺少起点或终点")
            return

        # 绘制直线
        ax.plot(
            [entity.start_point.x, entity.end_point.x],
            [entity.start_point.y, entity.end_point.y],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

    def render_circle(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#2ca02c",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染圆

        Args:
            entity: 圆实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 检查实体是否有必需属性
        if not hasattr(entity, "center") or not hasattr(entity, "radius"):
            print(f"警告: 圆实体缺少必需属性 center 或 radius")
            return

        # 创建圆形
        circle = plt.Circle(
            (entity.center.x, entity.center.y),
            entity.radius,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

        # 添加到图中
        ax.add_patch(circle)

    def render_arc(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#1f77b4",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染圆弧

        Args:
            entity: 圆弧实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        if (
            not hasattr(entity, "center")
            or not hasattr(entity, "radius")
            or not hasattr(entity, "start_angle")
            or not hasattr(entity, "end_angle")
        ):
            print(f"警告: 圆弧实体缺少必需属性")
            return

        # 计算角度（弧度转角度）
        start_angle_deg = math.degrees(entity.start_angle)
        end_angle_deg = math.degrees(entity.end_angle)

        # 确保角度是增加的
        if end_angle_deg < start_angle_deg:
            end_angle_deg += 360

        # 创建圆弧
        arc = patches.Arc(
            (entity.center.x, entity.center.y),
            2 * entity.radius,
            2 * entity.radius,
            theta1=start_angle_deg,
            theta2=end_angle_deg,
            angle=0,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

        # 添加到图中
        ax.add_patch(arc)

    def render_ellipse(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#d62728",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染椭圆

        Args:
            entity: 椭圆实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 检查实体是否有必需属性
        required_attrs = ["center", "major_radius", "minor_radius", "rotation_angle"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(entity, attr)]
        if missing_attrs:
            print(f"警告: 椭圆实体缺少必需属性: {', '.join(missing_attrs)}")
            return

        # 计算椭圆参数
        width = 2 * entity.major_radius
        height = 2 * entity.minor_radius

        # 计算旋转角度（弧度转角度）
        angle_degrees = math.degrees(entity.rotation_angle)

        # 创建椭圆
        ellipse = patches.Ellipse(
            (entity.center.x, entity.center.y),
            width,
            height,
            angle=angle_degrees,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

        # 添加到图中
        ax.add_patch(ellipse)

    def render_text(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#9467bd",
        alpha: float = 1.0,
        linewidth: float = None,
        linestyle: str = None,
        zorder: int = 10,
        **kwargs,
    ):
        """
        渲染文本

        Args:
            entity: 文本实体
            ax: Matplotlib轴对象
            color: 颜色
            alpha: 透明度
            linewidth: 线宽（忽略）
            linestyle: 线型（忽略）
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 确保实体有必需属性
        if not hasattr(entity, "position") or not hasattr(entity, "text"):
            print(f"警告: 文本实体缺少必需属性")
            return

        # 计算旋转角度（弧度转角度）
        angle_degrees = (
            math.degrees(entity.rotation) if hasattr(entity, "rotation") else 0
        )

        # 设置对齐方式
        halign = "center"
        valign = "center"

        if hasattr(entity, "alignment"):
            if entity.alignment == "LEFT":
                halign = "left"
            elif entity.alignment == "RIGHT":
                halign = "right"
            elif entity.alignment == "CENTER":
                halign = "center"

        # 获取文本内容，转换为字符串
        text_content = str(entity.text)

        # 字体大小，使用height属性或默认值
        fontsize = entity.height if hasattr(entity, "height") else 10

        # 对中文文本使用专门的中文字体
        text_obj = ax.text(
            entity.position.x,
            entity.position.y,
            text_content,
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            rotation=angle_degrees,
            horizontalalignment=halign,
            verticalalignment=valign,
            fontproperties=chinese_font,  # 使用中文字体
            zorder=zorder,
        )

        # 调整字体回退策略，避免缺失字形的警告
        text_obj.set_fontfamily(["DejaVu Sans", "SimHei", "Arial Unicode MS"])

    def render_spline(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#8c564b",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染样条曲线

        Args:
            entity: 样条曲线实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 如果有控制点，使用它们计算曲线点
        if hasattr(entity, "control_points") and entity.control_points:
            points = entity.control_points

            # 简单实现：连接控制点
            x_vals = [p.x for p in points]
            y_vals = [p.y for p in points]

            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )

            # 更精确的实现可以使用scipy.interpolate计算B样条或NURBS

    def render_polyline(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#e377c2",
        linewidth: float = 1.0,
        linestyle: str = "-",
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染多段线

        Args:
            entity: 多段线实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 获取顶点
        if hasattr(entity, "vertices") and entity.vertices:
            vertices = entity.vertices

            # 连接顶点
            x_vals = [v.x for v in vertices]
            y_vals = [v.y for v in vertices]

            # 如果是闭合多段线，添加起点
            if hasattr(entity, "is_closed") and entity.is_closed and vertices:
                x_vals.append(vertices[0].x)
                y_vals.append(vertices[0].y)

            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )

    def render_point(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#7f7f7f",
        size: float = 5.0,
        alpha: float = 1.0,
        zorder: int = 5,
        **kwargs,
    ):
        """
        渲染点

        Args:
            entity: 点实体
            ax: Matplotlib轴对象
            color: 颜色
            size: 点大小
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        if hasattr(entity, "position"):
            ax.scatter(
                entity.position.x,
                entity.position.y,
                color=color,
                s=size,
                alpha=alpha,
                zorder=zorder,
            )

    def render_hatch(
        self, entity: Any, ax: plt.Axes, color: str = "#bcbd22", alpha: float = 0.3
    ):
        """
        渲染填充

        Args:
            entity: 填充实体
            ax: Matplotlib轴对象
            color: 颜色
            alpha: 透明度
        """
        # 填充区域渲染，如果有边界数据
        if hasattr(entity, "boundaries") and entity.boundaries:
            for boundary in entity.boundaries:
                if hasattr(boundary, "vertices") and boundary.vertices:
                    vertices = boundary.vertices
                    x_vals = [v.x for v in vertices]
                    y_vals = [v.y for v in vertices]

                    # 创建多边形
                    polygon = patches.Polygon(
                        np.column_stack((x_vals, y_vals)),
                        closed=True,
                        fill=True,
                        facecolor=color,
                        alpha=alpha,
                    )

                    ax.add_patch(polygon)

    def render_arrow(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        alpha: float = 1.0,
        zorder: int = 15,
        **kwargs,
    ):
        """
        渲染箭头实体

        Args:
            entity: 箭头实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        if not hasattr(entity, "start_point") or not hasattr(entity, "end_point"):
            print(f"警告: 箭头实体缺少起点或终点")
            return

        # 绘制箭头线
        ax.plot(
            [entity.start_point.x, entity.end_point.x],
            [entity.start_point.y, entity.end_point.y],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        # 计算箭头方向
        dx = entity.end_point.x - entity.start_point.x
        dy = entity.end_point.y - entity.start_point.y
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0:
            # 添加箭头头部
            ax.arrow(
                entity.end_point.x - dx * 0.1,
                entity.end_point.y - dy * 0.1,
                dx * 0.1,
                dy * 0.1,
                head_width=length * 0.1,
                head_length=length * 0.2,
                fc=color,
                ec=color,
                alpha=alpha,
                zorder=zorder,
            )

    def render_insert(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#00ff00",
        linewidth: float = 1.0,
        alpha: float = 0.7,
        zorder: int = 10,
        **kwargs,
    ):
        """
        渲染块插入实体

        Args:
            entity: 块插入实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        if not hasattr(entity, "position"):
            print(f"警告: 块插入实体缺少位置信息")
            return

        # 绘制块插入位置标记
        ax.scatter(
            entity.position.x,
            entity.position.y,
            color=color,
            s=50,
            alpha=alpha,
            zorder=zorder,
            marker="s",  # 方形标记
        )

        # 如果有名称，添加文本标签
        if hasattr(entity, "name"):
            ax.text(
                entity.position.x,
                entity.position.y,
                entity.name,
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                alpha=alpha,
                zorder=zorder + 1,
            )

    def render_leader(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#00bfff",
        linewidth: float = 1.0,
        **kwargs,
    ):
        """渲染 LEADER（引线）实体为折线"""
        if hasattr(entity, "vertices") and entity.vertices:
            xs = [pt.x if hasattr(pt, "x") else pt[0] for pt in entity.vertices]
            ys = [pt.y if hasattr(pt, "y") else pt[1] for pt in entity.vertices]
            ax.plot(xs, ys, color=color, linewidth=linewidth, **kwargs)

    def render_solid(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#a0522d",
        alpha: float = 0.5,
        **kwargs,
    ):
        """渲染 SOLID 实体为填充四边形"""
        if hasattr(entity, "points") and len(entity.points) == 4:
            xs = [pt.x if hasattr(pt, "x") else pt[0] for pt in entity.points]
            ys = [pt.y if hasattr(pt, "y") else pt[1] for pt in entity.points]
            ax.fill(xs, ys, color=color, alpha=alpha, **kwargs)

    def render_unknown(
        self,
        entity: Any,
        ax: plt.Axes,
        color: str = "#999999",
        linewidth: float = 0.5,
        linestyle: str = ":",
        alpha: float = 0.5,
        zorder: int = 1,
        **kwargs,
    ):
        """
        渲染未知类型的实体

        Args:
            entity: 未知实体
            ax: Matplotlib轴对象
            color: 颜色
            linewidth: 线宽
            linestyle: 线型
            alpha: 透明度
            zorder: 图层顺序
            **kwargs: 其他参数
        """
        # 尝试提取实体的基本信息并渲染为点
        if hasattr(entity, "position"):
            # 如果有位置属性，渲染为点
            self.render_point(
                entity, ax, color=color, size=3.0, alpha=alpha, zorder=zorder
            )
        elif hasattr(entity, "center"):
            # 如果有中心点属性，渲染为点
            ax.scatter(
                entity.center.x,
                entity.center.y,
                color=color,
                s=3.0,
                alpha=alpha,
                zorder=zorder,
            )
        elif hasattr(entity, "start_point") and hasattr(entity, "end_point"):
            # 如果有起点和终点，渲染为虚线
            ax.plot(
                [entity.start_point.x, entity.end_point.x],
                [entity.start_point.y, entity.end_point.y],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
        elif hasattr(entity, "bounding_box") and entity.bounding_box():
            # 如果有边界框，渲染边界框
            min_point = entity.bounding_box().min_point
            max_point = entity.bounding_box().max_point
            width = max_point.x - min_point.x
            height = max_point.y - min_point.y

            rect = patches.Rectangle(
                (min_point.x, min_point.y),
                width,
                height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
            ax.add_patch(rect)
        # 如果没有任何可用信息，不渲染任何内容
        print(f"未被渲染的实体: {entity}，类型: {type(entity)}")
