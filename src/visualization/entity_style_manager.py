"""
实体样式管理器
提供DXF实体的样式管理和配置功能
"""

from typing import Dict, Any, Optional
from src.core.data_structures import EntityType


class EntityStyleManager:
    """
    实体样式管理器类
    
    负责管理:
    1. 实体颜色配置
    2. 线型和线宽设置
    3. 图层顺序(zorder)
    4. 透明度和其他视觉属性
    """
    
    def __init__(self):
        """初始化样式管理器"""
        # 基本颜色配置
        self.colors = {
            "background": "#f5f5f5",
            "grid": "#cccccc",
            "line": "#1f77b4",
            "circle": "#ff7f0e",
            "arc": "#2ca02c",
            "text": "#9467bd",
            "ellipse": "#8c564b",
            "polyline": "#e377c2",
            "spline": "#7f7f7f",
            "point": "#bcbd22",
            "insert": "#17becf",
            "block": "#e377c2",
            "connection": "#7f7f7f",
            "highlight": "#d62728",
            "unknown": "#999999",
        }
        
        # 线型配置
        self.line_styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-.",
        }
        
        # DXF线型名称到Matplotlib线型的映射
        self.dxf_linetype_map = {
            "CONTINUOUS": self.line_styles["solid"],
            "DASHED": self.line_styles["dashed"],
            "DOTTED": self.line_styles["dotted"],
            "DASHDOT": self.line_styles["dashdot"],
            "HIDDEN": self.line_styles["dashed"],
            "CENTER": self.line_styles["dashdot"],
            "PHANTOM": self.line_styles["dashdot"],
            "BORDER": self.line_styles["solid"],
            # 添加更多常见的DXF线型映射
        }
        
        # Z序列（图层顺序）
        self.zorder = {
            "hidden": 1,        # 隐藏图层、图框等
            "construction": 2,  # 辅助线、构造线
            "dimension": 3,     # 尺寸标注
            "hatch": 4,         # 填充图案
            "geometry": 5,      # 基本几何实体
            "text": 10,         # 文本、标注
            "highlight": 15,    # 高亮元素
        }
        
        # 特殊图层名称到线型的映射
        self.layer_linetype_map = {
            "HIDDEN": self.line_styles["dashed"],
            "CENTER": self.line_styles["dashdot"],
            "PHANTOM": self.line_styles["dashdot"],
            "DIMENSION": self.line_styles["solid"],
            "CONSTRUCTION": self.line_styles["dotted"],
        }
        
        # 实体类型到样式映射
        self.entity_style_map = {
            EntityType.LINE: {"color": self.colors["line"]},
            EntityType.CIRCLE: {"color": self.colors["circle"]},
            EntityType.ARC: {"color": self.colors["arc"]},
            EntityType.ELLIPSE: {"color": self.colors["ellipse"]},
            EntityType.TEXT: {"color": self.colors["text"], "zorder": self.zorder["text"]},
            EntityType.MTEXT: {"color": self.colors["text"], "zorder": self.zorder["text"]},
            EntityType.POLYLINE: {"color": self.colors["polyline"]},
            EntityType.LWPOLYLINE: {"color": self.colors["polyline"]},
            EntityType.SPLINE: {"color": self.colors["spline"]},
            EntityType.POINT: {"color": self.colors["point"]},
            EntityType.INSERT: {"color": self.colors["insert"]},
            EntityType.UNKNOWN: {"color": self.colors["unknown"], "zorder": self.zorder["hidden"]},
        }
    
    def get_entity_style(self, entity: Any, is_highlighted: bool = False) -> Dict[str, Any]:
        """
        根据实体类型和高亮状态获取样式
        
        Args:
            entity: 实体对象
            is_highlighted: 是否高亮显示
            
        Returns:
            样式参数字典
        """
        entity_type = self._get_entity_type(entity)
        
        # 获取基本样式
        base_style = self.entity_style_map.get(entity_type, {}).copy()
        if not base_style:
            # 使用默认样式
            base_style = {
                "color": self.colors["unknown"],
                "linewidth": 1.0,
                "linestyle": self.line_styles["solid"],
                "alpha": 0.8,
                "zorder": self.zorder["geometry"]
            }
        else:
            # 填充未指定的默认值
            if "linewidth" not in base_style:
                base_style["linewidth"] = 1.0
            if "linestyle" not in base_style:
                base_style["linestyle"] = self._get_entity_linestyle(entity)
            if "alpha" not in base_style:
                base_style["alpha"] = 1.0
            if "zorder" not in base_style:
                base_style["zorder"] = self.zorder["geometry"]
        
        # 应用高亮样式
        if is_highlighted:
            base_style.update({
                "color": self.colors["highlight"],
                "linewidth": 2.0,
                "zorder": self.zorder["highlight"]
            })
        
        return base_style
    
    def _get_entity_linestyle(self, entity: Any) -> str:
        """
        尝试从实体中提取线型信息
        
        Args:
            entity: 实体对象
            
        Returns:
            Matplotlib线型字符串
        """
        # 1. 首先检查是否有直接的linetype属性
        if hasattr(entity, "linetype"):
            linetype = entity.linetype
            if isinstance(linetype, str) and linetype.upper() in self.dxf_linetype_map:
                return self.dxf_linetype_map[linetype.upper()]
        
        # 2. 检查dxf子属性中的linetype
        if hasattr(entity, "dxf") and hasattr(entity.dxf, "linetype"):
            linetype = entity.dxf.linetype
            if isinstance(linetype, str) and linetype.upper() in self.dxf_linetype_map:
                return self.dxf_linetype_map[linetype.upper()]
        
        # 3. 检查图层名称是否暗示线型
        if hasattr(entity, "layer"):
            layer_name = entity.layer.upper() if isinstance(entity.layer, str) else ""
            # 检查图层名称中是否包含特定关键字
            for key, style in self.layer_linetype_map.items():
                if key in layer_name:
                    return style
            
            # 特殊处理一些常见的图层命名模式
            if "DASH" in layer_name or "HID" in layer_name:
                return self.line_styles["dashed"]
            if "DOT" in layer_name:
                return self.line_styles["dotted"]
            if "CENTER" in layer_name or "CENTRE" in layer_name:
                return self.line_styles["dashdot"]
            if "PHANTOM" in layer_name:
                return self.line_styles["dashdot"]
        
        # 4. 根据实体类型设置默认线型
        if hasattr(entity, "entity_type"):
            entity_type = entity.entity_type
            # 可以为特定实体类型设置默认线型
            if entity_type == EntityType.INSERT or entity_type == "INSERT":
                return self.line_styles["solid"]  # 块引用默认使用实线
        
        # 默认使用实线
        return self.line_styles["solid"]
    
    def set_colors(self, color_dict: Dict[str, str]) -> None:
        """
        更新颜色配置
        
        Args:
            color_dict: 颜色字典，键为实体类型，值为颜色代码
        """
        self.colors.update(color_dict)
        
        # 更新实体样式映射中的颜色
        for entity_type, style in self.entity_style_map.items():
            type_name = entity_type.name.lower() if hasattr(entity_type, 'name') else str(entity_type).lower()
            if type_name in color_dict:
                style["color"] = color_dict[type_name]
    
    def _get_entity_type(self, entity: Any) -> EntityType:
        """提取实体的类型"""
        entity_type = getattr(entity, "entity_type", None)
        
        # 支持枚举类型
        if isinstance(entity_type, EntityType):
            return entity_type
            
        # 支持字符串类型
        elif isinstance(entity_type, str):
            try:
                return EntityType[entity_type.upper()]
            except (KeyError, ValueError):
                pass
        
        return EntityType.UNKNOWN 

    def debug_entity_linetype(self, entity: Any) -> None:
        """
        打印实体线型信息，用于调试
        
        Args:
            entity: 实体对象
        """
        print(f"\n=== 实体线型调试信息 ===")
        print(f"实体类型: {getattr(entity, 'entity_type', 'Unknown')}")
        print(f"实体ID: {getattr(entity, 'id', 'Unknown')}")
        
        # 检查直接的linetype属性
        if hasattr(entity, "linetype"):
            print(f"直接linetype属性: {entity.linetype}")
            
        # 检查dxf子属性中的linetype
        if hasattr(entity, "dxf"):
            if hasattr(entity.dxf, "linetype"):
                print(f"dxf.linetype属性: {entity.dxf.linetype}")
                
        # 检查图层信息
        if hasattr(entity, "layer"):
            print(f"图层: {entity.layer}")
            
        # 最终确定的线型
        final_linestyle = self._get_entity_linestyle(entity)
        print(f"最终确定的线型: {final_linestyle}")
        print("=== 调试信息结束 ===\n") 