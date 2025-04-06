import matplotlib.pyplot as plt
from src.visualization.entity_renderer import EntityRenderer

class DXFVisualization:
    """DXF可视化类，负责DXF图纸的显示"""
    
    # 基本图层顺序
    ZORDER = {
        'hidden': 1,       # 图框、隐藏图层等
        'construction': 2, # 构造线、辅助线 
        'dimension': 3,    # 尺寸标注
        'hatch': 4,        # 填充图案
        'geometry': 5,     # 基本几何
        'text': 10,        # 文本、注释
        'highlight': 15    # 高亮元素
    }
    
    def __init__(self, dxf_processor):
        self.dxf_processor = dxf_processor
        self.renderer = EntityRenderer()
        self.fig = None
        self.ax = None
        self.entity_colors = {}  # 实体颜色映射
        self.entity_styles = {}  # 实体样式映射
        self.entity_types = {}   # 实体类型映射
        self.layer_visibility = {}  # 图层可见性
        self.layer_colors = {}   # 图层颜色映射
        
        # 初始化样式映射
        self._init_style_maps()
    
    def _init_style_maps(self):
        """初始化实体样式映射"""
        # 实体类型到颜色的映射
        self.entity_types = {
            'LINE': 'geometry',
            'CIRCLE': 'geometry',
            'ARC': 'geometry',
            'ELLIPSE': 'geometry',
            'LWPOLYLINE': 'geometry',
            'POLYLINE': 'geometry',
            'POINT': 'geometry',
            'SPLINE': 'geometry',
            'TEXT': 'text',
            'MTEXT': 'text',
            'DIMENSION': 'dimension',
            'HATCH': 'hatch',
            'INSERT': 'geometry',
            # 其他实体类型...
        }
        
        # 为不同类型实体设置默认颜色
        self.entity_colors = {
            'LINE': '#1f77b4',       # 蓝色
            'CIRCLE': '#2ca02c',     # 绿色
            'ARC': '#1f77b4',        # 蓝色
            'ELLIPSE': '#d62728',    # 红色
            'LWPOLYLINE': '#ff7f0e', # 橙色
            'POLYLINE': '#e377c2',   # 粉色
            'POINT': '#7f7f7f',      # 灰色
            'SPLINE': '#8c564b',     # 棕色
            'TEXT': '#9467bd',       # 紫色
            'MTEXT': '#9467bd',      # 紫色
            'DIMENSION': '#17becf',  # 青色
            'HATCH': '#bcbd22',      # 黄绿色
            'INSERT': '#1f77b4',     # 蓝色
            # 其他实体类型...
        }
        
        # 初始化所有图层为可见
        if self.dxf_processor.dxf_doc:
            for layer in self.dxf_processor.dxf_doc.layers:
                self.layer_visibility[layer.dxf.name] = True
                # 保存图层颜色
                self.layer_colors[layer.dxf.name] = layer.dxf.color
    
    def render_dxf(self, ax=None, figsize=(10, 8), dpi=100, 
                  layer_filter=None, entity_filter=None, highlight_entities=None,
                  title=None, grid=True, equal_aspect=True):
        """
        渲染DXF文档
        
        Args:
            ax: 可选的现有Matplotlib轴
            figsize: 图形大小
            dpi: 分辨率
            layer_filter: 图层过滤函数
            entity_filter: 实体过滤函数
            highlight_entities: 需要高亮的实体列表
            title: 图表标题
            grid: 是否显示网格
            equal_aspect: 是否保持纵横比
        
        Returns:
            Matplotlib图形对象
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.ax = ax
            self.fig = ax.figure
        
        # 设置坐标轴标签和标题
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if title:
            self.ax.set_title(title)
        
        if grid:
            self.ax.grid(True, linestyle='--', alpha=0.7)
        
        if equal_aspect:
            self.ax.set_aspect('equal')
        
        # 获取DXF中的所有实体
        entities = self.dxf_processor.get_entities()
        
        # 过滤实体
        if layer_filter:
            entities = [e for e in entities if layer_filter(e.layer)]
        
        if entity_filter:
            entities = [e for e in entities if entity_filter(e)]
        
        # 渲染所有实体
        for entity in entities:
            if not self._should_render_entity(entity):
                continue
            
            # 获取实体颜色
            entity_type = getattr(entity, 'dxftype', 'UNKNOWN')
            color = self._get_entity_color(entity)
            
            # 获取图层类型对应的zorder
            zorder = self._get_entity_zorder(entity)
            
            # 如果实体在高亮列表中，使用高亮样式
            if highlight_entities and entity in highlight_entities:
                color = '#ff0000'  # 高亮红色
                linewidth = 2.0    # 增加线宽
                zorder = self.ZORDER['highlight']  # 使用高亮zorder
            else:
                linewidth = 1.0
            
            # 渲染实体
            self.renderer.render_entity(
                entity, 
                self.ax, 
                color=color, 
                linewidth=linewidth,
                zorder=zorder
            )
        
        # 设置适当的坐标轴范围
        self._set_axes_limits()
        
        return self.fig
    
    def _get_entity_zorder(self, entity):
        """获取实体的zorder值"""
        entity_type = getattr(entity, 'dxftype', 'UNKNOWN')
        
        # 查找实体类型对应的分类
        category = self.entity_types.get(entity_type, 'geometry')
        
        # 返回对应分类的zorder值
        return self.ZORDER.get(category, self.ZORDER['geometry'])
    
    def _should_render_entity(self, entity):
        """
        检查实体是否应该被渲染
        
        Args:
            entity: DXF实体
        
        Returns:
            bool: 是否渲染
        """
        # 检查图层可见性
        if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'layer'):
            layer_name = entity.dxf.layer
            if layer_name in self.layer_visibility and not self.layer_visibility[layer_name]:
                return False
        
        return True
    
    def _get_entity_color(self, entity):
        """
        获取实体的颜色
        
        Args:
            entity: DXF实体
        
        Returns:
            str: 颜色代码
        """
        entity_type = getattr(entity, 'dxftype', 'UNKNOWN')
        
        # 如果实体有颜色属性，使用实体颜色
        if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'color') and entity.dxf.color != 256:
            # 转换DXF颜色索引为RGB
            try:
                from ezdxf.tools.rgb import dxf_color_to_rgb
                rgb = dxf_color_to_rgb(entity.dxf.color)
                return '#{:02x}{:02x}{:02x}'.format(*rgb)
            except (ImportError, ValueError):
                pass
        
        # 如果图层有颜色属性，使用图层颜色
        if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'layer'):
            layer_name = entity.dxf.layer
            if layer_name in self.layer_colors:
                layer_color = self.layer_colors[layer_name]
                if layer_color != 256:  # BYLAYER
                    try:
                        from ezdxf.tools.rgb import dxf_color_to_rgb
                        rgb = dxf_color_to_rgb(layer_color)
                        return '#{:02x}{:02x}{:02x}'.format(*rgb)
                    except (ImportError, ValueError):
                        pass
        
        # 使用实体类型的默认颜色
        return self.entity_colors.get(entity_type, '#333333')
    
    def _set_axes_limits(self):
        """设置坐标轴范围，确保所有内容可见"""
        # 获取当前轴的范围
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # 如果范围太小，设置一个默认范围
        if abs(x_max - x_min) < 1 or abs(y_max - y_min) < 1:
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
        
        # 添加一些边距
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        self.ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    def set_layer_visibility(self, layer_name, visible=True):
        """
        设置图层可见性
        
        Args:
            layer_name: 图层名称
            visible: 是否可见
        """
        self.layer_visibility[layer_name] = visible
    
    def toggle_layer_visibility(self, layer_name):
        """
        切换图层可见性
        
        Args:
            layer_name: 图层名称
        """
        if layer_name in self.layer_visibility:
            self.layer_visibility[layer_name] = not self.layer_visibility[layer_name]
    
    def get_layer_list(self):
        """
        获取所有图层名称列表
        
        Returns:
            list: 图层名称列表
        """
        if not self.dxf_processor.dxf_doc:
            return []
        
        return [layer.dxf.name for layer in self.dxf_processor.dxf_doc.layers] 