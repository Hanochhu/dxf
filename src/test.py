import os
import math
import json
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import ezdxf
from ezdxf import bbox
from ezdxf.math import Vec3, Matrix44
import pythoncom
import win32com.client

# ------------------------------
# 1. Base Data Structures
# ------------------------------

class EntityType(Enum):
    """实体类型枚举"""
    LINE = "LINE"
    CIRCLE = "CIRCLE"
    ARC = "ARC"
    POLYLINE = "POLYLINE"
    LWPOLYLINE = "LWPOLYLINE"
    TEXT = "TEXT"
    MTEXT = "MTEXT"
    INSERT = "INSERT"
    ATTRIB = "ATTRIB"
    ELLIPSE = "ELLIPSE"
    SPLINE = "SPLINE"
    ARROW = "ARROW"  # 特殊类型：箭头
    UNKNOWN = "UNKNOWN"


@dataclass
class Point:
    """三维点表示"""
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一点的欧几里得距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        tolerance = 0.001  # 浮点比较的容差
        return (abs(self.x - other.x) < tolerance and 
                abs(self.y - other.y) < tolerance and 
                abs(self.z - other.z) < tolerance)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组表示"""
        return (self.x, self.y, self.z)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float]) -> 'Point':
        """从元组创建点"""
        return cls(coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0)
    
    @classmethod
    def from_vec3(cls, vec: Vec3) -> 'Point':
        """从ezdxf的Vec3创建点"""
        return cls(vec.x, vec.y, vec.z)


@dataclass
class BoundingBox:
    """边界框表示"""
    min_point: Point
    max_point: Point
    
    @property
    def width(self) -> float:
        return self.max_point.x - self.min_point.x
    
    @property
    def height(self) -> float:
        return self.max_point.y - self.min_point.y
    
    @property
    def depth(self) -> float:
        return self.max_point.z - self.min_point.z
    
    @property
    def aspect_ratio(self) -> float:
        """计算长宽比"""
        return self.width / self.height if self.height != 0 else float('inf')
    
    @property
    def center(self) -> Point:
        """计算中心点"""
        return Point(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )
    
    def overlaps(self, other: 'BoundingBox', tolerance: float = 0.001) -> bool:
        """检查边界框是否与另一边界框重叠"""
        return not (self.min_point.x > other.max_point.x + tolerance or
                    self.max_point.x < other.min_point.x - tolerance or
                    self.min_point.y > other.max_point.y + tolerance or
                    self.max_point.y < other.min_point.y - tolerance)
    
    def contains_point(self, point: Point, tolerance: float = 0.001) -> bool:
        """检查边界框是否包含点"""
        return (self.min_point.x - tolerance <= point.x <= self.max_point.x + tolerance and
                self.min_point.y - tolerance <= point.y <= self.max_point.y + tolerance and
                self.min_point.z - tolerance <= point.z <= self.max_point.z + tolerance)
    
    def to_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """转换为元组表示"""
        return ((self.min_point.x, self.min_point.y), (self.max_point.x, self.max_point.y))
    
    @classmethod
    def from_ezdxf_bbox(cls, bbox_obj) -> 'BoundingBox':
        """从ezdxf的边界框创建"""
        return cls(
            Point(bbox_obj.extmin.x, bbox_obj.extmin.y, bbox_obj.extmin.z),
            Point(bbox_obj.extmax.x, bbox_obj.extmax.y, bbox_obj.extmax.z)
        )


@dataclass
class AttributeInfo:
    """属性信息类"""
    tag: str  # 属性标签
    value: str  # 属性值
    position: Tuple[float, float, float]  # 属性位置
    height: float  # 属性文本高度
    rotation: float  # 属性旋转角度
    layer: str  # 属性所在图层
    style: str  # 属性文本样式
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'tag': self.tag,
            'value': self.value,
            'position': self.position,
            'height': self.height,
            'rotation': self.rotation,
            'layer': self.layer,
            'style': self.style
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AttributeInfo':
        """从字典创建属性信息"""
        return cls(
            tag=data['tag'],
            value=data['value'],
            position=data['position'],
            height=data['height'],
            rotation=data['rotation'],
            layer=data['layer'],
            style=data['style']
        )


@dataclass
class Entity:
    """基础实体类"""
    id: str
    entity_type: EntityType
    layer: str
    bounding_box: Optional[BoundingBox] = None
    
    def get_feature_vector(self) -> List[float]:
        """生成实体的特征向量（用于识别）"""
        if not self.bounding_box:
            return [0, 0, 0]
        
        return [
            self.bounding_box.width,
            self.bounding_box.height,
            self.bounding_box.aspect_ratio
        ]
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = {
            'id': self.id,
            'type': self.entity_type.value,
            'layer': self.layer,
        }
        
        if self.bounding_box:
            result['bounding_box'] = {
                'min': self.bounding_box.min_point.to_tuple(),
                'max': self.bounding_box.max_point.to_tuple()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """从字典创建实体"""
        bbox = None
        if 'bounding_box' in data:
            bbox = BoundingBox(
                Point.from_tuple(data['bounding_box']['min']),
                Point.from_tuple(data['bounding_box']['max'])
            )
        
        return cls(
            id=data['id'],
            entity_type=EntityType(data['type']),
            layer=data['layer'],
            bounding_box=bbox
        )


@dataclass
class LineEntity(Entity):
    """线段实体"""
    start_point: Point
    end_point: Point
    
    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            min_x = min(self.start_point.x, self.end_point.x)
            min_y = min(self.start_point.y, self.end_point.y)
            min_z = min(self.start_point.z, self.end_point.z)
            
            max_x = max(self.start_point.x, self.end_point.x)
            max_y = max(self.start_point.y, self.end_point.y)
            max_z = max(self.start_point.z, self.end_point.z)
            
            self.bounding_box = BoundingBox(
                Point(min_x, min_y, min_z),
                Point(max_x, max_y, max_z)
            )
    
    def get_direction(self) -> Tuple[float, float, float]:
        """获取方向向量（标准化）"""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        dz = self.end_point.z - self.start_point.z
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if length == 0:
            return (0, 0, 0)
        
        return (dx/length, dy/length, dz/length)
    
    def get_length(self) -> float:
        """获取线段长度"""
        return self.start_point.distance_to(self.end_point)
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'start_point': self.start_point.to_tuple(),
            'end_point': self.end_point.to_tuple(),
            'length': self.get_length(),
            'direction': self.get_direction()
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LineEntity':
        """从字典创建线段实体"""
        base_entity = Entity.from_dict(data)
        
        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            start_point=Point.from_tuple(data['start_point']),
            end_point=Point.from_tuple(data['end_point'])
        )
    
    @classmethod
    def from_dxf_line(cls, line_entity, entity_id: str = None) -> 'LineEntity':
        """从DXF线段实体创建"""
        start = Point.from_vec3(line_entity.dxf.start)
        end = Point.from_vec3(line_entity.dxf.end)
        
        return cls(
            id=entity_id or str(line_entity.dxf.handle),
            entity_type=EntityType.LINE,
            layer=line_entity.dxf.layer,
            start_point=start,
            end_point=end
        )


@dataclass
class CircleEntity(Entity):
    """圆形实体"""
    center: Point
    radius: float
    
    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            self.bounding_box = BoundingBox(
                Point(self.center.x - self.radius, self.center.y - self.radius, self.center.z),
                Point(self.center.x + self.radius, self.center.y + self.radius, self.center.z)
            )
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'center': self.center.to_tuple(),
            'radius': self.radius
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CircleEntity':
        """从字典创建圆形实体"""
        base_entity = Entity.from_dict(data)
        
        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            center=Point.from_tuple(data['center']),
            radius=data['radius']
        )
    
    @classmethod
    def from_dxf_circle(cls, circle_entity, entity_id: str = None) -> 'CircleEntity':
        """从DXF圆形实体创建"""
        center = Point.from_vec3(circle_entity.dxf.center)
        
        return cls(
            id=entity_id or str(circle_entity.dxf.handle),
            entity_type=EntityType.CIRCLE,
            layer=circle_entity.dxf.layer,
            center=center,
            radius=circle_entity.dxf.radius
        )


@dataclass
class ArcEntity(Entity):
    """弧形实体"""
    center: Point
    radius: float
    start_angle: float
    end_angle: float
    
    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            # 简化版边界框计算，实际应考虑弧的起止角度
            self.bounding_box = BoundingBox(
                Point(self.center.x - self.radius, self.center.y - self.radius, self.center.z),
                Point(self.center.x + self.radius, self.center.y + self.radius, self.center.z)
            )
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'center': self.center.to_tuple(),
            'radius': self.radius,
            'start_angle': self.start_angle,
            'end_angle': self.end_angle
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArcEntity':
        """从字典创建弧形实体"""
        base_entity = Entity.from_dict(data)
        
        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            center=Point.from_tuple(data['center']),
            radius=data['radius'],
            start_angle=data['start_angle'],
            end_angle=data['end_angle']
        )
    
    @classmethod
    def from_dxf_arc(cls, arc_entity, entity_id: str = None) -> 'ArcEntity':
        """从DXF弧形实体创建"""
        center = Point.from_vec3(arc_entity.dxf.center)
        
        return cls(
            id=entity_id or str(arc_entity.dxf.handle),
            entity_type=EntityType.ARC,
            layer=arc_entity.dxf.layer,
            center=center,
            radius=arc_entity.dxf.radius,
            start_angle=arc_entity.dxf.start_angle,
            end_angle=arc_entity.dxf.end_angle
        )


@dataclass
class TextEntity(Entity):
    """文本实体"""
    text: str
    position: Point
    height: float
    rotation: float = 0.0
    
    def __post_init__(self):
        """初始化后计算边界框（简化版）"""
        if not self.bounding_box:
            # 文本边界框计算是粗略的估计
            text_width = len(self.text) * self.height * 0.6
            
            min_x = self.position.x
            min_y = self.position.y - self.height
            
            max_x = self.position.x + text_width
            max_y = self.position.y + self.height
            
            self.bounding_box = BoundingBox(
                Point(min_x, min_y, self.position.z),
                Point(max_x, max_y, self.position.z)
            )
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({
            'text': self.text,
            'position': self.position.to_tuple(),
            'height': self.height,
            'rotation': self.rotation
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TextEntity':
        """从字典创建文本实体"""
        base_entity = Entity.from_dict(data)
        
        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            text=data['text'],
            position=Point.from_tuple(data['position']),
            height=data['height'],
            rotation=data.get('rotation', 0.0)
        )
    
    @classmethod
    def from_dxf_text(cls, text_entity, entity_id: str = None) -> 'TextEntity':
        """从DXF文本实体创建"""
        position = Point.from_vec3(text_entity.dxf.insert)
        
        return cls(
            id=entity_id or str(text_entity.dxf.handle),
            entity_type=EntityType.TEXT if text_entity.dxftype() == 'TEXT' else EntityType.MTEXT,
            layer=text_entity.dxf.layer,
            text=text_entity.dxf.text,
            position=position,
            height=text_entity.dxf.height,
            rotation=text_entity.dxf.rotation if hasattr(text_entity.dxf, 'rotation') else 0.0
        )


@dataclass
class BlockReference:
    """块引用信息"""
    id: str
    name: str
    position: Point
    rotation: float
    scale: Tuple[float, float, float]
    attributes: List[AttributeInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'id': self.id,
            'name': self.name,
            'position': self.position.to_tuple(),
            'rotation': self.rotation,
            'scale': self.scale,
            'attributes': [attr.to_dict() for attr in self.attributes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlockReference':
        """从字典创建块引用"""
        return cls(
            id=data['id'],
            name=data['name'],
            position=Point.from_tuple(data['position']),
            rotation=data['rotation'],
            scale=data['scale'],
            attributes=[AttributeInfo.from_dict(attr) for attr in data.get('attributes', [])]
        )
    
    @classmethod
    def from_dxf_insert(cls, insert_entity) -> 'BlockReference':
        """从DXF INSERT实体创建"""
        position = Point.from_vec3(insert_entity.dxf.insert)
        
        block_ref = cls(
            id=str(insert_entity.dxf.handle),
            name=insert_entity.dxf.name,
            position=position,
            rotation=insert_entity.dxf.rotation,
            scale=(insert_entity.dxf.xscale, insert_entity.dxf.yscale, insert_entity.dxf.zscale)
        )
        
        # 处理属性
        if hasattr(insert_entity, 'attribs'):
            for attrib in insert_entity.attribs:
                if attrib.dxftype() == 'ATTRIB':
                    attrib_info = AttributeInfo(
                        tag=attrib.dxf.tag,
                        value=attrib.dxf.text,
                        position=tuple(attrib.dxf.insert),
                        height=attrib.dxf.height,
                        rotation=attrib.dxf.rotation,
                        layer=attrib.dxf.layer,
                        style=attrib.dxf.style
                    )
                    block_ref.attributes.append(attrib_info)
        
        return block_ref


@dataclass
class Block:
    """块类 - 代表一个包含多个实体的块定义或实例"""
    id: str
    name: str
    entities: List[Entity]
    bounding_box: Optional[BoundingBox] = None
    reference: Optional[BlockReference] = None
    is_arrow: bool = False
    
    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box and self.entities:
            # 计算包含所有实体的边界框
            min_points = [e.bounding_box.min_point for e in self.entities if e.bounding_box]
            max_points = [e.bounding_box.max_point for e in self.entities if e.bounding_box]
            
            if min_points and max_points:
                min_x = min(p.x for p in min_points)
                min_y = min(p.y for p in min_points)
                min_z = min(p.z for p in min_points)
                
                max_x = max(p.x for p in max_points)
                max_y = max(p.y for p in max_points)
                max_z = max(p.z for p in max_points)
                
                self.bounding_box = BoundingBox(
                    Point(min_x, min_y, min_z),
                    Point(max_x, max_y, max_z)
                )
    
    @property
    def center(self) -> Optional[Point]:
        """获取块的中心点"""
        return self.bounding_box.center if self.bounding_box else None
    
    def count_entity_types(self) -> Dict[EntityType, int]:
        """统计块中各类实体的数量"""
        counts = {etype: 0 for etype in EntityType}
        for entity in self.entities:
            counts[entity.entity_type] += 1
        return counts
    
    def get_feature_vector(self) -> List[float]:
        """生成块的特征向量（用于识别）"""
        # 特征1：边界框尺寸和长宽比
        if not self.bounding_box:
            box_features = [0, 0, 0]
        else:
            box_features = [
                self.bounding_box.width,
                self.bounding_box.height,
                self.bounding_box.aspect_ratio
            ]
        
        # 特征2：实体类型计数
        type_counts = self.count_entity_types()
        type_features = [type_counts[etype] for etype in EntityType]
        
        # 特征3：实体密度
        area = self.bounding_box.width * self.bounding_box.height if self.bounding_box else 0
        density = len(self.entities) / area if area > 0 else 0
        
        return box_features + type_features + [density]
    
    def check_is_arrow(self) -> bool:
        """检查该块是否为箭头"""
        # 1. 判断名称中是否包含"arrow"或"箭头"
        if "ARROW" in self.name.upper() or "箭头" in self.name:
            return True
        
        # 2. 检查形状特征（长宽比、线段和多段线组合）
        if self.bounding_box and self.bounding_box.aspect_ratio > 2.0:
            # 检查线段和多段线数量
            type_counts = self.count_entity_types()
            line_count = type_counts[EntityType.LINE]
            polyline_count = type_counts[EntityType.POLYLINE] + type_counts[EntityType.LWPOLYLINE]
            
            # 简单启发式：一定数量的线和多段线组合可能表示箭头
            if (line_count > 0 and polyline_count > 0) or line_count >= 2:
                # 进一步检查是否有三角形结构（箭头头部）
                return True
        
        return False
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = {
            'id': self.id,
            'name': self.name,
            'entity_count': len(self.entities),
            'entity_types': [e.entity_type.value for e in self.entities],
            'is_arrow': self.is_arrow
        }
        
        if self.bounding_box:
            result['bounding_box'] = {
                'min': self.bounding_box.min_point.to_tuple(),
                'max': self.bounding_box.max_point.to_tuple(),
                'width': self.bounding_box.width,
                'height': self.bounding_box.height,
                'aspect_ratio': self.bounding_box.aspect_ratio
            }
        
        if self.reference:
            result['reference'] = self.reference.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        """从字典创建块"""
        bbox = None
        if 'bounding_box' in data:
            bbox = BoundingBox(
                Point.from_tuple(data['bounding_box']['min']),
                Point.from_tuple(data['bounding_box']['max'])
            )
        
        reference = None
        if 'reference' in data:
            reference = BlockReference.from_dict(data['reference'])
        
        # 注意：这里不包括实体列表，需要单独处理
        return cls(
            id=data['id'],
            name=data['name'],
            entities=[],  # 实体列表需要单独处理
            bounding_box=bbox,
            reference=reference,
            is_arrow=data.get('is_arrow', False)
        )


@dataclass
class Connection:
    """连接类 - 表示两个块之间的连接"""
    id: str
    source_block: Block
    target_block: Block
    path_segments: List[LineEntity]
    has_explicit_direction: bool = False
    connection_type: str = "regular"  # "regular", "indirect", "special"
    
    @property
    def direction(self) -> Tuple[float, float, float]:
        """确定连接的整体方向"""
        if not self.path_segments:
            # 如果没有路径段，使用从源到目标的默认方向
            if not self.source_block.center or not self.target_block.center:
                return (0, 0, 0)
            
            dx = self.target_block.center.x - self.source_block.center.x
            dy = self.target_block.center.y - self.source_block.center.y
            dz = self.target_block.center.z - self.source_block.center.z
            
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            if length == 0:
                return (0, 0, 0)
            
            return (dx/length, dy/length, dz/length)
        
        # 如果有多个路径段，返回最后一段的方向
        return self.path_segments[-1].get_direction()
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'id': self.id,
            'source_block_id': self.source_block.id,
            'target_block_id': self.target_block.id,
            'path_segment_ids': [segment.id for segment in self.path_segments],
            'has_explicit_direction': self.has_explicit_direction,
            'connection_type': self.connection_type,
            'direction': self.direction
        }


@dataclass
class BlockFeature:
    """块特征类 - 用于定义识别特定类型块的特征"""
    name: str
    description: str = ""
    entity_types: Set[EntityType] = field(default_factory=set)
    min_entity_count: int = 0
    max_entity_count: int = float('inf')
    min_aspect_ratio: float = 0
    max_aspect_ratio: float = float('inf')
    min_width: float = 0
    max_width: float = float('inf')
    min_height: float = 0
    max_height: float = float('inf')
    additional_checks: List[str] = field(default_factory=list)  # 额外的检查函数名
    
    def matches(self, block: Block, tolerance: float = 0.1) -> bool:
        """检查块是否匹配此特征"""
        # 检查实体数量
        if not (self.min_entity_count <= len(block.entities) <= self.max_entity_count):
            return False
        
        # 检查实体类型
        block_entity_types = set(entity.entity_type for entity in block.entities)
        if not all(etype in block_entity_types for etype in self.entity_types):
            return False
        
        # 检查边界框特征
        if block.bounding_box:
            # 检查长宽比
            if not (self.min_aspect_ratio <= block.bounding_box.aspect_ratio <= self.max_aspect_ratio):
                return False
            
            # 检查宽度
            if not (self.min_width <= block.bounding_box.width <= self.max_width):
                return False
            
            # 检查高度
            if not (self.min_height <= block.bounding_box.height <= self.max_height):
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'name': self.name,
            'description': self.description,
            'entity_types': [etype.value for etype in self.entity_types],
            'min_entity_count': self.min_entity_count,
            'max_entity_count': self.max_entity_count,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_width': self.min_width,
            'max_width': self.max_width,
            'min_height': self.min_height,
            'max_height': self.max_height,
            'additional_checks': self.additional_checks
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlockFeature':
        """从字典创建块特征"""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            entity_types={EntityType(etype) for etype in data.get('entity_types', [])},
            min_entity_count=data.get('min_entity_count', 0),
            max_entity_count=data.get('max_entity_count', float('inf')),
            min_aspect_ratio=data.get('min_aspect_ratio', 0),
            max_aspect_ratio=data.get('max_aspect_ratio', float('inf')),
            min_width=data.get('min_width', 0),
            max_width=data.get('max_width', float('inf')),
            min_height=data.get('min_height', 0),
            max_height=data.get('max_height', float('inf')),
            additional_checks=data.get('additional_checks', [])
        )
    
    @classmethod
    def from_sample_block(cls, block: Block, name: str, description: str = "", tolerance: float = 0.2) -> 'BlockFeature':
        """从样例块创建特征模板"""
        # 计算特征范围
        entity_count = len(block.entities)
        entity_types = set(entity.entity_type for entity in block.entities)
        
        feature = cls(
            name=name,
            description=description,
            entity_types=entity_types,
            min_entity_count=max(1, int(entity_count * (1 - tolerance))),
            max_entity_count=int(entity_count * (1 + tolerance))
        )
        
        if block.bounding_box:
            # 设置长宽比范围
            aspect_ratio = block.bounding_box.aspect_ratio
            feature.min_aspect_ratio = aspect_ratio * (1 - tolerance)
            feature.max_aspect_ratio = aspect_ratio * (1 + tolerance)
            
            # 设置宽度范围
            width = block.bounding_box.width
            feature.min_width = width * (1 - tolerance)
            feature.max_width = width * (1 + tolerance)
            
            # 设置高度范围
            height = block.bounding_box.height
            feature.min_height = height * (1 - tolerance)
            feature.max_height = height * (1 + tolerance)
        
        return feature


# ------------------------------
# 2. 文件解析模块
# ------------------------------

class CADFileParser:
    """CAD文件解析器基类"""
    
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """解析CAD文件，返回实体、块和附加信息"""
        raise NotImplementedError("子类必须实现此方法")


class DXFParser(CADFileParser):
    """DXF文件解析器"""
    
    def __init__(self):
        self.doc = None
        self.msp = None
        self.blocks = {}
        self.entities = []
        self.block_instances = []
    
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """解析DXF文件"""
        try:
            self.doc = ezdxf.readfile(file_path)
            self.msp = self.doc.modelspace()
            
            # 读取块定义
            self._parse_block_definitions()
            
            # 读取模型空间实体
            self._parse_modelspace_entities()
            
            # 处理块实例
            self._process_block_instances()
            
            return self.entities, self.block_instances, {"doc": self.doc}
        
        except Exception as e:
            print(f"解析DXF文件时出错: {e}")
            return [], [], {}
    
    def _parse_block_definitions(self):
        """解析所有块定义"""
        self.blocks = {}
        
        for block in self.doc.blocks:
            block_entities = []
            
            for entity in block:
                if entity.dxftype() not in ('ATTDEF', 'SEQEND'):
                    entity_obj = self._create_entity_from_dxf(entity)
                    if entity_obj:
                        block_entities.append(entity_obj)
            
            if block_entities:
                # 创建块定义
                block_obj = Block(
                    id=str(block.name),
                    name=block.name,
                    entities=block_entities
                )
                
                # 检查是否为箭头
                block_obj.is_arrow = block_obj.check_is_arrow()
                
                self.blocks[block.name] = block_obj
    
    def _parse_modelspace_entities(self):
        """解析模型空间中的所有实体"""
        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                # INSERT实体将在下一步中处理
                continue
            
            entity_obj = self._create_entity_from_dxf(entity)
            if entity_obj:
                self.entities.append(entity_obj)
    
    def _process_block_instances(self):
        """处理模型空间中的块引用（INSERT实体）"""
        for insert in self.msp.query('INSERT'):
            block_name = insert.dxf.name
            block_def = self.blocks.get(block_name)
            
            if block_def:
                # 创建块引用
                block_ref = BlockReference.from_dxf_insert(insert)
                
                # 变换实体
                transformed_entities = self._transform_block_entities(
                    block_def.entities,
                    insert
                )
                
                # 创建块实例
                block_instance = Block(
                    id=f"{block_name}_{insert.dxf.handle}",
                    name=block_name,
                    entities=transformed_entities,
                    reference=block_ref,
                    is_arrow=block_def.is_arrow
                )
                
                self.block_instances.append(block_instance)
    
    def _create_entity_from_dxf(self, entity) -> Optional[Entity]:
        """从DXF实体创建相应的实体对象"""
        entity_type = entity.dxftype()
        
        try:
            if entity_type == 'LINE':
                return LineEntity.from_dxf_line(entity)
            elif entity_type == 'CIRCLE':
                return CircleEntity.from_dxf_circle(entity)
            elif entity_type == 'ARC':
                return ArcEntity.from_dxf_arc(entity)
            elif entity_type in ('TEXT', 'MTEXT'):
                return TextEntity.from_dxf_text(entity)
            elif entity_type in ('POLYLINE', 'LWPOLYLINE'):
                # 这里应该处理多段线，暂时简化
                entity_id = str(entity.dxf.handle)
                return Entity(
                    id=entity_id,
                    entity_type=EntityType.POLYLINE if entity_type == 'POLYLINE' else EntityType.LWPOLYLINE,
                    layer=entity.dxf.layer
                )
            elif entity_type in ('SPLINE', 'ELLIPSE'):
                # 简化处理
                entity_id = str(entity.dxf.handle)
                entity_enum_type = EntityType.SPLINE if entity_type == 'SPLINE' else EntityType.ELLIPSE
                return Entity(
                    id=entity_id,
                    entity_type=entity_enum_type,
                    layer=entity.dxf.layer
                )
            else:
                # 其他类型实体，保留基本信息
                entity_id = str(entity.dxf.handle)
                return Entity(
                    id=entity_id,
                    entity_type=EntityType.UNKNOWN,
                    layer=entity.dxf.layer
                )
        except Exception as e:
            print(f"处理实体 {entity_type} 时出错: {e}")
            return None
    
    def _transform_block_entities(self, entities: List[Entity], insert) -> List[Entity]:
        """根据INSERT参数变换块实体"""
        transformed_entities = []
        
        # 获取变换参数
        pos_x, pos_y, pos_z = insert.dxf.insert.x, insert.dxf.insert.y, insert.dxf.insert.z
        scale_x, scale_y, scale_z = insert.dxf.xscale, insert.dxf.yscale, insert.dxf.zscale
        rotation = math.radians(insert.dxf.rotation)
        
        for entity in entities:
            # 复制实体
            transformed = self._transform_entity(entity, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            if transformed:
                transformed_entities.append(transformed)
        
        return transformed_entities
    
    def _transform_entity(self, entity: Entity, pos_x: float, pos_y: float, pos_z: float, 
                         scale_x: float, scale_y: float, scale_z: float, rotation: float) -> Optional[Entity]:
        """变换单个实体"""
        if isinstance(entity, LineEntity):
            # 变换起点和终点
            start = self._transform_point(entity.start_point, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            end = self._transform_point(entity.end_point, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            
            return LineEntity(
                id=f"{entity.id}_transformed",
                entity_type=entity.entity_type,
                layer=entity.layer,
                start_point=start,
                end_point=end
            )
        elif isinstance(entity, CircleEntity):
            # 变换中心点和半径
            center = self._transform_point(entity.center, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            # 对于圆，我们取x和y缩放的平均值作为半径缩放
            radius_scale = (scale_x + scale_y) / 2
            
            return CircleEntity(
                id=f"{entity.id}_transformed",
                entity_type=entity.entity_type,
                layer=entity.layer,
                center=center,
                radius=entity.radius * radius_scale
            )
        elif isinstance(entity, ArcEntity):
            # 变换中心点、半径和角度
            center = self._transform_point(entity.center, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            radius_scale = (scale_x + scale_y) / 2
            
            # 注意：旋转会影响弧的起止角度
            start_angle = entity.start_angle + math.degrees(rotation)
            end_angle = entity.end_angle + math.degrees(rotation)
            
            return ArcEntity(
                id=f"{entity.id}_transformed",
                entity_type=entity.entity_type,
                layer=entity.layer,
                center=center,
                radius=entity.radius * radius_scale,
                start_angle=start_angle,
                end_angle=end_angle
            )
        elif isinstance(entity, TextEntity):
            # 变换位置
            position = self._transform_point(entity.position, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rotation)
            
            return TextEntity(
                id=f"{entity.id}_transformed",
                entity_type=entity.entity_type,
                layer=entity.layer,
                text=entity.text,
                position=position,
                height=entity.height * scale_y,
                rotation=entity.rotation + math.degrees(rotation)
            )
        else:
            # 其他类型实体简化处理
            return Entity(
                id=f"{entity.id}_transformed",
                entity_type=entity.entity_type,
                layer=entity.layer
            )
    
    def _transform_point(self, point: Point, pos_x: float, pos_y: float, pos_z: float, 
                        scale_x: float, scale_y: float, scale_z: float, rotation: float) -> Point:
        """变换点坐标"""
        # 缩放
        x = point.x * scale_x
        y = point.y * scale_y
        z = point.z * scale_z
        
        # 旋转
        cos_angle = math.cos(rotation)
        sin_angle = math.sin(rotation)
        
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        
        # 平移
        x_final = x_rot + pos_x
        y_final = y_rot + pos_y
        z_final = z + pos_z
        
        return Point(x_final, y_final, z_final)


class SolidWorksParser(CADFileParser):
    """SolidWorks文件解析器"""
    
    def __init__(self):
        self.sw_app = None
        self.entities = []
        self.blocks = []
    
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """解析SolidWorks文件"""
        try:
            pythoncom.CoInitialize()
            self.sw_app = win32com.client.Dispatch("SldWorks.Application")
            self.sw_app.Visible = False
            
            # 确定文件类型
            doc_type = self._get_doc_type(file_path)
            
            # 打开SolidWorks文档
            sw_doc = self.sw_app.OpenDoc(file_path, doc_type)
            
            if doc_type == 2:  # 工程图
                self._parse_drawing(sw_doc)
            
            # 关闭文档
            self.sw_app.CloseDoc(file_path)
            
            return self.entities, self.blocks, {"doc_type": doc_type}
        
        except Exception as e:
            print(f"解析SolidWorks文件时出错: {e}")
            return [], [], {}
        
        finally:
            if self.sw_app:
                self.sw_app.Visible = False
                self.sw_app = None
                pythoncom.CoUninitialize()
    
    def _get_doc_type(self, file_path: str) -> int:
        """根据文件扩展名确定SolidWorks文档类型"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".sldprt":
            return 1  # 零件
        elif ext == ".slddrw":
            return 2  # 工程图
        elif ext == ".sldasm":
            return 3  # 装配体
        else:
            raise ValueError(f"不支持的SolidWorks文件类型: {ext}")
    
    def _parse_drawing(self, drawing):
        """解析SolidWorks工程图"""
        sheet_count = drawing.GetSheetCount()
        
        for i in range(sheet_count):
            sheet = drawing.Sheet(i+1)
            view_count = sheet.GetViewCount()
            
            for j in range(view_count):
                view = sheet.GetView(j+1)
                if view:
                    # 处理视图中的实体
                    self._parse_view_entities(view)
                    
                    # 处理视图中的块
                    self._parse_view_blocks(view)
    
    def _parse_view_entities(self, view):
        """解析视图中的实体（简化版）"""
        # 此处需要根据SolidWorks API文档实现
        # 因SolidWorks API较为复杂，此处简化处理
        pass
    
    def _parse_view_blocks(self, view):
        """解析视图中的块（简化版）"""
        # 此处需要根据SolidWorks API文档实现
        # 因SolidWorks API较为复杂，此处简化处理
        pass


# ------------------------------
# 3. 特征提取和块识别
# ------------------------------

class BlockFeatureExtractor:
    """块特征提取器"""
    
    def extract_features(self, block: Block) -> np.ndarray:
        """提取块的特征向量"""
        return np.array(block.get_feature_vector())
    
    def extract_feature_dict(self, block: Block) -> Dict:
        """提取块的特征字典"""
        feature_vector = block.get_feature_vector()
        
        # 计算边界框特征
        bbox_features = {}
        if block.bounding_box:
            bbox_features = {
                'width': block.bounding_box.width,
                'height': block.bounding_box.height,
                'aspect_ratio': block.bounding_box.aspect_ratio,
                'center': (block.bounding_box.center.x, block.bounding_box.center.y)
            }
        
        # 计算实体类型分布
        type_counts = block.count_entity_types()
        entity_types = {etype.value: count for etype, count in type_counts.items() if count > 0}
        
        # 构建特征字典
        return {
            'block_name': block.name,
            'entity_count': len(block.entities),
            'entity_types': entity_types,
            'bbox': bbox_features,
            'is_arrow': block.is_arrow
        }


class BlockIdentifier:
    """块识别器"""
    
    def __init__(self):
        self.feature_extractor = BlockFeatureExtractor()
        self.block_features = {}  # 存储块特征
    
    def add_block_template(self, name: str, block: Block, tolerance: float = 0.2):
        """添加块模板供识别"""
        features = BlockFeature.from_sample_block(block, name, tolerance=tolerance)
        self.block_features[name] = features
    
    def add_block_feature(self, feature: BlockFeature):
        """添加块特征"""
        self.block_features[feature.name] = feature
    
    def identify_block(self, block: Block) -> List[Tuple[str, float]]:
        """识别块，返回匹配的特征及置信度"""
        matches = []
        
        for name, feature in self.block_features.items():
            confidence = 0.0
            
            if feature.matches(block):
                # 计算匹配置信度（简化版）
                confidence = 0.8
                
                # 进一步微调置信度
                if block.name == feature.name:
                    confidence = 0.95
                
                matches.append((name, confidence))
        
        # 按置信度排序
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def is_arrow_block(self, block: Block) -> bool:
        """检查一个块是否是箭头"""
        # 直接检查
        if block.is_arrow:
            return True
        
        # 通过特征模板检查
        matches = self.identify_block(block)
        for name, confidence in matches:
            if "ARROW" in name.upper() and confidence > 0.7:
                return True
        
        return False
    
    def save_templates(self, file_path: str):
        """保存块特征模板到文件"""
        templates = {name: feature.to_dict() for name, feature in self.block_features.items()}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=2)
    
    def load_templates(self, file_path: str):
        """从文件加载块特征模板"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            self.block_features = {}
            for name, feature_dict in templates.items():
                self.block_features[name] = BlockFeature.from_dict(feature_dict)
            
            return True
        except Exception as e:
            print(f"加载特征模板时出错: {e}")
            return False


# ------------------------------
# 4. 连接分析
# ------------------------------

class ConnectionAnalyzer:
    """连接分析器"""
    
    def __init__(self, block_identifier: BlockIdentifier):
        self.block_identifier = block_identifier
        
        # 配置参数
        self.max_gap_distance = 10.0  # 最大间隙距离
        self.connection_angle_tolerance = 0.2  # 连接角度容差
    
    def find_connections(self, blocks: List[Block], lines: List[LineEntity]) -> List[Connection]:
        """查找块之间的连接"""
        connections = []
        connection_id = 0
        
        # 首先查找直接连接
        for line in lines:
            source_block = self._find_connected_block(line.start_point, blocks)
            target_block = self._find_connected_block(line.end_point, blocks)
            
            if source_block and target_block and source_block != target_block:
                # 检查是否为已有连接的一部分
                existing_conn = self._find_existing_connection(connections, source_block, target_block)
                
                if existing_conn:
                    # 添加到现有连接
                    existing_conn.path_segments.append(line)
                else:
                    # 创建新连接
                    connection_id += 1
                    connections.append(Connection(
                        id=f"conn_{connection_id}",
                        source_block=source_block,
                        target_block=target_block,
                        path_segments=[line],
                        has_explicit_direction=False  # 后续确定
                    ))
        
        # 查找间接连接（有间隙的线段）
        indirect_connections = self._find_indirect_connections(blocks, lines, connections)
        connections.extend(indirect_connections)
        
        # 根据箭头块确定连接方向
        self._determine_connection_directions(connections, blocks)
        
        return connections
    
    def _find_connected_block(self, point: Point, blocks: List[Block]) -> Optional[Block]:
        """查找包含或非常接近点的块"""
        for block in blocks:
            if not block.bounding_box:
                continue
            
            # 检查点是否在块的边界框内
            if block.bounding_box.contains_point(point):
                return block
            
            # 检查点是否非常接近块的边界框
            tolerance = 2.0  # 小距离容差
            extended_bbox = BoundingBox(
                Point(
                    block.bounding_box.min_point.x - tolerance,
                    block.bounding_box.min_point.y - tolerance,
                    block.bounding_box.min_point.z - tolerance
                ),
                Point(
                    block.bounding_box.max_point.x + tolerance,
                    block.bounding_box.max_point.y + tolerance,
                    block.bounding_box.max_point.z + tolerance
                )
            )
            
            if extended_bbox.contains_point(point):
                return block
        
        return None
    
    def _find_existing_connection(self, connections: List[Connection], 
                                 source: Block, target: Block) -> Optional[Connection]:
        """查找源块和目标块之间的已有连接"""
        for conn in connections:
            if ((conn.source_block == source and conn.target_block == target) or
                (conn.source_block == target and conn.target_block == source and not conn.has_explicit_direction)):
                return conn
        return None
    
    def _find_indirect_connections(self, blocks: List[Block], 
                                  lines: List[LineEntity], 
                                  direct_connections: List[Connection]) -> List[Connection]:
        """查找间接连接（带间隙或通过特殊符号）"""
        indirect_connections = []
        
        # 分组未连接的线段
        connected_lines = set()
        for conn in direct_connections:
            for segment in conn.path_segments:
                connected_lines.add(segment.id)
        
        unconnected_lines = [line for line in lines if line.id not in connected_lines]
        
        # 尝试将小间隙的线段连接起来
        grouped_segments = self._group_aligned_segments(unconnected_lines)
        
        # 对于每组，尝试找到端点连接的块
        connection_id = len(direct_connections)
        for segment_group in grouped_segments:
            if len(segment_group) > 0:
                # 找到整个线段组的端点
                start_points = [segment.start_point for segment in segment_group]
                end_points = [segment.end_point for segment in segment_group]
                
                # 找到相距最远的两点
                max_distance = 0
                furthest_pair = (start_points[0], end_points[0])
                
                for start in start_points:
                    for end in end_points:
                        distance = start.distance_to(end)
                        if distance > max_distance:
                            max_distance = distance
                            furthest_pair = (start, end)
                
                # 检查这些端点是否连接到块
                source_block = self._find_connected_block(furthest_pair[0], blocks)
                target_block = self._find_connected_block(furthest_pair[1], blocks)
                
                if source_block and target_block and source_block != target_block:
                    connection_id += 1
                    indirect_connections.append(Connection(
                        id=f"conn_{connection_id}",
                        source_block=source_block,
                        target_block=target_block,
                        path_segments=segment_group,
                        has_explicit_direction=False,
                        connection_type="indirect"
                    ))
        
        return indirect_connections
    
    def _group_aligned_segments(self, lines: List[LineEntity]) -> List[List[LineEntity]]:
        """将看起来对齐或有小间隙连接的线段分组"""
        if not lines:
            return []
        
        # 创建潜在连接的线段图
        segment_graph = nx.Graph()
        
        for i, line1 in enumerate(lines):
            segment_graph.add_node(i, line=line1)
            
            for j, line2 in enumerate(lines):
                if i != j:
                    # 检查line2是否可能与line1连接
                    if self._are_segments_connected(line1, line2):
                        segment_graph.add_edge(i, j)
        
        # 查找连通分量（连接线段组）
        groups = []
        for component in nx.connected_components(segment_graph):
            group = [segment_graph.nodes[i]['line'] for i in component]
            groups.append(group)
        
        return groups
    
    def _are_segments_connected(self, line1: LineEntity, line2: LineEntity) -> bool:
        """检查两条线段是否可能连接（对齐、小间隙）"""
        # 检查端点间距离
        distances = [
            (line1.start_point.distance_to(line2.start_point), (line1.start_point, line2.start_point)),
            (line1.start_point.distance_to(line2.end_point), (line1.start_point, line2.end_point)),
            (line1.end_point.distance_to(line2.start_point), (line1.end_point, line2.start_point)),
            (line1.end_point.distance_to(line2.end_point), (line1.end_point, line2.end_point))
        ]
        
        # 找到最近的端点
        closest = min(distances, key=lambda x: x[0])
        
        # 检查它们是否足够近
        if closest[0] > self.max_gap_distance:
            return False
        
        # 检查它们是否大致对齐（方向相似）
        dir1 = line1.get_direction()
        dir2 = line2.get_direction()
        
        # 方向相似度的点积
        dot_product = (dir1[0] * dir2[0] + dir1[1] * dir2[1] + dir1[2] * dir2[2])
        
        # 如果点积接近1或-1，则它们大致对齐
        return abs(abs(dot_product) - 1.0) <= self.connection_angle_tolerance
    
    def _determine_connection_directions(self, connections: List[Connection], blocks: List[Block]):
        """根据箭头块确定连接方向"""
        # 识别所有箭头块
        arrow_blocks = [block for block in blocks if self.block_identifier.is_arrow_block(block)]
        
        for connection in connections:
            # 检查此连接上是否有箭头块
            for arrow in arrow_blocks:
                # 检查箭头是否在连接的任何线段上
                for segment in connection.path_segments:
                    if self._is_arrow_on_segment(arrow, segment):
                        # 根据箭头方向确定方向
                        arrow_dir = self._get_arrow_direction(arrow)
                        segment_dir = segment.get_direction()
                        
                        # 点积，查看它们是否指向相同方向
                        dot_product = (arrow_dir[0] * segment_dir[0] + 
                                     arrow_dir[1] * segment_dir[1] + 
                                     arrow_dir[2] * segment_dir[2])
                        
                        # 如果点积为正，它们指向相同方向
                        # 确保连接的源/目标与箭头方向匹配
                        if dot_product < 0:
                            # 如果不匹配，交换源和目标
                            connection.source_block, connection.target_block = connection.target_block, connection.source_block
                        
                        connection.has_explicit_direction = True
                        break
                
                if connection.has_explicit_direction:
                    break
        
        # 对于没有明确方向的连接，从已连接的有方向连接推断
        self._infer_connection_directions(connections)
    
    def _is_arrow_on_segment(self, arrow_block: Block, segment: LineEntity) -> bool:
        """检查箭头块是否在线段上"""
        if not arrow_block.bounding_box:
            return False
        
        # 检查箭头边界框是否与线相交
        # 首先，为线创建一个有厚度的边界框
        thickness = 1.0  # 根据需要调整
        
        line_min_x = min(segment.start_point.x, segment.end_point.x) - thickness
        line_min_y = min(segment.start_point.y, segment.end_point.y) - thickness
        line_min_z = min(segment.start_point.z, segment.end_point.z) - thickness
        
        line_max_x = max(segment.start_point.x, segment.end_point.x) + thickness
        line_max_y = max(segment.start_point.y, segment.end_point.y) + thickness
        line_max_z = max(segment.start_point.z, segment.end_point.z) + thickness
        
        line_bbox = BoundingBox(
            Point(line_min_x, line_min_y, line_min_z),
            Point(line_max_x, line_max_y, line_max_z)
        )
        
        # 检查箭头的边界框是否与线的边界框重叠
        if not arrow_block.bounding_box.overlaps(line_bbox):
            return False
        
        # 为了更精确，检查箭头的中心是否靠近线
        # 计算点到线的距离
        if not arrow_block.center:
            return False
        
        return self._point_line_distance(arrow_block.center, segment.start_point, segment.end_point) < 5.0
    
    def _point_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """计算点到线段的距离"""
        # 从线段起点到终点的向量
        line_vec = (
            line_end.x - line_start.x,
            line_end.y - line_start.y,
            line_end.z - line_start.z
        )
        
        # 从线段起点到点的向量
        point_vec = (
            point.x - line_start.x,
            point.y - line_start.y,
            point.z - line_start.z
        )
        
        # 线长度的平方
        line_length_sq = (line_vec[0]**2 + line_vec[1]**2 + line_vec[2]**2)
        
        # 边界情况：零长度线
        if line_length_sq == 0:
            return math.sqrt(point_vec[0]**2 + point_vec[1]**2 + point_vec[2]**2)
        
        # 计算投影因子
        t = max(0, min(1, (point_vec[0] * line_vec[0] + 
                        point_vec[1] * line_vec[1] + 
                        point_vec[2] * line_vec[2]) / line_length_sq))
        
        # 计算线上最近点
        closest_x = line_start.x + t * line_vec[0]
        closest_y = line_start.y + t * line_vec[1]
        closest_z = line_start.z + t * line_vec[2]
        
        # 计算距离
        return math.sqrt((point.x - closest_x)**2 + 
                       (point.y - closest_y)**2 + 
                       (point.z - closest_z)**2)
    
    def _get_arrow_direction(self, arrow_block: Block) -> Tuple[float, float, float]:
        """确定箭头块的指向方向"""
        if not arrow_block.bounding_box:
            return (0, 0, 0)
        
        # 简化方法：使用箭头的边界框长宽比确定主要方向
        aspect = arrow_block.bounding_box.aspect_ratio
        
        if aspect > 1.0:  # 宽大于高
            # 根据实体分布确定是指向左还是右
            left_count = 0
            right_count = 0
            
            midpoint_x = (arrow_block.bounding_box.min_point.x + 
                        arrow_block.bounding_box.max_point.x) / 2
            
            for entity in arrow_block.entities:
                if isinstance(entity, LineEntity):
                    # 检查起点
                    if entity.start_point.x < midpoint_x:
                        left_count += 1
                    else:
                        right_count += 1
                    
                    # 检查终点
                    if entity.end_point.x < midpoint_x:
                        left_count += 1
                    else:
                        right_count += 1
            
            # 如果右侧端点更多，可能指向右侧
            if right_count > left_count:
                return (1.0, 0.0, 0.0)  # 右
            else:
                return (-1.0, 0.0, 0.0)  # 左
        else:  # 高大于宽
            # 确定是指向上还是下
            top_count = 0
            bottom_count = 0
            
            midpoint_y = (arrow_block.bounding_box.min_point.y + 
                        arrow_block.bounding_box.max_point.y) / 2
            
            for entity in arrow_block.entities:
                if isinstance(entity, LineEntity):
                    # 检查起点
                    if entity.start_point.y < midpoint_y:
                        bottom_count += 1
                    else:
                        top_count += 1
                    
                    # 检查终点
                    if entity.end_point.y < midpoint_y:
                        bottom_count += 1
                    else:
                        top_count += 1
            
            # 如果顶部端点更多，可能指向上
            if top_count > bottom_count:
                return (0.0, 1.0, 0.0)  # 上
            else:
                return (0.0, -1.0, 0.0)  # 下
    
    def _infer_connection_directions(self, connections: List[Connection]):
        """为没有明确方向的连接推断方向"""
        # 从有明确方向的连接构建有向图
        G = nx.DiGraph()
        
        # 添加所有块作为节点
        all_blocks = set()
        for conn in connections:
            all_blocks.add(conn.source_block)
            all_blocks.add(conn.target_block)
        
        for block in all_blocks:
            G.add_node(block.id, block=block)
        
        # 添加有明确方向的连接作为边
        for conn in connections:
            if conn.has_explicit_direction:
                G.add_edge(conn.source_block.id, conn.target_block.id, connection=conn)
        
        # 对每个无方向连接，尝试推断方向
        for conn in connections:
            if not conn.has_explicit_direction:
                source_id = conn.source_block.id
                target_id = conn.target_block.id
                
                # 检查是否有从源到目标或从目标到源的路径
                try:
                    if nx.has_path(G, source_id, target_id):
                        # 保持当前方向（源到目标）
                        conn.has_explicit_direction = True
                    elif nx.has_path(G, target_id, source_id):
                        # 反转方向（目标到源）
                        conn.source_block, conn.target_block = conn.target_block, conn.source_block
                        conn.has_explicit_direction = True
                except:
                    # 处理可能的NetworkX异常
                    pass


# ------------------------------
# 5. 图构建
# ------------------------------

class CADGraph:
    """CAD图形结构类"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.blocks = {}  # 块ID到块对象的映射
        self.connections = {}  # 连接ID到连接对象的映射
    
    def build_from_blocks_connections(self, blocks: List[Block], connections: List[Connection]):
        """从块和连接构建图"""
        # 添加块作为节点
        for block in blocks:
            self.add_block(block)
        
        # 添加连接作为边
        for connection in connections:
            self.add_connection(connection)
    
    def add_block(self, block: Block):
        """将块作为节点添加到图中"""
        self.blocks[block.id] = block
        self.graph.add_node(block.id, block=block)
    
    def add_connection(self, connection: Connection):
        """将连接作为边添加到图中"""
        self.connections[connection.id] = connection
        
        # 根据连接属性添加方向
        source_id = connection.source_block.id
        target_id = connection.target_block.id
        
        self.graph.add_edge(
            source_id, 
            target_id, 
            connection=connection,
            connection_id=connection.id,
            explicit_direction=connection.has_explicit_direction,
            connection_type=connection.connection_type
        )
    
    def get_predecessors(self, block_id: str) -> List[Block]:
        """获取指向指定块的所有块"""
        predecessors = list(self.graph.predecessors(block_id))
        return [self.blocks[pred_id] for pred_id in predecessors]
    
    def get_successors(self, block_id: str) -> List[Block]:
        """获取指定块指向的所有块"""
        successors = list(self.graph.successors(block_id))
        return [self.blocks[succ_id] for succ_id in successors]
    
    def get_in_degree(self, block_id: str) -> int:
        """获取块的入度（指向该块的连接数）"""
        return self.graph.in_degree(block_id)
    
    def get_out_degree(self, block_id: str) -> int:
        """获取块的出度"""
        return self.query_interface.get_out_degree(block_id)
    
    def check_block_has_path(self, source_id: str, target_id: str) -> bool:
        """检查是否存在从源块到目标块的路径"""
        path = self.query_interface.find_path(source_id, target_id)
        return len(path) > 0
    
    def get_block_connections(self, block_id: str, direction: str = "both") -> Dict:
        """获取块的连接信息"""
        block = self.query_interface.get_block_by_id(block_id)
        if not block:
            return {"error": f"找不到块: {block_id}"}
        
        result = {
            "block_id": block_id,
            "block_name": block.name,
            "in_degree": self.get_in_degree(block_id),
            "out_degree": self.get_out_degree(block_id)
        }
        
        if direction in ["in", "both"]:
            in_connections = self.query_interface.get_in_connections(block_id)
            result["in_connections"] = [
                {
                    "from_block_id": conn.source_block.id,
                    "from_block_name": conn.source_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction
                }
                for conn in in_connections
            ]
        
        if direction in ["out", "both"]:
            out_connections = self.query_interface.get_out_connections(block_id)
            result["out_connections"] = [
                {
                    "to_block_id": conn.target_block.id,
                    "to_block_name": conn.target_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction
                }
                for conn in out_connections
            ]
        
        return result


# ------------------------------
# 8. 工具函数
# ------------------------------

def merge_dxf_entities(file_path: str, layer_name: str = None, tolerance: float = 0.1) -> List[Entity]:
    """
    合并DXF文件中的实体，可选择特定图层
    
    Args:
        file_path: DXF文件路径
        layer_name: 要处理的图层名称，如果为None则处理所有图层
        tolerance: 合并端点的容差值
        
    Returns:
        合并后的实体列表
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # 收集线段
        lines = []
        for entity in msp:
            if entity.dxftype() == 'LINE':
                if layer_name is None or entity.dxf.layer == layer_name:
                    line = LineEntity.from_dxf_line(entity)
                    lines.append(line)
        
        # 构建连接图
        G = nx.Graph()
        
        for i, line1 in enumerate(lines):
            G.add_node(i, line=line1)
            
            for j, line2 in enumerate(lines):
                if i != j:
                    # 检查线段是否连接
                    if (line1.start_point.distance_to(line2.start_point) < tolerance or
                        line1.start_point.distance_to(line2.end_point) < tolerance or
                        line1.end_point.distance_to(line2.start_point) < tolerance or
                        line1.end_point.distance_to(line2.end_point) < tolerance):
                        G.add_edge(i, j)
        
        # 查找连通分量（连接的线段集）
        merged_entities = []
        for comp in nx.connected_components(G):
            comp_lines = [lines[i] for i in comp]
            
            if len(comp_lines) == 1:
                # 单条线段不需要合并
                merged_entities.append(comp_lines[0])
            else:
                # 合并多条线段
                merged = merge_connected_lines(comp_lines, tolerance)
                merged_entities.extend(merged)
        
        return merged_entities
    
    except Exception as e:
        print(f"合并实体时出错: {e}")
        return []


def merge_connected_lines(lines: List[LineEntity], tolerance: float = 0.1) -> List[LineEntity]:
    """
    合并连接的线段
    
    Args:
        lines: 要合并的线段列表
        tolerance: 合并端点的容差值
        
    Returns:
        合并后的线段列表
    """
    if not lines:
        return []
    
    if len(lines) == 1:
        return lines
    
    # 创建点到线段的映射
    points_map = {}
    
    for line in lines:
        start_key = (round(line.start_point.x/tolerance), round(line.start_point.y/tolerance))
        end_key = (round(line.end_point.x/tolerance), round(line.end_point.y/tolerance))
        
        if start_key not in points_map:
            points_map[start_key] = []
        if end_key not in points_map:
            points_map[end_key] = []
        
        points_map[start_key].append((line, "start"))
        points_map[end_key].append((line, "end"))
    
    # 查找端点
    endpoints = []
    for point_key, connections in points_map.items():
        if len(connections) == 1:
            endpoints.append(connections[0])
    
    if not endpoints:
        # 闭合回路，选择任意点作为起点
        start_line, start_end = list(points_map.values())[0][0]
    else:
        # 选择一个端点作为起点
        start_line, start_end = endpoints[0]
    
    # 跟踪路径，合并线段
    visited = set()
    current_line = start_line
    current_end = start_end
    
    merged_lines = []
    current_path = []
    
    while current_line is not None:
        visited.add(current_line.id)
        current_path.append((current_line, current_end))
        
        # 获取当前线段的另一端点
        if current_end == "start":
            current_point = current_line.end_point
        else:
            current_point = current_line.start_point
        
        # 寻找下一条线段
        next_point_key = (round(current_point.x/tolerance), round(current_point.y/tolerance))
        next_connections = points_map.get(next_point_key, [])
        
        next_line = None
        next_end = None
        
        for line, end in next_connections:
            if line.id != current_line.id and line.id not in visited:
                next_line = line
                next_end = end
                break
        
        if next_line is None or len(current_path) > 100:  # 防止无限循环
            # 合并当前路径
            if len(current_path) == 1:
                merged_lines.append(current_path[0][0])
            else:
                # 确定起点和终点
                if current_path[0][1] == "start":
                    path_start = current_path[0][0].start_point
                else:
                    path_start = current_path[0][0].end_point
                
                if current_path[-1][1] == "start":
                    path_end = current_path[-1][0].end_point
                else:
                    path_end = current_path[-1][0].start_point
                
                # 创建合并的线段
                merged_line = LineEntity(
                    id=f"merged_{'_'.join(line.id for line, _ in current_path)}",
                    entity_type=EntityType.LINE,
                    layer=current_path[0][0].layer,
                    start_point=path_start,
                    end_point=path_end
                )
                merged_lines.append(merged_line)
            
            # 开始新的路径
            current_path = []
            
            # 寻找下一个未访问的起点
            for line in lines:
                if line.id not in visited:
                    current_line = line
                    current_end = "start"  # 默认从起点开始
                    break
            else:
                current_line = None
        else:
            current_line = next_line
            current_end = next_end
    
    return merged_lines


def detect_block_features(file_path: str, name: str = None, tolerance: float = 0.2) -> Optional[BlockFeature]:
    """
    检测和提取文件中块的特征
    
    Args:
        file_path: 包含块的DXF文件路径
        name: 块特征的名称，如果为None则使用文件名
        tolerance: 特征范围的容差
        
    Returns:
        提取的块特征，如果失败则返回None
    """
    try:
        parser = DXFParser()
        entities, blocks, _ = parser.parse_file(file_path)
        
        if not blocks:
            print("文件中没有找到块")
            return None
        
        # 使用第一个块作为模板
        block = blocks[0]
        
        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 创建特征
        feature = BlockFeature.from_sample_block(block, name, tolerance=tolerance)
        
        return feature
    
    except Exception as e:
        print(f"检测块特征时出错: {e}")
        return None


def compare_dxf_files(file1: str, file2: str, tolerance: float = 0.5) -> Dict:
    """
    比较两个DXF文件，查找相似之处和差异
    
    Args:
        file1: 第一个DXF文件路径
        file2: 第二个DXF文件路径
        tolerance: 比较的容差值
        
    Returns:
        包含比较结果的字典
    """
    try:
        # 解析两个文件
        parser = DXFParser()
        entities1, blocks1, _ = parser.parse_file(file1)
        entities2, blocks2, _ = parser.parse_file(file2)
        
        # 计算块的比较
        block_matches = []
        for block1 in blocks1:
            best_match = None
            best_similarity = 0
            
            for block2 in blocks2:
                # 计算两个块的相似度
                similarity = calculate_block_similarity(block1, block2)
                
                if similarity > tolerance and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "block1_id": block1.id,
                        "block1_name": block1.name,
                        "block2_id": block2.id,
                        "block2_name": block2.name,
                        "similarity": similarity
                    }
            
            if best_match:
                block_matches.append(best_match)
        
        return {
            "file1": file1,
            "file2": file2,
            "file1_block_count": len(blocks1),
            "file2_block_count": len(blocks2),
            "matched_blocks": block_matches,
            "match_count": len(block_matches)
        }
    
    except Exception as e:
        print(f"比较DXF文件时出错: {e}")
        return {"error": str(e)}


def calculate_block_similarity(block1: Block, block2: Block) -> float:
    """
    计算两个块之间的相似度
    
    Args:
        block1: 第一个块
        block2: 第二个块
        
    Returns:
        0到1之间的相似度值
    """
    # 比较块名称
    name_similarity = 1.0 if block1.name == block2.name else 0.0
    
    # 比较实体数量
    count1 = len(block1.entities)
    count2 = len(block2.entities)
    count_ratio = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
    
    # 比较实体类型分布
    type_counts1 = block1.count_entity_types()
    type_counts2 = block2.count_entity_types()
    
    type_similarity = 0.0
    total_types = sum(1 for t in EntityType if type_counts1[t] > 0 or type_counts2[t] > 0)
    if total_types > 0:
        matching_types = sum(1 for t in EntityType if type_counts1[t] > 0 and type_counts2[t] > 0)
        type_similarity = matching_types / total_types
    
    # 比较边界框（如果存在）
    bbox_similarity = 0.0
    if block1.bounding_box and block2.bounding_box:
        # 比较长宽比
        ar1 = block1.bounding_box.aspect_ratio
        ar2 = block2.bounding_box.aspect_ratio
        ar_ratio = min(ar1, ar2) / max(ar1, ar2) if max(ar1, ar2) > 0 else 0
        
        # 比较面积
        area1 = block1.bounding_box.width * block1.bounding_box.height
        area2 = block2.bounding_box.width * block2.bounding_box.height
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        bbox_similarity = (ar_ratio + area_ratio) / 2
    
    # 综合计算相似度
    weights = {
        "name": 0.4,
        "count": 0.2,
        "type": 0.2,
        "bbox": 0.2
    }
    
    similarity = (
        weights["name"] * name_similarity +
        weights["count"] * count_ratio +
        weights["type"] * type_similarity +
        weights["bbox"] * bbox_similarity
    )
    
    return similarity


# ------------------------------
# 9. 用法示例
# ------------------------------

def main():
    """CAD分析系统使用示例"""
    # 创建系统实例
    system = CADAnalysisSystem()
    
    # 示例1：DXF文件分析
    print("\n=== 示例1：DXF文件分析 ===")
    dxf_file = "example.dxf"
    if os.path.exists(dxf_file):
        print(f"分析DXF文件: {dxf_file}")
        success = system.analyze_file(dxf_file)
        
        if success:
            # 输出统计信息
            block_count = len(system.cad_graph.blocks)
            connection_count = len(system.cad_graph.connections)
            print(f"分析完成，发现 {block_count} 个块和 {connection_count} 个连接")
            
            # 检查块类型
            arrow_blocks = [block for block_id, block in system.cad_graph.blocks.items() if block.is_arrow]
            print(f"检测到 {len(arrow_blocks)} 个箭头块")
            
            # 查找复杂结构
            for block_id, block in system.cad_graph.blocks.items():
                in_degree = system.get_in_degree(block_id)
                out_degree = system.get_out_degree(block_id)
                
                if in_degree > 2 and out_degree > 2:
                    print(f"发现复杂节点: {block.name} (id: {block_id})，入度: {in_degree}, 出度: {out_degree}")
            
            # 保存分析结果
            system.save_analysis("analysis_results.json")
    else:
        print(f"文件 {dxf_file} 不存在")
    
    # 示例2：检测块特征
    print("\n=== 示例2：检测块特征 ===")
    block_file = "example_block.dxf"
    if os.path.exists(block_file):
        print(f"从文件提取块特征: {block_file}")
        feature = detect_block_features(block_file, "valve")
        
        if feature:
            print(f"已提取特征: {feature.name}")
            print(f"实体类型: {[t.value for t in feature.entity_types]}")
            print(f"实体数量: {feature.min_entity_count} - {feature.max_entity_count}")
            print(f"长宽比范围: {feature.min_aspect_ratio:.2f} - {feature.max_aspect_ratio:.2f}")
            
            # 添加特征到系统
            system.add_block_feature(feature)
            print("特征已添加到系统")
            
            # 在另一个文件中查找匹配
            search_file = "search_example.dxf"
            if os.path.exists(search_file):
                print(f"\n在文件 {search_file} 中查找匹配...")
                system.analyze_file(search_file)
                
                matches = system.get_blocks_by_type("valve")
                print(f"找到 {len(matches)} 个匹配的块")
                
                for i, block in enumerate(matches):
                    print(f"匹配 #{i+1}: {block.name} (id: {block.id})")
                    if block.bounding_box:
                        center = block.bounding_box.center
                        print(f"  位置: ({center.x:.2f}, {center.y:.2f})")
    else:
        print(f"文件 {block_file} 不存在")
    
    # 示例3：比较两个DXF文件
    print("\n=== 示例3：比较两个DXF文件 ===")
    file1 = "version1.dxf"
    file2 = "version2.dxf"
    
    if os.path.exists(file1) and os.path.exists(file2):
        print(f"比较文件: {file1} 和 {file2}")
        comparison = compare_dxf_files(file1, file2)
        
        print(f"文件1块数量: {comparison['file1_block_count']}")
        print(f"文件2块数量: {comparison['file2_block_count']}")
        print(f"匹配块数量: {comparison['match_count']}")
        
        if comparison['match_count'] > 0:
            print("\n匹配详情:")
            for i, match in enumerate(comparison['matched_blocks']):
                print(f"匹配 #{i+1}:")
                print(f"  文件1块: {match['block1_name']} (id: {match['block1_id']})")
                print(f"  文件2块: {match['block2_name']} (id: {match['block2_id']})")
                print(f"  相似度: {match['similarity']:.2f}")
    else:
        print(f"文件 {file1} 或 {file2} 不存在")
    
    # 示例4：查询特定块的连接
    print("\n=== 示例4：查询特定块的连接 ===")
    if len(system.cad_graph.blocks) > 0:
        block_id = next(iter(system.cad_graph.blocks.keys()))
        print(f"查询块 {block_id} 的连接:")
        
        connections = system.get_block_connections(block_id)
        print(f"块名称: {connections['block_name']}")
        print(f"入度: {connections['in_degree']}")
        print(f"出度: {connections['out_degree']}")
        
        if 'in_connections' in connections and connections['in_connections']:
            print("\n入连接:")
            for conn in connections['in_connections']:
                print(f"  来自: {conn['from_block_name']} (id: {conn['from_block_id']})")
                print(f"  连接类型: {conn['connection_type']}")
        
        if 'out_connections' in connections and connections['out_connections']:
            print("\n出连接:")
            for conn in connections['out_connections']:
                print(f"  到: {conn['to_block_name']} (id: {conn['to_block_id']})")
                print(f"  连接类型: {conn['connection_type']}")
    else:
        print("没有可用的块数据，请先分析文件")


if __name__ == "__main__":
    main()块的出度"""
        return self.query_interface.get_out_degree(block_id)
    
    def check_block_has_path(self, source_id: str, target_id: str) -> bool:
        """检查是否存在从源块到目标块的路径"""
        path = self.query_interface.find_path(source_id, target_id)
        return len(path) > 0
    
    def get_block_connections(self, block_id: str, direction: str = "both") -> Dict:
        """获取块的连接信息"""
        block = self.query_interface.get_block_by_id(block_id)
        if not block:
            return {"error": f"找不到块: {block_id}"}
        
        result = {
            "block_id": block_id,
            "block_name": block.name,
            "in_degree": self.get_in_degree(block_id),
            "out_degree": self.get_out_degree(block_id)
        }
        
        if direction in ["in", "both"]:
            in_connections = self.query_interface.get_in_connections(block_id)
            result["in_connections"] = [
                {
                    "from_block_id": conn.source_block.id,
                    "from_block_name": conn.source_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction
                }
                for conn in in_connections
            ]
        
        if direction in ["out", "both"]:
            out_connections = self.query_interface.get_out_connections(block_id)
            result["out_connections"] = [
                {
                    "to_block_id": conn.target_block.id,
                    "to_block_name": conn.target_block.name,
                    "connection_id": conn.id,
                    "connection_type": conn.connection_type,
                    "has_explicit_direction": conn.has_explicit_direction
                }
                for conn in out_connections
            ]
        
        return result


# ------------------------------
# 8. 工具函数
# ------------------------------

def merge_dxf_entities(file_path: str, layer_name: str = None, tolerance: float = 0.1) -> List[Entity]:
    """
    合并DXF文件中的实体，可选择特定图层
    
    Args:
        file_path: DXF文件路径
        layer_name: 要处理的图层名称，如果为None则处理所有图层
        tolerance: 合并端点的容差值
        
    Returns:
        合并后的实体列表
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # 收集线段
        lines = []
        for entity in msp:
            if entity.dxftype() == 'LINE':
                if layer_name is None or entity.dxf.layer == layer_name:
                    line = LineEntity.from_dxf_line(entity)
                    lines.append(line)
        
        # 构建连接图
        G = nx.Graph()
        
        for i, line1 in enumerate(lines):
            G.add_node(i, line=line1)
            
            for j, line2 in enumerate(lines):
                if i != j:
                    # 检查线段是否连接
                    if (line1.start_point.distance_to(line2.start_point) < tolerance or
                        line1.start_point.distance_to(line2.end_point) < tolerance or
                        line1.end_point.distance_to(line2.start_point) < tolerance or
                        line1.end_point.distance_to(line2.end_point) < tolerance):
                        G.add_edge(i, j)
        
        # 查找连通分量（连接的线段集）
        merged_entities = []
        for comp in nx.connected_components(G):
            comp_lines = [lines[i] for i in comp]
            
            if len(comp_lines) == 1:
                # 单条线段不需要合并
                merged_entities.append(comp_lines[0])
            else:
                # 合并多条线段
                merged = merge_connected_lines(comp_lines, tolerance)
                merged_entities.extend(merged)
        
        return merged_entities
    
    except Exception as e:
        print(f"合并实体时出错: {e}")
        return []


def merge_connected_lines(lines: List[LineEntity], tolerance: float = 0.1) -> List[LineEntity]:
    """
    合并连接的线段
    
    Args:
        lines: 要合并的线段列表
        tolerance: 合并端点的容差值
        
    Returns:
        合并后的线段列表
    """
    if not lines:
        return []
    
    if len(lines) == 1:
        return lines
    
    # 创建点到线段的映射
    points_map = {}
    
    for line in lines:
        start_key = (round(line.start_point.x/tolerance), round(line.start_point.y/tolerance))
        end_key = (round(line.end_point.x/tolerance), round(line.end_point.y/tolerance))
        
        if start_key not in points_map:
            points_map[start_key] = []
        if end_key not in points_map:
            points_map[end_key] = []
        
        points_map[start_key].append((line, "start"))
        points_map[end_key].append((line, "end"))
    
    # 查找端点
    endpoints = []
    for point_key, connections in points_map.items():
        if len(connections) == 1:
            endpoints.append(connections[0])
    
    if not endpoints:
        # 闭合回路，选择任意点作为起点
        start_line, start_end = list(points_map.values())[0][0]
    else:
        # 选择一个端点作为起点
        start_line, start_end = endpoints[0]
    
    # 跟踪路径，合并线段
    visited = set()
    current_line = start_line
    current_end = start_end
    
    merged_lines = []
    current_path = []
    
    while current_line is not None:
        visited.add(current_line.id)
        current_path.append((current_line, current_end))
        
        # 获取当前线段的另一端点
        if current_end == "start":
            current_point = current_line.end_point
        else:
            current_point = current_line.start_point
        
        # 寻找下一条线段
        next_point_key = (round(current_point.x/tolerance), round(current_point.y/tolerance))
        next_connections = points_map.get(next_point_key, [])
        
        next_line = None
        next_end = None
        
        for line, end in next_connections:
            if line.id != current_line.id and line.id not in visited:
                next_line = line
                next_end = end
                break
        
        if next_line is None or len(current_path) > 100:  # 防止无限循环
            # 合并当前路径
            if len(current_path) == 1:
                merged_lines.append(current_path[0][0])
            else:
                # 确定起点和终点
                if current_path[0][1] == "start":
                    path_start = current_path[0][0].start_point
                else:
                    path_start = current_path[0][0].end_point
                
                if current_path[-1][1] == "start":
                    path_end = current_path[-1][0].end_point
                else:
                    path_end = current_path[-1][0].start_point
                
                # 创建合并的线段
                merged_line = LineEntity(
                    id=f"merged_{'_'.join(line.id for line, _ in current_path)}",
                    entity_type=EntityType.LINE,
                    layer=current_path[0][0].layer,
                    start_point=path_start,
                    end_point=path_end
                )
                merged_lines.append(merged_line)
            
            # 开始新的路径
            current_path = []
            
            # 寻找下一个未访问的起点
            for line in lines:
                if line.id not in visited:
                    current_line = line
                    current_end = "start"  # 默认从起点开始
                    break
            else:
                current_line = None
        else:
            current_line = next_line
            current_end = next_end
    
    return merged_lines


def detect_block_features(file_path: str, name: str = None, tolerance: float = 0.2) -> Optional[BlockFeature]:
    """
    检测和提取文件中块的特征
    
    Args:
        file_path: 包含块的DXF文件路径
        name: 块特征的名称，如果为None则使用文件名
        tolerance: 特征范围的容差
        
    Returns:
        提取的块特征，如果失败则返回None
    """
    try:
        parser = DXFParser()
        entities, blocks, _ = parser.parse_file(file_path)
        
        if not blocks:
            print("文件中没有找到块")
            return None
        
        # 使用第一个块作为模板
        block = blocks[0]
        
        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 创建特征
        feature = BlockFeature.from_sample_block(block, name, tolerance=tolerance)
        
        return feature
    
    except Exception as e:
        print(f"检测块特征时出错: {e}")
        return None


def compare_dxf_files(file1: str, file2: str, tolerance: float = 0.5) -> Dict:
    """
    比较两个DXF文件，查找相似之处和差异
    
    Args:
        file1: 第一个DXF文件路径
        file2: 第二个DXF文件路径
        tolerance: 比较的容差值
        
    Returns:
        包含比较结果的字典
    """
    try:
        # 解析两个文件
        parser = DXFParser()
        entities1, blocks1, _ = parser.parse_file(file1)
        entities2, blocks2, _ = parser.parse_file(file2)
        
        # 计算块的比较
        block_matches = []
        for block1 in blocks1:
            best_match = None
            best_similarity = 0
            
            for block2 in blocks2:
                # 计算两个块的相似度
                similarity = calculate_block_similarity(block1, block2)
                
                if similarity > tolerance and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "block1_id": block1.id,
                        "block1_name": block1.name,
                        "block2_id": block2.id,
                        "block2_name": block2.name,
                        "similarity": similarity
                    }
            
            if best_match:
                block_matches.append(best_match)
        
        return {
            "file1": file1,
            "file2": file2,
            "file1_block_count": len(blocks1),
            "file2_block_count": len(blocks2),
            "matched_blocks": block_matches,
            "match_count": len(block_matches)
        }
    
    except Exception as e:
        print(f"比较DXF文件时出错: {e}")
        return {"error": str(e)}


def calculate_block_similarity(block1: Block, block2: Block) -> float:
    """
    计算两个块之间的相似度
    
    Args:
        block1: 第一个块
        block2: 第二个块
        
    Returns:
        0到1之间的相似度值
    """
    # 比较块名称
    name_similarity = 1.0 if block1.name == block2.name else 0.0
    
    # 比较实体数量
    count1 = len(block1.entities)
    count2 = len(block2.entities)
    count_ratio = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
    
    # 比较实体类型分布
    type_counts1 = block1.count_entity_types()
    type_counts2 = block2.count_entity_types()
    
    type_similarity = 0.0
    total_types = sum(1 for t in EntityType if type_counts1[t] > 0 or type_counts2[t] > 0)
    if total_types > 0:
        matching_types = sum(1 for t in EntityType if type_counts1[t] > 0 and type_counts2[t] > 0)
        type_similarity = matching_types / total_types
    
    # 比较边界框（如果存在）
    bbox_similarity = 0.0
    if block1.bounding_box and block2.bounding_box:
        # 比较长宽比
        ar1 = block1.bounding_box.aspect_ratio
        ar2 = block2.bounding_box.aspect_ratio
        ar_ratio = min(ar1, ar2) / max(ar1, ar2) if max(ar1, ar2) > 0 else 0
        
        # 比较面积
        area1 = block1.bounding_box.width * block1.bounding_box.height
        area2 = block2.bounding_box.width * block2.bounding_box.height
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        bbox_similarity = (ar_ratio + area_ratio) / 2
    
    # 综合计算相似度
    weights = {
        "name": 0.4,
        "count": 0.2,
        "type": 0.2,
        "bbox": 0.2
    }
    
    similarity = (
        weights["name"] * name_similarity +
        weights["count"] * count_ratio +
        weights["type"] * type_similarity +
        weights["bbox"] * bbox_similarity
    )
    
    return similarity


# ------------------------------
# 9. 用法示例
# ------------------------------

def main():
    """CAD分析系统使用示例"""
    # 创建系统实例
    system = CADAnalysisSystem()
    
    # 示例1：DXF文件分析
    print("\n=== 示例1：DXF文件分析 ===")
    dxf_file = "example.dxf"
    if os.path.exists(dxf_file):
        print(f"分析DXF文件: {dxf_file}")
        success = system.analyze_file(dxf_file)
        
        if success:
            # 输出统计信息
            block_count = len(system.cad_graph.blocks)
            connection_count = len(system.cad_graph.connections)
            print(f"分析完成，发现 {block_count} 个块和 {connection_count} 个连接")
            
            # 检查块类型
            arrow_blocks = [block for block_id, block in system.cad_graph.blocks.items() if block.is_arrow]
            print(f"检测到 {len(arrow_blocks)} 个箭头块")
            
            # 查找复杂结构
            for block_id, block in system.cad_graph.blocks.items():
                in_degree = system.get_in_degree(block_id)
                out_degree = system.get_out_degree(block_id)
                
                if in_degree > 2 and out_degree > 2:
                    print(f"发现复杂节点: {block.name} (id: {block_id})，入度: {in_degree}, 出度: {out_degree}")
            
            # 保存分析结果
            system.save_analysis("analysis_results.json")
    else:
        print(f"文件 {dxf_file} 不存在")
    
    # 示例2：检测块特征
    print("\n=== 示例2：检测块特征 ===")
    block_file = "example_block.dxf"
    if os.path.exists(block_file):
        print(f"从文件提取块特征: {block_file}")
        feature = detect_block_features(block_file, "valve")
        
        if feature:
            print(f"已提取特征: {feature.name}")
            print(f"实体类型: {[t.value for t in feature.entity_types]}")
            print(f"实体数量: {feature.min_entity_count} - {feature.max_entity_count}")
            print(f"长宽比范围: {feature.min_aspect_ratio:.2f} - {feature.max_aspect_ratio:.2f}")
            
            # 添加特征到系统
            system.add_block_feature(feature)
            print("特征已添加到系统")
            
            # 在另一个文件中查找匹配
            search_file = "search_example.dxf"
            if os.path.exists(search_file):
                print(f"\n在文件 {search_file} 中查找匹配...")
                system.analyze_file(search_file)
                
                matches = system.get_blocks_by_type("valve")
                print(f"找到 {len(matches)} 个匹配的块")
                
                for i, block in enumerate(matches):
                    print(f"匹配 #{i+1}: {block.name} (id: {block.id})")
                    if block.bounding_box:
                        center = block.bounding_box.center
                        print(f"  位置: ({center.x:.2f}, {center.y:.2f})")
    else:
        print(f"文件 {block_file} 不存在")
    
    # 示例3：比较两个DXF文件
    print("\n=== 示例3：比较两个DXF文件 ===")
    file1 = "version1.dxf"
    file2 = "version2.dxf"
    
    if os.path.exists(file1) and os.path.exists(file2):
        print(f"比较文件: {file1} 和 {file2}")
        comparison = compare_dxf_files(file1, file2)
        
        print(f"文件1块数量: {comparison['file1_block_count']}")
        print(f"文件2块数量: {comparison['file2_block_count']}")
        print(f"匹配块数量: {comparison['match_count']}")
        
        if comparison['match_count'] > 0:
            print("\n匹配详情:")
            for i, match in enumerate(comparison['matched_blocks']):
                print(f"匹配 #{i+1}:")
                print(f"  文件1块: {match['block1_name']} (id: {match['block1_id']})")
                print(f"  文件2块: {match['block2_name']} (id: {match['block2_id']})")
                print(f"  相似度: {match['similarity']:.2f}")
    else:
        print(f"文件 {file1} 或 {file2} 不存在")
    
    # 示例4：查询特定块的连接
    print("\n=== 示例4：查询特定块的连接 ===")
    if len(system.cad_graph.blocks) > 0:
        block_id = next(iter(system.cad_graph.blocks.keys()))
        print(f"查询块 {block_id} 的连接:")
        
        connections = system.get_block_connections(block_id)
        print(f"块名称: {connections['block_name']}")
        print(f"入度: {connections['in_degree']}")
        print(f"出度: {connections['out_degree']}")
        
        if 'in_connections' in connections and connections['in_connections']:
            print("\n入连接:")
            for conn in connections['in_connections']:
                print(f"  来自: {conn['from_block_name']} (id: {conn['from_block_id']})")
                print(f"  连接类型: {conn['connection_type']}")
        
        if 'out_connections' in connections and connections['out_connections']:
            print("\n出连接:")
            for conn in connections['out_connections']:
                print(f"  到: {conn['to_block_name']} (id: {conn['to_block_id']})")
                print(f"  连接类型: {conn['connection_type']}")
    else:
        print("没有可用的块数据，请先分析文件")


if __name__ == "__main__":
    main()块的出度（从该块出发的连接数）"""
        return self.graph.out_degree(block_id)
    
    def has_connection(self, source_id: str, target_id: str) -> bool:
        """检查源块和目标块之间是否有直接连接"""
        return self.graph.has_edge(source_id, target_id)
    
    def get_path(self, source_id: str, target_id: str) -> List[str]:
        """查找从源块到目标块的路径"""
        if not nx.has_path(self.graph, source_id, target_id):
            return []
        
        return nx.shortest_path(self.graph, source_id, target_id)
    
    def to_networkx(self) -> nx.DiGraph:
        """获取底层NetworkX图"""
        return self.graph
    
    def export_to_json(self, file_path: str):
        """将图结构导出为JSON"""
        data = {
            'blocks': {block_id: block.to_dict() for block_id, block in self.blocks.items()},
            'connections': {conn_id: conn.to_dict() for conn_id, conn in self.connections.items()}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, file_path: str) -> bool:
        """从JSON导入图结构"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清空当前图
            self.graph = nx.DiGraph()
            self.blocks = {}
            self.connections = {}
            
            # 首先加载所有块
            for block_id, block_data in data['blocks'].items():
                block = Block.from_dict(block_data)
                self.add_block(block)
            
            # 然后加载连接
            for conn_id, conn_data in data['connections'].items():
                source_id = conn_data['source_block_id']
                target_id = conn_data['target_block_id']
                
                if source_id in self.blocks and target_id in self.blocks:
                    connection = Connection(
                        id=conn_id,
                        source_block=self.blocks[source_id],
                        target_block=self.blocks[target_id],
                        path_segments=[],  # 简化处理
                        has_explicit_direction=conn_data['has_explicit_direction'],
                        connection_type=conn_data['connection_type']
                    )
                    self.add_connection(connection)
            
            return True
        
        except Exception as e:
            print(f"导入JSON时出错: {e}")
            return False


# ------------------------------
# 6. 查询接口
# ------------------------------

class CADQueryInterface:
    """CAD图形查询接口"""
    
    def __init__(self, cad_graph: CADGraph, block_identifier: BlockIdentifier):
        self.graph = cad_graph
        self.block_identifier = block_identifier
    
    def get_block_by_id(self, block_id: str) -> Optional[Block]:
        """根据ID获取块"""
        return self.graph.blocks.get(block_id)
    
    def get_blocks_by_name(self, name: str) -> List[Block]:
        """获取指定名称的所有块"""
        return [block for block_id, block in self.graph.blocks.items() 
                if name.lower() in block.name.lower()]
    
    def get_blocks_by_type(self, block_type: str, threshold: float = 0.7) -> List[Block]:
        """根据特征匹配获取指定类型的所有块"""
        matches = []
        
        for block_id, block in self.graph.blocks.items():
            block_matches = self.block_identifier.identify_block(block)
            for match_type, similarity in block_matches:
                if match_type == block_type and similarity >= threshold:
                    matches.append(block)
                    break
        
        return matches
    
    def has_in_connection_from(self, target_id: str, source_id: str) -> bool:
        """检查目标块是否有来自源块的入连接"""
        return self.graph.has_connection(source_id, target_id)
    
    def has_out_connection_to(self, source_id: str, target_id: str) -> bool:
        """检查源块是否有指向目标块的出连接"""
        return self.graph.has_connection(source_id, target_id)
    
    def get_in_connections(self, block_id: str) -> List[Connection]:
        """获取块的所有入连接"""
        connections = []
        for pred_id in self.graph.graph.predecessors(block_id):
            edge_data = self.graph.graph.get_edge_data(pred_id, block_id)
            if edge_data and 'connection_id' in edge_data:
                conn_id = edge_data['connection_id']
                connections.append(self.graph.connections[conn_id])
        
        return connections
    
    def get_out_connections(self, block_id: str) -> List[Connection]:
        """获取块的所有出连接"""
        connections = []
        for succ_id in self.graph.graph.successors(block_id):
            edge_data = self.graph.graph.get_edge_data(block_id, succ_id)
            if edge_data and 'connection_id' in edge_data:
                conn_id = edge_data['connection_id']
                connections.append(self.graph.connections[conn_id])
        
        return connections
    
    def get_connected_blocks_of_type(self, block_id: str, block_type: str, 
                                    direction: str = "both") -> List[Block]:
        """获取与指定块连接的指定类型的所有块"""
        connected_blocks = []
        
        # 检查入连接
        if direction in ["in", "both"]:
            for pred_id in self.graph.graph.predecessors(block_id):
                pred_block = self.graph.blocks[pred_id]
                matches = self.block_identifier.identify_block(pred_block)
                for match_type, similarity in matches:
                    if match_type == block_type and similarity >= 0.7:
                        connected_blocks.append(pred_block)
                        break
        
        # 检查出连接
        if direction in ["out", "both"]:
            for succ_id in self.graph.graph.successors(block_id):
                succ_block = self.graph.blocks[succ_id]
                matches = self.block_identifier.identify_block(succ_block)
                for match_type, similarity in matches:
                    if match_type == block_type and similarity >= 0.7:
                        connected_blocks.append(succ_block)
                        break
        
        return connected_blocks
    
    def get_in_degree(self, block_id: str) -> int:
        """获取入度（入连接数）"""
        return self.graph.get_in_degree(block_id)
    
    def get_out_degree(self, block_id: str) -> int:
        """获取出度（出连接数）"""
        return self.graph.get_out_degree(block_id)
    
    def find_path(self, source_id: str, target_id: str) -> List[Block]:
        """查找从源块到目标块的路径"""
        path_ids = self.graph.get_path(source_id, target_id)
        return [self.graph.blocks[block_id] for block_id in path_ids]
    
    def find_all_paths(self, source_id: str, target_id: str, 
                      cutoff: int = None) -> List[List[Block]]:
        """查找从源到目标的所有路径"""
        all_paths = []
        try:
            path_lists = list(nx.all_simple_paths(
                self.graph.graph, source_id, target_id, cutoff=cutoff))
            
            for path in path_lists:
                block_path = [self.graph.blocks[block_id] for block_id in path]
                all_paths.append(block_path)
        except nx.NetworkXNoPath:
            pass
        
        return all_paths
    
    def query(self, query_str: str) -> Dict:
        """自然语言查询接口（简化版）"""
        result = {"success": False, "message": "", "data": None}
        
        try:
            # 这是一个简化的解释-实际实现应使用NLP
            parts = query_str.strip().lower().split()
            
            if "indegree" in query_str or "in degree" in query_str or "incoming" in query_str:
                # 关于入连接的查询
                block_id = self._extract_block_id(query_str)
                if block_id:
                    in_degree = self.get_in_degree(block_id)
                    result = {
                        "success": True,
                        "message": f"块 {block_id} 有 {in_degree} 个入连接",
                        "data": in_degree
                    }
            
            elif "outdegree" in query_str or "out degree" in query_str or "outgoing" in query_str:
                # 关于出连接的查询
                block_id = self._extract_block_id(query_str)
                if block_id:
                    out_degree = self.get_out_degree(block_id)
                    result = {
                        "success": True,
                        "message": f"块 {block_id} 有 {out_degree} 个出连接",
                        "data": out_degree
                    }
            
            elif "has connection" in query_str or "is connected" in query_str:
                # 关于特定连接的查询
                source_id = self._extract_source_id(query_str)
                target_id = self._extract_target_id(query_str)
                
                if source_id and target_id:
                    has_connection = self.has_out_connection_to(source_id, target_id)
                    result = {
                        "success": True,
                        "message": f"从 {source_id} 到 {target_id} 的连接: {has_connection}",
                        "data": has_connection
                    }
        
        except Exception as e:
            result["message"] = f"处理查询时出错: {e}"
        
        return result
    
    def _extract_block_id(self, query_str: str) -> Optional[str]:
        """从查询字符串中提取块ID（简化版）"""
        words = query_str.split()
        for i, word in enumerate(words):
            if word == "block" and i+1 < len(words):
                return words[i+1]
        return None
    
    def _extract_source_id(self, query_str: str) -> Optional[str]:
        """从查询字符串中提取源块ID（简化版）"""
        words = query_str.split()
        for i, word in enumerate(words):
            if word == "from" and i+1 < len(words):
                return words[i+1]
        return None
    
    def _extract_target_id(self, query_str: str) -> Optional[str]:
        """从查询字符串中提取目标块ID（简化版）"""
        words = query_str.split()
        for i, word in enumerate(words):
            if word == "to" and i+1 < len(words):
                return words[i+1]
        return None


# ------------------------------
# 7. 主系统类
# ------------------------------

class CADAnalysisSystem:
    """CAD分析系统主类"""
    
    def __init__(self):
        # 初始化核心组件
        self.block_identifier = BlockIdentifier()
        self.connection_analyzer = ConnectionAnalyzer(self.block_identifier)
        self.cad_graph = CADGraph()
        self.query_interface = CADQueryInterface(self.cad_graph, self.block_identifier)
        
        # 初始化解析器
        self.parsers = {
            '.dxf': DXFParser(),
            '.slddrw': SolidWorksParser(),
            '.sldprt': SolidWorksParser(),
            '.sldasm': SolidWorksParser()
        }
    
    def analyze_file(self, file_path: str) -> bool:
        """分析CAD文件并构建内部表示"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.parsers:
            print(f"不支持的文件类型: {file_ext}")
            return False
        
        try:
            # 解析文件
            parser = self.parsers[file_ext]
            entities, blocks, extra_info = parser.parse_file(file_path)
            
            # 提取线段用于连接分析
            lines = [entity for entity in entities if isinstance(entity, LineEntity)]
            
            # 查找块之间的连接
            connections = self.connection_analyzer.find_connections(blocks, lines)
            
            # 构建图
            self.cad_graph.build_from_blocks_connections(blocks, connections)
            
            return True
        
        except Exception as e:
            print(f"分析文件时出错: {e}")
            return False
    
    def load_block_templates(self, template_file: str) -> bool:
        """从文件加载块模板"""
        return self.block_identifier.load_templates(template_file)
    
    def add_block_template(self, name: str, template_file: str) -> bool:
        """从模板文件添加块模板"""
        try:
            file_ext = os.path.splitext(template_file)[1].lower()
            
            if file_ext not in self.parsers:
                print(f"不支持的文件类型: {file_ext}")
                return False
            
            parser = self.parsers[file_ext]
            entities, blocks, _ = parser.parse_file(template_file)
            
            if not blocks:
                print("模板文件中没有找到块")
                return False
            
            # 使用第一个块作为模板
            self.block_identifier.add_block_template(name, blocks[0])
            return True
        
        except Exception as e:
            print(f"添加块模板时出错: {e}")
            return False
    
    def add_block_feature(self, feature: BlockFeature) -> bool:
        """添加块特征"""
        try:
            self.block_identifier.add_block_feature(feature)
            return True
        except Exception as e:
            print(f"添加块特征时出错: {e}")
            return False
    
    def save_analysis(self, output_file: str) -> bool:
        """保存分析结果"""
        try:
            self.cad_graph.export_to_json(output_file)
            return True
        except Exception as e:
            print(f"保存分析结果时出错: {e}")
            return False
    
    def load_analysis(self, input_file: str) -> bool:
        """加载分析结果"""
        return self.cad_graph.import_from_json(input_file)
    
    def query(self, query_str: str) -> Dict:
        """执行查询"""
        return self.query_interface.query(query_str)
    
    def get_blocks_by_type(self, block_type: str) -> List[Block]:
        """获取特定类型的所有块"""
        return self.query_interface.get_blocks_by_type(block_type)
    
    def has_connection(self, source_id: str, target_id: str) -> bool:
        """检查两个块之间是否有连接"""
        return self.query_interface.has_out_connection_to(source_id, target_id)
    
    def get_in_degree(self, block_id: str) -> int:
        """获取块的入度"""
        return self.query_interface.get_in_degree(block_id)
    
    def get_out_degree(self, block_id: str) -> int:
        """获取块的出度"""