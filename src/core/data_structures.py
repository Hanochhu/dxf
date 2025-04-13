"""
核心数据结构模块
提供基本的点、线、块等数据结构的定义
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from enum import Enum


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
    LEADER = "LEADER"
    SOLID = "SOLID"
    POINT = "POINT"


@dataclass
class Point:
    """三维点表示"""

    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: "Point") -> float:
        """计算到另一点的欧几里得距离"""
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        tolerance = 0.001  # 浮点比较的容差
        return (
            abs(self.x - other.x) < tolerance
            and abs(self.y - other.y) < tolerance
            and abs(self.z - other.z) < tolerance
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组表示"""
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, coords) -> "Point":
        """
        从元组、列表或字典创建点，自动兼容多种格式，并强制转换为 float
        支持：(x, y, z)、[x, y, z]、{'x':x,'y':y,'z':z}、{'x':x,'y':y}
        """

        def to_float(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        if isinstance(coords, dict):
            x = to_float(coords.get("x", 0.0))
            y = to_float(coords.get("y", 0.0))
            z = to_float(coords.get("z", 0.0))
            return cls(x, y, z)
        elif isinstance(coords, (tuple, list)):
            x = to_float(coords[0]) if len(coords) > 0 else 0.0
            y = to_float(coords[1]) if len(coords) > 1 else 0.0
            z = to_float(coords[2]) if len(coords) > 2 else 0.0
            return cls(x, y, z)
        else:
            raise ValueError("Point.from_tuple 不支持的类型: {}".format(type(coords)))


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
        return self.width / self.height if self.height != 0 else float("inf")

    @property
    def center(self) -> Point:
        """计算中心点"""
        return Point(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2,
        )

    def overlaps(self, other: "BoundingBox", tolerance: float = 0.001) -> bool:
        """检查边界框是否与另一边界框重叠"""
        return not (
            self.min_point.x > other.max_point.x + tolerance
            or self.max_point.x < other.min_point.x - tolerance
            or self.min_point.y > other.max_point.y + tolerance
            or self.max_point.y < other.min_point.y - tolerance
        )

    def contains_point(self, point: Point, tolerance: float = 0.001) -> bool:
        """检查边界框是否包含点"""
        return (
            self.min_point.x - tolerance <= point.x <= self.max_point.x + tolerance
            and self.min_point.y - tolerance <= point.y <= self.max_point.y + tolerance
            and self.min_point.z - tolerance <= point.z <= self.max_point.z + tolerance
        )

    def to_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """转换为元组表示"""
        return (
            (self.min_point.x, self.min_point.y),
            (self.max_point.x, self.max_point.y),
        )

    @classmethod
    def from_points(cls, points: List[Point]) -> "BoundingBox":
        """从点列表创建边界框"""
        if not points:
            raise ValueError("Points list cannot be empty")

        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        min_z = min(p.z for p in points)

        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        max_z = max(p.z for p in points)

        return cls(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))

    @classmethod
    def from_entities(cls, entities: List["Entity"]) -> Optional["BoundingBox"]:
        """从实体列表创建整体边界框（跳过无效 bounding_box 的实体）"""
        min_points = []
        max_points = []
        for e in entities:
            if hasattr(e, "bounding_box") and e.bounding_box:
                min_points.append(e.bounding_box.min_point)
                max_points.append(e.bounding_box.max_point)
        if not min_points or not max_points:
            return None
        min_x = min(p.x for p in min_points)
        min_y = min(p.y for p in min_points)
        min_z = min(p.z for p in min_points)
        max_x = max(p.x for p in max_points)
        max_y = max(p.y for p in max_points)
        max_z = max(p.z for p in max_points)
        return cls(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))

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
            "tag": self.tag,
            "value": self.value,
            "position": self.position,
            "height": self.height,
            "rotation": self.rotation,
            "layer": self.layer,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AttributeInfo":
        """从字典创建属性信息"""
        return cls(
            tag=data["tag"],
            value=data["value"],
            position=data["position"],
            height=data["height"],
            rotation=data["rotation"],
            layer=data["layer"],
            style=data["style"],
        )


@dataclass
class Entity:
    """基础实体类"""

    id: str
    entity_type: EntityType
    layer: str
    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """所有实体必须实现自己的边界框计算"""
        raise NotImplementedError("子类必须实现 bounding_box 属性")

    def get_feature_vector(self) -> List[float]:
        """生成实体的特征向量（用于识别）"""
        if not self.bounding_box:
            return [0, 0, 0]

        return [
            self.bounding_box.width,
            self.bounding_box.height,
            self.bounding_box.aspect_ratio,
        ]

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "type": self.entity_type.value,
            "layer": self.layer,
        }
        if self.bounding_box:
            result["bounding_box"] = {
                "min": self.bounding_box.min_point.to_tuple(),
                "max": self.bounding_box.max_point.to_tuple(),
            }
        return result


@dataclass
class EllipseEntity(Entity):
    center: Point
    major_axis: tuple
    ratio: float
    start_param: float
    end_param: float
    major_radius: float
    minor_radius: float

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """返回椭圆的边界框（简化：不考虑旋转，仅主轴方向）"""
        if not hasattr(self, 'center') or not hasattr(self, 'major_radius') or not hasattr(self, 'minor_radius'):
            return None
        min_x = self.center.x - self.major_radius
        max_x = self.center.x + self.major_radius
        min_y = self.center.y - self.minor_radius
        max_y = self.center.y + self.minor_radius
        min_z = self.center.z
        max_z = self.center.z
        return BoundingBox(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))

    rotation_angle: float


@dataclass
class LeaderEntity(Entity):

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """返回引线的边界框"""
        if not self.vertices:
            return None
        pts = [p if isinstance(p, Point) else Point.from_tuple(p) for p in self.vertices]
        return BoundingBox.from_points(pts)

    vertices: list


@dataclass
class SolidEntity(Entity):
    points: list

    @property
    def bounding_box(self) -> BoundingBox:
        """返回四个顶点的边界框"""
        pts = [p if isinstance(p, Point) else Point.from_tuple(p) for p in self.points]
        return BoundingBox.from_points(pts)


@dataclass
class PointEntity(Entity):
    position: Point

    @property
    def bounding_box(self) -> BoundingBox:
        """返回以该点为中心、边长为0的边界框"""
        return BoundingBox(self.position, self.position)

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = {
            "id": self.id,
            "type": self.entity_type.value,
            "layer": self.layer,
        }

        if self.bounding_box:
            result["bounding_box"] = {
                "min": self.bounding_box.min_point.to_tuple(),
                "max": self.bounding_box.max_point.to_tuple(),
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        """从字典创建实体"""
        bbox = None
        if "bounding_box" in data:
            bbox = BoundingBox(
                Point.from_tuple(data["bounding_box"]["min"]),
                Point.from_tuple(data["bounding_box"]["max"]),
            )

        return cls(
            id=data["id"],
            entity_type=EntityType(data["type"]),
            layer=data["layer"],
            bounding_box=bbox,
        )


@dataclass
class LineEntity(Entity):
    """线段实体"""

    start_point: Point = field(default_factory=lambda: Point(0, 0, 0))
    end_point: Point = field(default_factory=lambda: Point(0, 0, 0))

    @property
    def bounding_box(self) -> BoundingBox:
        """返回线段的边界框"""
        min_x = min(self.start_point.x, self.end_point.x)
        min_y = min(self.start_point.y, self.end_point.y)
        min_z = min(self.start_point.z, self.end_point.z)
        max_x = max(self.start_point.x, self.end_point.x)
        max_y = max(self.start_point.y, self.end_point.y)
        max_z = max(self.start_point.z, self.end_point.z)
        return BoundingBox(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))


@dataclass
class PolylineEntity(Entity):
    """多段线实体（POLYLINE）"""

    vertices: List[Point] = field(default_factory=list)
    is_closed: bool = False


@dataclass
class LwPolylineEntity(Entity):
    """轻量级多段线实体（LWPOLYLINE）"""

    vertices: List[Point] = field(default_factory=list)
    is_closed: bool = False


    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """返回多段线的边界框"""
        if not self.vertices:
            return None
        return BoundingBox.from_points(self.vertices)

    start_point: Point = field(default_factory=lambda: Point(0, 0, 0))
    end_point: Point = field(default_factory=lambda: Point(0, 0, 0))

    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            min_x = min(self.start_point.x, self.end_point.x)
            min_y = min(self.start_point.y, self.end_point.y)
            min_z = min(self.start_point.z, self.end_point.z)

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """返回轻量级多段线的边界框"""
        if not self.vertices:
            return None
        return BoundingBox.from_points(self.vertices)


    def get_direction(self) -> Tuple[float, float, float]:
        """获取方向向量（标准化）"""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        dz = self.end_point.z - self.start_point.z
        length = math.sqrt(dx**2 + dy**2 + dz**2)

        if length == 0:
            return (0, 0, 0)

        return (dx / length, dy / length, dz / length)

    def get_length(self) -> float:
        """获取线段长度"""
        return self.start_point.distance_to(self.end_point)

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update(
            {
                "start_point": self.start_point.to_tuple(),
                "end_point": self.end_point.to_tuple(),
                "length": self.get_length(),
                "direction": self.get_direction(),
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "LineEntity":
        """从字典创建线段实体"""
        base_entity = Entity.from_dict(data)

        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            start_point=Point.from_tuple(data["start_point"]),
            end_point=Point.from_tuple(data["end_point"]),
        )


@dataclass
class CircleEntity(Entity):
    """圆形实体"""

    center: Point = field(default_factory=Point)
    radius: float = 0.0

    @property
    def bounding_box(self) -> BoundingBox:
        """返回圆的边界框"""
        min_point = Point(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.z
        )
        max_point = Point(
            self.center.x + self.radius,
            self.center.y + self.radius,
            self.center.z
        )
        return BoundingBox(min_point, max_point)


    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            self.bounding_box = BoundingBox(
                Point(
                    self.center.x - self.radius,
                    self.center.y - self.radius,
                    self.center.z,
                ),
                Point(
                    self.center.x + self.radius,
                    self.center.y + self.radius,
                    self.center.z,
                ),
            )

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update({"center": self.center.to_tuple(), "radius": self.radius})
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "CircleEntity":
        """从字典创建圆形实体"""
        base_entity = Entity.from_dict(data)

        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            center=Point.from_tuple(data["center"]),
            radius=data["radius"],
        )


@dataclass
class ArcEntity(Entity):
    """弧形实体"""

    center: Point = field(default_factory=Point)
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0

    @property
    def bounding_box(self) -> BoundingBox:
        """返回弧的边界框（简化为整圆外接矩形）"""
        min_point = Point(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.z
        )
        max_point = Point(
            self.center.x + self.radius,
            self.center.y + self.radius,
            self.center.z
        )
        return BoundingBox(min_point, max_point)


    def __post_init__(self):
        """初始化后计算边界框"""
        if not self.bounding_box:
            # 简化版边界框计算，实际应考虑弧的起止角度
            self.bounding_box = BoundingBox(
                Point(
                    self.center.x - self.radius,
                    self.center.y - self.radius,
                    self.center.z,
                ),
                Point(
                    self.center.x + self.radius,
                    self.center.y + self.radius,
                    self.center.z,
                ),
            )

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update(
            {
                "center": self.center.to_tuple(),
                "radius": self.radius,
                "start_angle": self.start_angle,
                "end_angle": self.end_angle,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "ArcEntity":
        """从字典创建弧形实体"""
        base_entity = Entity.from_dict(data)

        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            center=Point.from_tuple(data["center"]),
            radius=data["radius"],
            start_angle=data["start_angle"],
            end_angle=data["end_angle"],
        )


@dataclass
class TextEntity(Entity):
    """文本实体"""

    text: str = ""
    position: Point = field(default_factory=Point)
    height: float = 0.0

    @property
    def bounding_box(self) -> BoundingBox:
        """返回文本的边界框（简化版）"""
        text_width = len(self.text) * self.height * 0.6
        min_x = self.position.x
        min_y = self.position.y - self.height
        max_x = self.position.x + text_width
        max_y = self.position.y + self.height
        min_z = self.position.z
        max_z = self.position.z
        return BoundingBox(Point(min_x, min_y, min_z), Point(max_x, max_y, max_z))

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
                Point(max_x, max_y, self.position.z),
            )

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = super().to_dict()
        result.update(
            {
                "text": self.text,
                "position": self.position.to_tuple(),
                "height": self.height,
                "rotation": self.rotation,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "TextEntity":
        """从字典创建文本实体"""
        base_entity = Entity.from_dict(data)

        return cls(
            id=base_entity.id,
            entity_type=base_entity.entity_type,
            layer=base_entity.layer,
            bounding_box=base_entity.bounding_box,
            text=data["text"],
            position=Point.from_tuple(data["position"]),
            height=data["height"],
            rotation=data.get("rotation", 0.0),
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
    entity_type: EntityType = EntityType.INSERT  # BlockReference对应INSERT类型实体

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position.to_tuple(),
            "rotation": self.rotation,
            "scale": self.scale,
            "attributes": [attr.to_dict() for attr in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BlockReference":
        """从字典创建块引用"""
        return cls(
            id=data["id"],
            name=data["name"],
            position=Point.from_tuple(data["position"]),
            rotation=data["rotation"],
            scale=data["scale"],
            attributes=[
                AttributeInfo.from_dict(attr) for attr in data.get("attributes", [])
            ],
            entity_type=EntityType.INSERT if "EntityType" in globals() else None,
        )


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
            min_points = [
                e.bounding_box.min_point for e in self.entities if e.bounding_box
            ]
            max_points = [
                e.bounding_box.max_point for e in self.entities if e.bounding_box
            ]

            if min_points and max_points:
                min_x = min(p.x for p in min_points)
                min_y = min(p.y for p in min_points)
                min_z = min(p.z for p in min_points)

                max_x = max(p.x for p in max_points)
                max_y = max(p.y for p in max_points)
                max_z = max(p.z for p in max_points)

                self.bounding_box = BoundingBox(
                    Point(min_x, min_y, min_z), Point(max_x, max_y, max_z)
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
                self.bounding_box.aspect_ratio,
            ]

        # 特征2：实体类型计数
        type_counts = self.count_entity_types()
        type_features = [type_counts[etype] for etype in EntityType]

        # 特征3：实体密度
        area = (
            self.bounding_box.width * self.bounding_box.height
            if self.bounding_box
            else 0
        )
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
            polyline_count = (
                type_counts[EntityType.POLYLINE] + type_counts[EntityType.LWPOLYLINE]
            )

            # 简单启发式：一定数量的线和多段线组合可能表示箭头
            if (line_count > 0 and polyline_count > 0) or line_count >= 2:
                # 进一步检查是否有三角形结构（箭头头部）
                return True

        return False

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        result = {
            "id": self.id,
            "name": self.name,
            "entity_count": len(self.entities),
            "entity_types": [e.entity_type.value for e in self.entities],
            "is_arrow": self.is_arrow,
        }

        if self.bounding_box:
            result["bounding_box"] = {
                "min": self.bounding_box.min_point.to_tuple(),
                "max": self.bounding_box.max_point.to_tuple(),
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
                "aspect_ratio": self.bounding_box.aspect_ratio,
            }

        if self.reference:
            result["reference"] = self.reference.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Block":
        """从字典创建块"""
        bbox = None
        if "bounding_box" in data:
            bbox = BoundingBox(
                Point.from_tuple(data["bounding_box"]["min"]),
                Point.from_tuple(data["bounding_box"]["max"]),
            )

        reference = None
        if "reference" in data:
            reference = BlockReference.from_dict(data["reference"])

        # 注意：这里不包括实体列表，需要单独处理
        return cls(
            id=data["id"],
            name=data["name"],
            entities=[],  # 实体列表需要单独处理
            bounding_box=bbox,
            reference=reference,
            is_arrow=data.get("is_arrow", False),
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

            return (dx / length, dy / length, dz / length)

        # 如果有多个路径段，返回最后一段的方向
        return self.path_segments[-1].get_direction()

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "id": self.id,
            "source_block_id": self.source_block.id,
            "target_block_id": self.target_block.id,
            "path_segment_ids": [segment.id for segment in self.path_segments],
            "has_explicit_direction": self.has_explicit_direction,
            "connection_type": self.connection_type,
            "direction": self.direction,
        }


@dataclass
class BlockFeature:
    """块特征类 - 用于定义识别特定类型块的特征"""

    name: str
    description: str = ""
    entity_types: Set[EntityType] = field(default_factory=set)
    min_entity_count: int = 0
    max_entity_count: int = float("inf")
    min_aspect_ratio: float = 0
    max_aspect_ratio: float = float("inf")
    min_width: float = 0
    max_width: float = float("inf")
    min_height: float = 0
    max_height: float = float("inf")
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
            if not (
                self.min_aspect_ratio
                <= block.bounding_box.aspect_ratio
                <= self.max_aspect_ratio
            ):
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
            "name": self.name,
            "description": self.description,
            "entity_types": [etype.value for etype in self.entity_types],
            "min_entity_count": self.min_entity_count,
            "max_entity_count": self.max_entity_count,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "additional_checks": self.additional_checks,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BlockFeature":
        """从字典创建块特征"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            entity_types={EntityType(etype) for etype in data.get("entity_types", [])},
            min_entity_count=data.get("min_entity_count", 0),
            max_entity_count=data.get("max_entity_count", float("inf")),
            min_aspect_ratio=data.get("min_aspect_ratio", 0),
            max_aspect_ratio=data.get("max_aspect_ratio", float("inf")),
            min_width=data.get("min_width", 0),
            max_width=data.get("max_width", float("inf")),
            min_height=data.get("min_height", 0),
            max_height=data.get("max_height", float("inf")),
            additional_checks=data.get("additional_checks", []),
        )

    @classmethod
    def from_sample_block(
        cls, block: Block, name: str, description: str = "", tolerance: float = 0.2
    ) -> "BlockFeature":
        """从样例块创建特征模板"""
        # 计算特征范围
        entity_count = len(block.entities)
        entity_types = set(entity.entity_type for entity in block.entities)

        feature = cls(
            name=name,
            description=description,
            entity_types=entity_types,
            min_entity_count=max(1, int(entity_count * (1 - tolerance))),
            max_entity_count=int(entity_count * (1 + tolerance)),
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
