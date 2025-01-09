import ezdxf
from ezdxf import bbox
from ezdxf.math import Vec3
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Type
import json
import uuid
import math
import networkx as nx


@dataclass
class AttributeInfo:
    """ATTRIB实体信息类"""
    tag: str  # 属性标签
    value: str  # 属性值
    position: tuple  # 属性位置
    height: float  # 属性文字高度
    rotation: float  # 属性旋转角度
    layer: str  # 属性所在图层
    style: str  # 属性文字样式


@dataclass
class EntityInfo:
    """实体信息类"""
    dxf_entity: object  # ezdxf entity
    entity_type: str
    layer: str
    block_name: Optional[str] = None  # 如果实体是 INSERT 类型，存储其引用的 block 名称
    rotation: float = 0.0  # block 的旋转角度
    scale: tuple = (1.0, 1.0, 1.0)  # block 的缩放因子
    position: tuple = (0.0, 0.0, 0.0)  # block 的插入点
    attributes: List[AttributeInfo] = field(
        default_factory=list)  # 新增：存储ATTRIB实体
    sub_entities: List[str] = field(default_factory=list)  # 新增：存储组成实体的ID
    # TODO 使用ezdxf内部的uuid
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 添加唯一标识符

    def add_attribute(self, attrib: 'AttributeInfo'):
        """添加ATTRIB实体"""
        self.attributes.append(attrib)

    def get_attribute(self, tag: str) -> Optional['AttributeInfo']:
        """根据标签获取ATTRIB实体"""
        for attrib in self.attributes:
            if attrib.tag == tag:
                return attrib
        return None

    def get_attribute_value(self, tag: str) -> Optional[str]:
        """根据标签获取ATTRIB实体的值"""
        attrib = self.get_attribute(tag)
        return attrib.value if attrib else None

    def set_attribute_value(self, tag: str, value: str) -> bool:
        """设置ATTRIB实体的值"""
        attrib = self.get_attribute(tag)
        if attrib:
            attrib.value = value
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, EntityInfo):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class EntityGroup:
    """实体组类，用于表示一组相关的实体"""
    name: str
    entities: List[EntityInfo]
    properties: Dict = field(default_factory=dict)


@dataclass
class BlockPattern:
    """块模式类，用于定义块的特征模式"""
    name: str
    entity_types: Set[str]
    layers: Set[str]
    entity_count: int
    width_range: tuple = (0, float('inf'))
    height_range: tuple = (0, float('inf'))
    aspect_ratio_range: tuple = (0, float('inf'))

    @classmethod
    def from_block_features(cls, features: dict, tolerance: float = 0.1) -> Optional['BlockPattern']:
        """从block特征创建模式"""
        if not features:
            return None

        width = features.get('width')
        height = features.get('height')
        aspect_ratio = features.get('aspect_ratio')

        return cls(
            name=features['name'],
            entity_types=features['entity_types'],
            layers=features['layers'],
            entity_count=features['entity_count'],
            width_range=(
                width * (1 - tolerance) if width is not None else 0,
                width * (1 + tolerance) if width is not None else float('inf')
            ),
            height_range=(
                height * (1 - tolerance) if height is not None else 0,
                height * (1 + tolerance) if height is not None else float('inf')
            ),
            aspect_ratio_range=(
                aspect_ratio *
                (1 - tolerance) if aspect_ratio is not None else 0,
                aspect_ratio *
                (1 + tolerance) if aspect_ratio is not None else float('inf')
            )
        )

    def matches(self, features: dict) -> bool:
        """检查特征是否匹配模式"""
        if not features:
            return False

        # 检查实体类型
        if not self.entity_types.issubset(features['entity_types']):
            return False

        # # 检查图层
        # if not any(layer in self.layers for layer in features['layers']):
        #     return False

        # 检查实体数量
        if self.entity_count != features['entity_count']:
            return False

        # # 检查尺寸
        # width = features.get('width')
        # if width is not None and not (self.width_range[0] <= width <= self.width_range[1]):
        #     return False

        # height = features.get('height')
        # if height is not None and not (self.height_range[0] <= height <= self.height_range[1]):
        #     return False

        aspect_ratio = features.get('aspect_ratio')
        if aspect_ratio is not None and not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False

        return True


class EntityNetwork:
    """实体网络类，用于管理实体之间的连接关系"""

    def __init__(self, dxf_path: str):
        self.doc = ezdxf.readfile(dxf_path, encoding='utf-8')
        self.msp = self.doc.modelspace()
        self.entities: List[EntityInfo] = []
        self.connections: Dict[str, Set[str]] = {}
        self.entity_groups: Dict[str, EntityGroup] = {}
        self.blocks = {}
        self.load_entities()

    def load_entities(self):
        """从DXF文件加载所有实体"""
        # 加载所有 blocks
        self.blocks = {}
        for block in self.doc.blocks:
            self.blocks[block.name] = block

        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                # 处理 block 引用
                entity_info = EntityInfo(
                    dxf_entity=entity,
                    entity_type='INSERT',
                    layer=entity.dxf.layer,
                    block_name=entity.dxf.name,
                    rotation=entity.dxf.rotation,
                    scale=(entity.dxf.xscale, entity.dxf.yscale,
                           entity.dxf.zscale),
                    position=tuple(entity.dxf.insert)
                )

                # 加载INSERT实体的属性
                if hasattr(entity, 'attribs'):
                    for attrib in entity.attribs:
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
                            entity_info.add_attribute(attrib_info)

                # 添加INSERT实体到entities列表
                if entity.dxftype() == 'INSERT':
                    self.entities.append(entity_info)
                    self.update_connections(entity_info)
                    # 加载 block 的组成实体
                    block = self.blocks.get(entity_info.block_name)
                    if block:
                        for sub_entity in block:
                            sub_entity_info = EntityInfo(
                                dxf_entity=sub_entity,
                                entity_type=sub_entity.dxftype(),
                                layer=sub_entity.dxf.layer
                            )
                            entity_info.sub_entities.append(sub_entity_info.id)
            else:
                entity_info = EntityInfo(
                    dxf_entity=entity,
                    entity_type=entity.dxftype(),
                    layer=entity.dxf.layer
                )

                # 加载ATTRIB实体
                if entity.dxftype() == 'ATTRIB':
                    attrib_info = AttributeInfo(
                        tag=entity.dxf.tag,
                        value=entity.dxf.text,
                        position=tuple(entity.dxf.insert),
                        height=entity.dxf.height,
                        rotation=entity.dxf.rotation,
                        layer=entity.dxf.layer,
                        style=entity.dxf.style
                    )
                    # 查找对应的INSERT实体并添加属性
                    insert_entity = next((e for e in self.entities if e.entity_type == 'INSERT' and
                                          e.dxf_entity.dxf.handle == entity.dxf.owner), None)
                    if insert_entity:
                        insert_entity.add_attribute(attrib_info)
                        continue  # 跳过后续的添加操作
                    else:
                        # 如果没有找到对应的INSERT实体，创建一个新的EntityInfo
                        entity_info = EntityInfo(
                            dxf_entity=entity,
                            entity_type='ATTRIB',
                            layer=entity.dxf.layer
                        )
                        entity_info.add_attribute(attrib_info)
                        self.entities.append(entity_info)
                        self.update_connections(entity_info)
                        continue  # 跳过后续的添加操作

                # 添加非INSERT和非ATTRIB实体
                self.entities.append(entity_info)
                self.update_connections(entity_info)

    def update_connections(self, entity_info: EntityInfo):
        """更新实体之间的连接关系（改进版）"""
        tolerance = 0.01

        if entity_info.entity_type == 'LINE':
            line = entity_info.dxf_entity
            start_point = Vec3(line.dxf.start)
            end_point = Vec3(line.dxf.end)

            for other in self.entities:
                if other.id == entity_info.id:
                    continue

                if other.entity_type == 'LINE':
                    other_line = other.dxf_entity
                    other_start = Vec3(other_line.dxf.start)
                    other_end = Vec3(other_line.dxf.end)

                    if (start_point.isclose(other_start, abs_tol=tolerance) or
                            start_point.isclose(other_end, abs_tol=tolerance) or
                            end_point.isclose(other_start, abs_tol=tolerance) or
                            end_point.isclose(other_end, abs_tol=tolerance)):
                        self.add_connection(entity_info, other)
                elif other.entity_type in ('CIRCLE', 'ARC'):
                    center = Vec3(other.dxf_entity.dxf.center)
                    radius = other.dxf_entity.dxf.radius
                    for point in [start_point, end_point]:
                        if (point - center).magnitude < radius + tolerance and (point - center).magnitude > abs(radius - tolerance):
                            self.add_connection(entity_info, other)
                            break

    def add_connection(self, entity1: EntityInfo, entity2: EntityInfo):
        """添加两个实体之间的连接关系"""
        if entity1.id not in self.connections:
            self.connections[entity1.id] = set()
        if entity2.id not in self.connections:
            self.connections[entity2.id] = set()

        self.connections[entity1.id].add(entity2.id)
        self.connections[entity2.id].add(entity1.id)

    def find_connected_components(self, target_layer: str) -> list[EntityInfo]:
        """查找与指定图层上的实体相连的所有实体"""
        result = []
        start_entities = [e for e in self.entities if e.layer == target_layer]

        for start_entity in start_entities:
            if start_entity not in self.connections:
                continue

            connected = set()
            to_visit = [start_entity]

            while to_visit:
                current = to_visit.pop()
                if current not in connected:
                    connected.add(current)
                    to_visit.extend(
                        neighbor for neighbor in self.connections.get(current, set())
                        if neighbor not in connected
                    )

            result.extend(list(connected))

        return result

    def create_entity_group(self, name: str, entities: List[EntityInfo], properties: Dict = None) -> EntityGroup:
        """创建一个新的实体组"""
        group = EntityGroup(name=name, entities=entities,
                            properties=properties or {})
        self.entity_groups[name] = group
        return group

    def find_similar_entity_groups(self, pattern: BlockPattern, tolerance: float = 0.1) -> List[EntityGroup]:
        """查找与模式相似的实体组（改进版）"""
        similar_groups = []
        visited = set()

        # 1. 查找所有特殊实体（按优先级排序）
        def entity_specialness(entity: EntityInfo) -> int:
            """定义实体特殊性"""
            if entity.entity_type == 'INSERT' and entity.block_name in [p.name for p in self.extract_block_patterns()]:
                return 3  # 模式中存在的块实例优先级最高
            if entity.entity_type not in ('LINE', 'POLYLINE', 'LWPOLYLINE'):
                return 2  # 非线段实体优先级较高
            return 1

        special_entities = sorted(
            [e for e in self.entities if e.entity_type in pattern.entity_types], 
            key=entity_specialness, 
            reverse=True
        )

        # 2. 以特殊实体为起点进行扩展
        for special_entity in special_entities:
            if special_entity.id in visited:
                continue

            connected = set()
            to_visit = [special_entity]

            while to_visit:
                current = to_visit.pop()
                if current.id not in visited:
                    connected.add(current)
                    visited.add(current.id)

                    # 限制扩展数量，避免无限扩展
                    if len(connected) > pattern.entity_count * 2:
                        break

                    # 获取相连实体
                    connected_ids = self.connections.get(current.id, set())
                    to_visit.extend(
                        next((e for e in self.entities if e.id == connected_id), None)
                        for connected_id in connected_ids 
                        if connected_id not in visited
                    )

                # 如果当前连通分量的实体数量等于目标数量，检查是否匹配
                if len(connected) == pattern.entity_count:
                    if self._match_entity_group_to_pattern(list(connected), pattern, tolerance):
                        similar_groups.append(EntityGroup(
                            name=f"EntityGroup_{len(similar_groups)}",
                            entities=list(connected)
                        ))
                    break

        return similar_groups

    def _match_entity_group_to_pattern(self, entities: List[EntityInfo], pattern: BlockPattern, tolerance: float) -> bool:
        """检查一组实体是否匹配指定的模式（改进版）"""
        # 检查实体数量
        if len(entities) != pattern.entity_count:
            return False

        # 检查实体类型
        entity_types = {e.entity_type for e in entities}
        if not pattern.entity_types.issubset(entity_types):
            return False

        # 检查长宽比
        mybbox = self.get_bounding_box(entities)
        if mybbox is not None:  # 只在能获取到边界框时进行长宽比检查
            width = mybbox[1][0] - mybbox[0][0]
            height = mybbox[1][1] - mybbox[0][1]
          
            if height != 0:
                aspect_ratio = width / height
                if not (pattern.aspect_ratio_range[0] <= aspect_ratio <= pattern.aspect_ratio_range[1]):
                    return False

        return True

    def get_entity_info(self, entity_info: EntityInfo) -> dict:
        """获取实体的详细信息"""
        entity = entity_info.dxf_entity
        info = {
            'type': entity_info.entity_type,
            'layer': entity_info.layer,
        }

        if entity_info.entity_type == 'INSERT':
            info.update({
                'block_name': entity_info.block_name,
                'rotation': entity_info.rotation,
                'scale': entity_info.scale,
                'position': entity_info.position,
                'block_features': self.get_block_features(entity_info.block_name),
                'attributes': [
                    {
                        'tag': attr.tag,
                        'value': attr.value,
                        'position': attr.position,
                        'height': attr.height,
                        'rotation': attr.rotation,
                        'layer': attr.layer,
                        'style': attr.style
                    }
                    for attr in entity_info.attributes
                ] if entity_info.attributes else []
            })
        elif entity_info.entity_type == 'LINE':
            info.update({
                'start': tuple(entity.dxf.start),
                'end': tuple(entity.dxf.end),
            })
        elif entity_info.entity_type == 'CIRCLE':
            info.update({
                'center': tuple(entity.dxf.center),
                'radius': entity.dxf.radius,
            })
        elif entity_info.entity_type == 'TEXT':
            info.update({
                'text': entity.dxf.text,
                'position': tuple(entity.dxf.insert),
            })
        elif entity_info.entity_type == 'LWPOLYLINE':
            info.update({
                'points': [tuple(float(coord) for coord in p) for p in entity.get_points()],
            })
        elif entity_info.entity_type == 'POLYLINE':
            info.update({
                'points': [tuple(float(coord) for coord in p) for p in entity.points()],
            })
        elif entity_info.entity_type == 'ARC':
            info.update({
                'center': tuple(entity.dxf.center),
                'radius': entity.dxf.radius,
                'start_angle': entity.dxf.start_angle,
                'end_angle': entity.dxf.end_angle,
            })
        elif entity_info.entity_type == 'ELLIPSE':
            info.update({
                'center': tuple(entity.dxf.center),
                'major_axis': tuple(entity.dxf.major_axis),
                'minor_axis': tuple(entity.dxf.minor_axis),
            })
        elif entity_info.entity_type == 'SPLINE':
            info.update({
                'control_points': [tuple(p) for p in entity.control_points],
            })
        elif entity_info.entity_type == 'MTEXT':
            info.update({
                'text': entity.dxf.text,
                'position': tuple(entity.dxf.insert),
            })

        return info

    def get_block_features(self, block_name: str, insert_entity: Optional['ezdxf.entities.Insert'] = None) -> Optional[dict]:
        """获取指定 block 的特征信息，可以传入 insert 实体来计算真实边界框"""
        if block_name not in self.blocks:
            return None

        block = self.blocks[block_name]
        features = {
            'name': block_name,
            'entities': [],
            'entity_count': 0,
            'entity_types': set(),
            'layers': set(),
            'scale': (1.0, 1.0, 1.0),  # 默认为单位缩放
            'position': (0.0, 0.0, 0.0),  # 默认为原点
            'rotation': 0.0,  # 默认为0度
        }

        # 收集所有实体信息
        entities = []
        for entity in block:
            entity_info = EntityInfo(
                dxf_entity=entity,
                entity_type=entity.dxftype(),
                layer=entity.dxf.layer
            )
            entities.append(entity_info)
          
            # 收集实体信息
            features['entities'].append(self.get_entity_info(entity_info))
            features['entity_types'].add(entity.dxftype())
            features['layers'].add(entity.dxf.layer)
            features['entity_count'] += 1

        # 如果提供了 insert 实体，则使用其缩放、旋转和平移
        if insert_entity:
            features['scale'] = (
                insert_entity.dxf.xscale,
                insert_entity.dxf.yscale,
                insert_entity.dxf.zscale
            )
            features['position'] = tuple(insert_entity.dxf.insert)
            features['rotation'] = insert_entity.dxf.rotation

        # 计算边界框
        mybbox = self.get_bounding_box(entities)
        if mybbox is None:
            features.update({
                'bounding_box': None,
                'width': None,
                'height': None,
                'center': None,
                'aspect_ratio': None
            })
        else:
            # 应用缩放、旋转和平移到边界框
            min_x, min_y = mybbox[0]
            max_x, max_y = mybbox[1]

            # 缩放
            scale_x, scale_y, _ = features['scale']
            min_x *= scale_x
            min_y *= scale_y
            max_x *= scale_x
            max_y *= scale_y

            # 旋转
            rotation = math.radians(features['rotation'])
            cos_theta = math.cos(rotation)
            sin_theta = math.sin(rotation)

            # 获取中心点
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # 将边界框的四个角点旋转
            corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y)
            ]

            rotated_corners = []
            for x, y in corners:
                # 平移到原点
                x -= center_x
                y -= center_y

                # 旋转
                rotated_x = x * cos_theta - y * sin_theta
                rotated_y = x * sin_theta + y * cos_theta

                # 平移回原位
                rotated_x += center_x
                rotated_y += center_y

                rotated_corners.append((rotated_x, rotated_y))

            # 计算旋转后的边界框
            rotated_min_x = min(x for x, _ in rotated_corners)
            rotated_min_y = min(y for _, y in rotated_corners)
            rotated_max_x = max(x for x, _ in rotated_corners)
            rotated_max_y = max(y for _, y in rotated_corners)

            # 平移
            pos_x, pos_y, _ = features['position']
            rotated_min_x += pos_x
            rotated_min_y += pos_y
            rotated_max_x += pos_x
            rotated_max_y += pos_y

            scaled_bbox = ((rotated_min_x, rotated_min_y),
                          (rotated_max_x, rotated_max_y))

            features.update({
                'bounding_box': scaled_bbox,
                'width': scaled_bbox[1][0] - scaled_bbox[0][0],
                'height': scaled_bbox[1][1] - scaled_bbox[0][1],
                'center': (
                    (scaled_bbox[0][0] + scaled_bbox[1][0]) / 2,
                    (scaled_bbox[0][1] + scaled_bbox[1][1]) / 2
                ),
                'aspect_ratio': (scaled_bbox[1][0] - scaled_bbox[0][0]) / (scaled_bbox[1][1] - scaled_bbox[0][1])
                if scaled_bbox[1][1] - scaled_bbox[0][1] != 0 else None
            })

        return features

    def extract_block_patterns(self, tolerance: float = 0.1) -> List[BlockPattern]:
        """提取所有block的特征模式"""
        patterns = []

        # 遍历所有blocks
        for block_name in self.blocks:
            features = self.get_block_features(block_name)
            if features is not None:
                pattern = BlockPattern.from_block_features(features, tolerance)
                if pattern is not None and block_name != "*Model_Space":  # 只添加有效的模式
                    patterns.append(pattern)

        return patterns

    def find_matching_blocks(self, pattern: BlockPattern) -> List[EntityInfo]:
        """在当前图纸中查找匹配指定模式的blocks"""
        matching_blocks = []

        for entity in self.entities:
            if entity.entity_type == 'INSERT':
                block_info = self.get_entity_info(entity)
                if pattern.matches(block_info['block_features']):
                    matching_blocks.append(entity)

        return matching_blocks

    def get_bounding_box(self, entities: List[EntityInfo]) -> Optional[tuple]:
        """获取一组实体的边界框"""
        if not entities:
            return None

        # 定义需要过滤的实体类型
        filter_types = {'ATTRIB', 'ATTDEF'}
      
        # 收集所有需要处理的实体
        entities_to_process = []
      
        for entity in entities:
            if entity.entity_type == 'INSERT':
                # 查找并处理所有子实体
                for sub_entity_id in entity.sub_entities:
                    sub_entity = next((e for e in self.entities if e.id == sub_entity_id), None)
                    if sub_entity and sub_entity.entity_type not in filter_types:
                        entities_to_process.append(sub_entity.dxf_entity)
            elif entity.entity_type not in filter_types:
                entities_to_process.append(entity.dxf_entity)
      
        # 如果没有实体需要处理，返回 None
        if not entities_to_process:
            return None
          
        # 一次性处理所有实体
        mybbox = bbox.extents(entities_to_process)
        return ((mybbox.extmin.x, mybbox.extmin.y), 
                (mybbox.extmax.x, mybbox.extmax.y))


def find_matching_entities(source_dxf_path: str, target_dxf_path: str) -> list:
    """
    从源DXF提取块模式，并在目标DXF中查找匹配的块和实体组。

    Args:
        source_dxf_path: 源DXF文件的路径。(单个原件)
        target_dxf_path: 目标DXF文件的路径（整张图）。

    Returns:
        一个列表，其中包含匹配的块和实体组的信息。
        每个匹配项都是一个字典，包含以下键：
        - type: "block" 或 "entity_group"
        - name: 如果是块，则为块名称；如果是实体组，则为None
        - position: 如果是块，则为块的位置；如果是实体组，则为None
        - rotation: 如果是块，则为块的旋转角度；如果是实体组，则为None
        - center: 匹配项的中心位置
        - bounding_box: 匹配项的边界框
    """
    source_network = EntityNetwork(source_dxf_path)
    block_patterns = source_network.extract_block_patterns()

    all_matching_groups = []

    if block_patterns:
        target_network = EntityNetwork(target_dxf_path)

        for pattern in block_patterns:
            # 查找block引用
            matching_blocks = target_network.find_matching_blocks(pattern)
            for block in matching_blocks:
                block_features = target_network.get_block_features(
                    block.block_name, block.dxf_entity)
                if block_features and block_features['bounding_box']:
                    mybbox = block_features['bounding_box']
                    center = (
                        (mybbox[0][0] + mybbox[1][0]) / 2,
                        (mybbox[0][1] + mybbox[1][1]) / 2
                    )
                    result = {
                        "type": "block",
                        "name": block.block_name,
                        "position": block.position,
                        "rotation": block.rotation,
                        "center": center,
                        "bounding_box": mybbox
                    }
                    # 如果有属性则添加
                    if block.attributes:
                        result["attributes"] = [
                            {
                                "tag": attr.tag,
                                "value": attr.value,
                                "position": attr.position,
                                "height": attr.height,
                                "rotation": attr.rotation,
                                "layer": attr.layer,
                                "style": attr.style
                            }
                            for attr in block.attributes
                        ]
                    all_matching_groups.append(result)
                else:
                    all_matching_groups.append({
                        "type": "block",
                        "name": block.block_name,
                        "position": block.position,
                        "rotation": block.rotation,
                        "center": None,
                        "bounding_box": None
                    })

            # 查找相似的实体组
            similar_groups = target_network.find_similar_entity_groups(pattern)
            for group in similar_groups:
                mybbox = target_network.get_bounding_box(group.entities)
                if mybbox:
                    center = (
                        (mybbox[0][0] + mybbox[1][0]) / 2,
                        (mybbox[0][1] + mybbox[1][1]) / 2
                    )
                    all_matching_groups.append({
                        "type": "entity_group",
                        "name": None,
                        "position": None,
                        "rotation": None,
                        "center": center,
                        "bounding_box": mybbox
                    })
                else:
                    all_matching_groups.append({
                        "type": "entity_group",
                        "name": None,
                        "position": None,
                        "rotation": None,
                        "center": None,
                        "bounding_box": None
                    })

    return all_matching_groups


if __name__ == "__main__":
    source_dxf = "extracted_blocks/VALLGA.dxf"
    target_dxf = "图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf"
    # target_dxf = "Drawing1.dxf"
    # target_dxf = "图例和流程图_仪表管件设备均为普通线条/2308PM-09-T3-2900.dxf"

    matching_results = find_matching_entities(source_dxf, target_dxf)

    # You can now work with the matching_results list
    # For example, print the number of matches:
    print(f"Found {len(matching_results)} matching entities.")

    # Or iterate through the results:
    for result in matching_results:
        print(result)
