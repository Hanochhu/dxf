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
class EntityInfo:
    """实体信息类"""
    dxf_entity: object  # ezdxf entity
    entity_type: str
    layer: str
    block_name: Optional[str] = None  # 如果实体是 INSERT 类型，存储其引用的 block 名称
    rotation: float = 0.0  # block 的旋转角度
    scale: tuple = (1.0, 1.0, 1.0)  # block 的缩放因子
    position: tuple = (0.0, 0.0, 0.0)  # block 的插入点
    #TODO 使用ezdxf内部的uuid
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 添加唯一标识符
    
    def __eq__(self, other):
        if not isinstance(other, EntityInfo):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
@dataclass
class CompositeEntity:
    """复合实体类，用于组合多个基础实体"""
    name: str
    entities: List[EntityInfo] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)
    
    def add_entity(self, entity: EntityInfo):
        """添加基础实体到复合实体中"""
        self.entities.append(entity)
    
    def get_bounding_box(self) -> Optional[tuple]:
        """获取复合实体的边界框，表示的是相对于插入点的坐标，而且是没缩放"""
        if not self.entities:
            return None
            
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for entity in self.entities:
            mybbox = bbox.extents([entity.dxf_entity])
            min_x = min(min_x, mybbox.extmin.x)
            min_y = min(min_y, mybbox.extmin.y)
            max_x = max(max_x, mybbox.extmax.x)
            max_y = max(max_y, mybbox.extmax.y)
                
        if min_x == float('inf'):
            return None
            
        return ((min_x, min_y), (max_x, max_y))

@dataclass
class BlockPattern:
    """Block模式类，用于存储和匹配Block的特征"""
    name: str
    entity_count: int
    entity_types: Set[str]
    width_range: tuple  # (min_width, max_width)
    height_range: tuple  # (min_height, max_height)
    aspect_ratio_range: tuple  # (min_ratio, max_ratio)
    layers: Set[str]
    tolerance: float = 0.1  # 用于数值比较的容差
    
    @classmethod
    def from_block_features(cls, features: dict, tolerance: float = 0.1) -> Optional['BlockPattern']:
        """从block特征创建模式"""
        width = features['width']
        height = features['height']
        aspect_ratio = features['aspect_ratio']
        
        # 如果没有有效的几何特征，返回 None
        if width is None or height is None:
            return None
        
        return cls(
            name=features['name'],
            entity_count=features['entity_count'],
            entity_types=features['entity_types'],
            width_range=(width * (1-tolerance), width * (1+tolerance)),
            height_range=(height * (1-tolerance), height * (1+tolerance)),
            aspect_ratio_range=(
                aspect_ratio * (1-tolerance), 
                aspect_ratio * (1+tolerance)
            ) if aspect_ratio is not None else (None, None),
            layers=features['layers'],
            tolerance=tolerance
        )
    
    def to_json(self) -> str:
        """将模式转换为JSON字符串"""
        data = {
            'name': self.name,
            'entity_count': self.entity_count,
            'entity_types': list(self.entity_types),
            'width_range': self.width_range,
            'height_range': self.height_range,
            'aspect_ratio_range': self.aspect_ratio_range,
            'layers': list(self.layers),
            'tolerance': self.tolerance
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BlockPattern':
        """从JSON字符串创建模式"""
        data = json.loads(json_str)
        data['entity_types'] = set(data['entity_types'])
        data['layers'] = set(data['layers'])
        return cls(**data)
    
    def matches(self, features: dict) -> bool:
        """检查block特征是否匹配该模式"""
        if features is None:
            return False
            
        # 检查基本属性
        if features['entity_count'] != self.entity_count:
            return False
            
        if not self.entity_types.issubset(features['entity_types']):
            return False
            
        if not any(layer in self.layers for layer in features['layers']):
            return False
            
        # 检查尺寸
        width = features['width']
        if width is not None and not (self.width_range[0] <= width <= self.width_range[1]):
            return False
            
        height = features['height']
        if height is not None and not (self.height_range[0] <= height <= self.height_range[1]):
            return False
            
        # 检查宽高比
        if features['aspect_ratio'] is not None and self.aspect_ratio_range[0] is not None:
            if not (self.aspect_ratio_range[0] <= features['aspect_ratio'] <= self.aspect_ratio_range[1]):
                return False
                
        return True

class EntityNetwork:
    """实体网络类，用于管理实体之间的连接关系"""
    def __init__(self, dxf_path: str):
        self.doc = ezdxf.readfile(dxf_path)
        self.msp = self.doc.modelspace()
        self.entities: List[EntityInfo] = []
        self.connections: Dict[str, Set[str]] = {}  # 使用 id 作为键
        self.composite_entities: Dict[str, CompositeEntity] = {}
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
                    scale=(entity.dxf.xscale, entity.dxf.yscale, entity.dxf.zscale),
                    position=tuple(entity.dxf.insert)
                )
            else:
                entity_info = EntityInfo(
                    dxf_entity=entity,
                    entity_type=entity.dxftype(),
                    layer=entity.dxf.layer
                )
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
    
    def create_composite_entity(self, name: str, entities: List[EntityInfo], properties: Dict = None) -> CompositeEntity:
        """创建一个新的复合实体"""
        composite = CompositeEntity(name=name, properties=properties or {})
        for entity in entities:
            composite.add_entity(entity)
        self.composite_entities[name] = composite
        return composite
    
    def find_composite_entities_by_pattern(self, pattern: Dict) -> List[CompositeEntity]:
        """根据模式查找复合实体
        pattern 示例: {
            'entity_count': 3,  # 实体数量
            'types': ['LINE', 'CIRCLE'],  # 包含的实体类型
            'max_size': (100, 100),  # 最大尺寸
            'layers': ['PART_LAYER']  # 图层
        }
        """
        result = []
        
        # 获取所有连通分量
        visited = set()
        for entity in self.entities:
            if entity in visited:
                continue
                
            connected = set()
            to_visit = [entity]
            
            while to_visit:
                current = to_visit.pop()
                if current not in connected:
                    connected.add(current)
                    visited.add(current)
                    to_visit.extend(
                        neighbor for neighbor in self.connections.get(current, set())
                        if neighbor not in connected
                    )
            
            # 检查连通分量是否匹配模式
            if self._match_pattern(connected, pattern):
                composite = CompositeEntity(
                    name=f"Composite_{len(result)}",
                    entities=list(connected)
                )
                result.append(composite)
        
        return result
    
    def _match_pattern(self, entities: Set[EntityInfo], pattern: Dict) -> bool:
        """检查一组实体是否匹配指定的模式"""
        if 'entity_count' in pattern and len(entities) != pattern['entity_count']:
            return False
            
        if 'types' in pattern:
            entity_types = {e.entity_type for e in entities}
            if not all(t in entity_types for t in pattern['types']):
                return False
                
        if 'layers' in pattern:
            entity_layers = {e.layer for e in entities}
            if not any(layer in pattern['layers'] for layer in entity_layers):
                return False
                
        if 'max_size' in pattern:
            composite = CompositeEntity("temp", list(entities))
            mybbox = composite.get_bounding_box()
            if mybbox is None:
                return False
            
            width = mybbox[1][0] - mybbox[0][0]
            height = mybbox[1][1] - mybbox[0][1]
            if width > pattern['max_size'][0] or height > pattern['max_size'][1]:
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
                'block_features': self.get_block_features(entity_info.block_name)
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
            'position': (0.0, 0.0, 0.0), # 默认为原点
            'rotation': 0.0, # 默认为0度
        }
        
        # 创建临时的 CompositeEntity 来存储 block 中的所有实体
        temp_composite = CompositeEntity(name=block_name)
        
        for entity in block:
            entity_info = EntityInfo(
                dxf_entity=entity,
                entity_type=entity.dxftype(),
                layer=entity.dxf.layer
            )
            temp_composite.add_entity(entity_info)
            
            # 收集实体信息
            features['entities'].append(self.get_entity_info(entity_info))
            features['entity_types'].add(entity.dxftype())
            features['layers'].add(entity.dxf.layer)
            features['entity_count'] += 1
        
        # 如果提供了 insert 实体，则使用其缩放、旋转和平移
        if insert_entity:
            features['scale'] = (insert_entity.dxf.xscale, insert_entity.dxf.yscale, insert_entity.dxf.zscale)
            features['position'] = tuple(insert_entity.dxf.insert)
            features['rotation'] = insert_entity.dxf.rotation
        
        # 计算边界框
        mybbox = temp_composite.get_bounding_box()
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
            
            # 创建一个旋转矩阵
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
            
            scaled_bbox = ((rotated_min_x, rotated_min_y), (rotated_max_x, rotated_max_y))
            
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
    
    def find_similar_entity_groups(self, pattern: BlockPattern, tolerance: float = 0.1, verbose: bool = False) -> List[List[EntityInfo]]:
        """查找与模式相似的实体组（改进版）"""
        similar_groups = []
        visited = set()

        if verbose:
            print(f"\n开始查找相似实体组，模式特征:")
            print(f"- 实体数量: {pattern.entity_count}")
            print(f"- 实体类型: {pattern.entity_types}")
            print(f"- 宽度范围: {pattern.width_range}")
            print(f"- 高度范围: {pattern.height_range}")
            print(f"- 宽高比范围: {pattern.aspect_ratio_range}")
            print(f"- 图层: {pattern.layers}")

        # 1. 查找所有特殊实体（按优先级排序）
        def entity_specialness(entity: EntityInfo) -> int:
            """定义实体特殊性"""
            if entity.entity_type == 'INSERT' and entity.block_name in [p.name for p in self.extract_block_patterns()]:
                return 3  # 模式中存在的块实例优先级最高
            if entity.entity_type not in ('LINE', 'POLYLINE', 'LWPOLYLINE'):
                return 2  # 非线段实体优先级较高
            return 1

        special_entities = sorted([e for e in self.entities if e.entity_type in pattern.entity_types], key=entity_specialness, reverse=True)

        if verbose:
            print(f"找到 {len(special_entities)} 个候选起始实体")

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

                    if len(connected) > pattern.entity_count * 2:  # 限制扩展数量，避免无限扩展
                        break

                    connected_ids = self.connections.get(current.id, set())
                    to_visit.extend(
                        next((e for e in self.entities if e.id == connected_id), None)
                        for connected_id in connected_ids if connected_id not in visited
                    )

            # 3. 模式匹配
            if verbose:
                print(f"\n找到候选实体组，包含 {len(connected)} 个实体")
                print(f"实体类型分布: {[e.entity_type for e in connected]}")
                print(f"图层分布: {[e.layer for e in connected]}")

            if self._match_entity_group_to_pattern(list(connected), pattern, tolerance, verbose):
                if verbose:
                    print("候选实体组匹配成功")
                similar_groups.append(list(connected))
            elif verbose:
                print("候选实体组匹配失败")

        return similar_groups

    def _find_similar_entity_groups_fallback(self, pattern: BlockPattern, tolerance: float) -> List[List[EntityInfo]]:
        """当没有找到特殊实体时使用的回退方法"""
        similar_groups = []
        visited = set()
        
        # 遍历所有实体
        for entity in self.entities:
            if entity.id in visited:
                continue
                
            # 从当前实体开始，找到所有相连的实体
            connected = set()
            to_visit = [entity]
            
            while to_visit:
                current = to_visit.pop()
                if current.id not in visited:
                    connected.add(current)
                    visited.add(current.id)
                    # 获取相连的实体ID
                    connected_ids = self.connections.get(current.id, set())
                    # 将相连的实体添加到待访问列表
                    to_visit.extend(
                        next((e for e in self.entities if e.id == connected_id), None)
                        for connected_id in connected_ids
                    )
            
            # 检查这组实体是否匹配模式
            if self._match_entity_group_to_pattern(list(connected), pattern, tolerance):
                similar_groups.append(list(connected))
        
        return similar_groups
    
    def _match_entity_group_to_pattern(self, entities: List[EntityInfo], pattern: BlockPattern, tolerance: float, verbose: bool = False) -> bool:
        """检查一组实体是否匹配指定的模式（改进版）"""
        if verbose:
            print(f"\n开始匹配实体组，包含 {len(entities)} 个实体")
            print(f"实体类型分布: {[e.entity_type for e in entities]}")
            print(f"图层分布: {[e.layer for e in entities]}")

        if len(entities) != pattern.entity_count:
            if verbose:
                print(f"实体数量不匹配，需要{pattern.entity_count}, 实际{len(entities)}")
            return False

        entity_types = {e.entity_type for e in entities}
        if not pattern.entity_types.issubset(entity_types):
            if verbose:
                print(f"实体类型不匹配，需要包含{pattern.entity_types}, 实际包含{entity_types}")
            return False
            
        entity_layers = {e.layer for e in entities}
        if not any(layer in pattern.layers for layer in entity_layers):
            if verbose:
                print(f"图层不匹配，需要在{pattern.layers}其中之一，实际在{entity_layers}")
            return False
            
        temp_composite = CompositeEntity(name="temp")
        for entity in entities:
            temp_composite.add_entity(entity)

        # mybbox = temp_composite.get_bounding_box()
        # if mybbox is None:
        #     if verbose:
        #         print(f"无法计算边界框")
        #     return False

        # width = mybbox[1][0] - mybbox[0][0]
        # height = mybbox[1][1] - mybbox[0][1]
        # aspect_ratio = width / height if height != 0 else None

        # if not (pattern.width_range[0] <= width <= pattern.width_range[1]):
        #     if verbose:
        #         print(f"宽度不匹配，需要在{pattern.width_range}之间，实际为{width}")
        #     return False

        # if not (pattern.height_range[0] <= height <= pattern.height_range[1]):
        #     if verbose:
        #         print(f"高度不匹配，需要在{pattern.height_range}之间，实际为{height}")
        #     return False

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
                block_features = target_network.get_block_features(block.block_name, block.dxf_entity)
                if block_features and block_features['bounding_box']:
                    mybbox = block_features['bounding_box']
                    center = (
                        (mybbox[0][0] + mybbox[1][0]) / 2,
                        (mybbox[0][1] + mybbox[1][1]) / 2
                    )
                    all_matching_groups.append({
                        "type": "block",
                        "name": block.block_name,
                        "position": block.position,
                        "rotation": block.rotation,
                        "center": center,
                        "bounding_box": mybbox
                    })
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
                mybbox = CompositeEntity("temp", list(group)).get_bounding_box()
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
    source_dxf = "extracted_blocks/A3.dxf"
    target_dxf = "图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf"
    # target_dxf = "图例和流程图_仪表管件设备均为普通线条/2308PM-01-T3-2158.dxf"
    
    matching_results = find_matching_entities(source_dxf, target_dxf)
    
    # You can now work with the matching_results list
    # For example, print the number of matches:
    print(f"Found {len(matching_results)} matching entities.")
    
    # Or iterate through the results:
    for result in matching_results:
        print(result)