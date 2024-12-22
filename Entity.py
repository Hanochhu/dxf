import ezdxf
from ezdxf.math import Vec3
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Type
import json
import uuid

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
        """获取复合实体的边界框"""
        if not self.entities:
            return None
            
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for entity in self.entities:
            if entity.entity_type == 'LINE':
                start = entity.dxf_entity.dxf.start
                end = entity.dxf_entity.dxf.end
                min_x = min(min_x, start[0], end[0])
                min_y = min(min_y, start[1], end[1])
                max_x = max(max_x, start[0], end[0])
                max_y = max(max_y, start[1], end[1])
            elif entity.entity_type == 'CIRCLE':
                center = entity.dxf_entity.dxf.center
                radius = entity.dxf_entity.dxf.radius
                min_x = min(min_x, center[0] - radius)
                min_y = min(min_y, center[1] - radius)
                max_x = max(max_x, center[0] + radius)
                max_y = max(max_y, center[1] + radius)
                
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
        """更新实体之间的连接关系"""
        if entity_info.entity_type == 'LINE':
            line = entity_info.dxf_entity
            start_point = Vec3(line.dxf.start)
            end_point = Vec3(line.dxf.end)
            
            for other in self.entities:
                if other.dxf_entity is line:
                    continue
                
                if other.entity_type == 'LINE':
                    other_line = other.dxf_entity
                    other_start = Vec3(other_line.dxf.start)
                    other_end = Vec3(other_line.dxf.end)
                    
                    # 检查端点是否相连（使用ezdxf的Vec3进行计算）
                    if (start_point.isclose(other_start) or 
                        start_point.isclose(other_end) or
                        end_point.isclose(other_start) or
                        end_point.isclose(other_end)):
                        self.add_connection(entity_info, other)
    
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
            bbox = composite.get_bounding_box()
            if bbox:
                width = bbox[1][0] - bbox[0][0]
                height = bbox[1][1] - bbox[0][1]
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
        
        return info
    
    def get_block_features(self, block_name: str) -> Optional[dict]:
        """获取指定 block 的特征信息"""
        if block_name not in self.blocks:
            return None
        
        block = self.blocks[block_name]
        features = {
            'name': block_name,
            'entities': [],
            'entity_count': 0,
            'entity_types': set(),
            'layers': set()
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
        
        # 计算边界框
        bbox = temp_composite.get_bounding_box()
        if bbox is None:
            features.update({
                'bounding_box': None,
                'width': None,
                'height': None,
                'center': None,
                'aspect_ratio': None
            })
        else:
            features.update({
                'bounding_box': bbox,
                'width': bbox[1][0] - bbox[0][0],
                'height': bbox[1][1] - bbox[0][1],
                'center': (
                    (bbox[0][0] + bbox[1][0]) / 2,
                    (bbox[0][1] + bbox[1][1]) / 2
                ),
                'aspect_ratio': (bbox[1][0] - bbox[0][0]) / (bbox[1][1] - bbox[0][1])
                if bbox[1][1] - bbox[0][1] != 0 else None
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
                if pattern is not None:  # 只添加有效的模式
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
    
if __name__ == "__main__":
    # 1. 从源文件提取所有block的模式
    source_network = EntityNetwork("单元素研究_阀门/单个模块-一个阀门-在0图层.dxf")
    block_patterns = source_network.extract_block_patterns()
    
    if block_patterns:
        print(f"提取了 {len(block_patterns)} 个block模式:")
        for pattern in block_patterns:
            print(f"\nBlock '{pattern.name}' 的特征:")
            print(f"- 实体数量: {pattern.entity_count}")
            print(f"- 实体类型: {pattern.entity_types}")
            print(f"- 宽度范围: {pattern.width_range}")
            print(f"- 高度范围: {pattern.height_range}")
            print(f"- 宽高比范围: {pattern.aspect_ratio_range}")
            print(f"- 图层: {pattern.layers}")
        
        # 2. 在目标文件中查找匹配的 blocks
        target_network = EntityNetwork("Drawing1.dxf")
        all_matching_blocks = []
        
        for pattern in block_patterns:
            matching_blocks = target_network.find_matching_blocks(pattern)
            all_matching_blocks.extend(matching_blocks)
        
        print(f"\n在目标文件中找到 {len(all_matching_blocks)} 个匹配的 blocks:")
        for block in all_matching_blocks:
            print(f"- 位置: {block.position}")
            print(f"- 旋转角度: {block.rotation}")
            print(f"- 缩放: {block.scale}")
    else:
        print("未能提取到任何block模式")