import logging
import os
from datetime import datetime
import ezdxf
from ezdxf.math import Vec3
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Type, Any
import json
import uuid
import math
from collections import defaultdict

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
log_file = os.path.join(log_dir, f"entity_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

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
        """检查特征是否匹配模式"""
        # 检查基本属性
        if features['entity_count'] != self.entity_count:
            return False
            
        # 检查实体类型
        if not self.entity_types.issubset(features['entity_types']):
            return False
            
        # 检查图层 - 放宽限制，只要有一个图层匹配即可，或者完全忽略图层检查
        # if not any(layer in self.layers for layer in features['layers']):
        #     return False
            
        # 检查尺寸
        width = features['width']
        if width is not None and not (self.width_range[0] <= width <= self.width_range[1]):
            return False
            
        height = features['height']
        if height is not None and not (self.height_range[0] <= height <= self.height_range[1]):
            return False
            
        return True

class EntityNetwork:
    """实体网络类，用于管理实体之间的连接关系"""
    def __init__(self, filename: str, tolerances: dict = None):
        """初始化实体网络
        
        Args:
            filename: DXF文件路径
            tolerances: 容差配置字典，包含以下可选项：
                - point_tolerance: 点距离容差 (默认 0.1)
                - count_tolerance: 实体数量容差 (默认 1)
                - size_tolerance: 尺寸容差百分比 (默认 0.2)
                - ratio_tolerance: 宽高比容差百分比 (默认 0.2)
        """
        self.filename = filename
        # 设置默认容差
        self.tolerances = {
            'point_tolerance': 0.1,    # 点距离容差
            'count_tolerance': 1,      # 实体数量容差
            'size_tolerance': 0.2,     # 尺寸容差（20%）
            'ratio_tolerance': 0.2,    # 宽高比容差（20%）
        }
        # 更新用户自定义的容差
        if tolerances:
            self.tolerances.update(tolerances)
            
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"DXF file not found: {filename}")
                
            self.doc = ezdxf.readfile(filename)
            self.msp = self.doc.modelspace()
            self.entities = []
            self.connections = {}
            self.composite_entities = {}
            self.blocks = {}
            self.logger = logging.getLogger(__name__)
            self.load_entities()
            
        except ezdxf.DXFError as e:
            logger.error(f"Failed to read DXF file {filename}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def load_entities(self):
        """从DXF文件加载所有实体"""
        try:
            # 加载所有 blocks
            for block in self.doc.blocks:
                self.blocks[block.name] = block
                logger.debug(f"加载 Block: {block.name}, 包含 {len(list(block))} 个实体")

            total_entities = 0
            entity_types_count = defaultdict(int)
            layer_counts = defaultdict(int)

            for entity in self.msp:
                try:
                    entity_type = entity.dxftype()
                    layer = entity.dxf.layer
                    entity_types_count[entity_type] += 1
                    layer_counts[layer] += 1

                    if entity_type == 'INSERT':
                        entity_info = self._create_insert_entity(entity)
                        logger.debug(
                            f"处理 INSERT 实体: Block={entity_info.block_name}, "
                            f"位置={entity_info.position}, "
                            f"旋转={entity_info.rotation:.2f}°, "
                            f"缩放={entity_info.scale}"
                        )
                    else:
                        entity_info = self._create_basic_entity(entity)
                        if entity_type == 'LINE':
                            logger.debug(
                                f"处理直线: 图层={layer}, "
                                f"起点={tuple(entity.dxf.start)}, "
                                f"终点={tuple(entity.dxf.end)}"
                            )
                        elif entity_type == 'CIRCLE':
                            logger.debug(
                                f"处理圆: 图层={layer}, "
                                f"圆心={tuple(entity.dxf.center)}, "
                                f"半径={entity.dxf.radius:.2f}"
                            )
                            
                    self.entities.append(entity_info)
                    self.update_connections(entity_info)
                    total_entities += 1
                    
                except Exception as e:
                    logger.warning(
                        f"处理实体失败: 类型={entity.dxftype()}, "
                        f"图层={entity.dxf.layer}, "
                        f"错误={str(e)}"
                    )
                    continue
                    
            # 输出统计信息
            logger.info("实体加载完成:")
            logger.info(f"- 总实体数: {total_entities}")
            logger.info("- 实体类型统计:")
            for etype, count in entity_types_count.items():
                logger.info(f"  - {etype}: {count}")
            logger.info("- 图层统计:")
            for layer, count in layer_counts.items():
                logger.info(f"  - {layer}: {count}")
            
        except Exception as e:
            logger.error(f"实体加载失败: {str(e)}")
            raise

    def _create_insert_entity(self, entity) -> EntityInfo:
        """创建INSERT类型的实体信息"""
        try:
            return EntityInfo(
                dxf_entity=entity,
                entity_type='INSERT',
                layer=entity.dxf.layer,
                block_name=entity.dxf.name,
                rotation=getattr(entity.dxf, 'rotation', 0.0),
                scale=(
                    getattr(entity.dxf, 'xscale', 1.0),
                    getattr(entity.dxf, 'yscale', 1.0),
                    getattr(entity.dxf, 'zscale', 1.0)
                ),
                position=tuple(entity.dxf.insert)
            )
        except Exception as e:
            logger.error(f"Failed to create INSERT entity: {str(e)}")
            raise

    def _create_basic_entity(self, entity) -> EntityInfo:
        """创建基本实体信息"""
        try:
            return EntityInfo(
                dxf_entity=entity,
                entity_type=entity.dxftype(),
                layer=entity.dxf.layer
            )
        except Exception as e:
            logger.error(f"Failed to create basic entity: {str(e)}")
            raise

    def update_connections(self, entity=None):
        """更新实体之间的连接关系
        
        Args:
            entity: 可选，如果提供则只更新与该实体相关的连接
        """
        try:
            # 如果没有实体，直接返回
            if not self.entities:
                return
            
            # 如果是单个实体更新
            if entity:
                # 避免重复更新相同的实体
                if hasattr(self, '_last_updated') and entity.id in self._last_updated:
                    return
                    
                self.connections.pop(entity.id, None)  # 移除旧的连接
                for other in self.entities:
                    if other.id == entity.id:
                        continue
                    
                    # 获取端点
                    points1 = self._get_entity_endpoints(entity)
                    points2 = self._get_entity_endpoints(other)
                    
                    # 检查端点是否接近
                    for p1 in points1:
                        for p2 in points2:
                            if self._points_are_close(p1, p2):
                                # 添加双向连接
                                self.connections.setdefault(entity.id, set()).add(other.id)
                                self.connections.setdefault(other.id, set()).add(entity.id)
                                break  # 找到一个连接点就足够了
                    if entity.id in self.connections:
                        break
                        
                # 记录已更新的实体
                if not hasattr(self, '_last_updated'):
                    self._last_updated = set()
                self._last_updated.add(entity.id)
                
            else:
                # 更新所有连接
                self.connections.clear()
                processed = set()
                
                # 遍历所有实体
                for i, e1 in enumerate(self.entities):
                    for j, e2 in enumerate(self.entities[i+1:], i+1):
                        if (e1.id, e2.id) in processed:
                            continue
                        
                        # 获取实体的端点
                        points1 = self._get_entity_endpoints(e1)
                        points2 = self._get_entity_endpoints(e2)
                        
                        # 检查端点是否接近
                        connected = False
                        for p1 in points1:
                            for p2 in points2:
                                if self._points_are_close(p1, p2):
                                    # 添加双向连接
                                    self.connections.setdefault(e1.id, set()).add(e2.id)
                                    self.connections.setdefault(e2.id, set()).add(e1.id)
                                    connected = True
                                    break
                            if connected:
                                break
                            
                        processed.add((e1.id, e2.id))
                        processed.add((e2.id, e1.id))
                
                # 重置更新记录
                self._last_updated = set()
            
            if len(self.connections) > 0:
                logger.debug(f"更新了 {len(self.connections)} 个实体的连接关系")
            
        except Exception as e:
            logger.error(f"更新连接关系失败: {str(e)}")

    def _get_entity_endpoints(self, entity: EntityInfo) -> List[tuple]:
        """获取实体的端点坐标
        
        Args:
            entity: 实体信息对象
            
        Returns:
            List[tuple]: 端点坐标列表
        """
        try:
            points = []
            e = entity.dxf_entity
            
            if entity.entity_type == 'LINE':
                points.extend([
                    (e.dxf.start[0], e.dxf.start[1]),
                    (e.dxf.end[0], e.dxf.end[1])
                ])
            elif entity.entity_type == 'CIRCLE':
                points.append((e.dxf.center[0], e.dxf.center[1]))
            elif entity.entity_type == 'ARC':
                points.extend([
                    (e.start_point[0], e.start_point[1]),
                    (e.end_point[0], e.end_point[1])
                ])
            elif entity.entity_type == 'POLYLINE':
                points.extend([
                    (vertex[0], vertex[1])
                    for vertex in e.vertices()
                ])
            elif entity.entity_type == 'INSERT':
                points.append((e.dxf.insert[0], e.dxf.insert[1]))
                
            return points
            
        except Exception as e:
            logger.error(f"获取实体端点失败: {str(e)}")
            return []

    def _points_are_close(self, p1: tuple, p2: tuple, tolerance: float = None) -> bool:
        """检查两点是否足够接近
        
        Args:
            p1: 第一个点坐标
            p2: 第二个点坐标
            tolerance: 容差值，如果不指定则使用配置中的 point_tolerance
            
        Returns:
            bool: 如果两点距离在容差范围内返回True
        """
        try:
            if tolerance is None:
                tolerance = self.tolerances['point_tolerance']
                
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            distance = math.sqrt(dx*dx + dy*dy)
            return distance <= tolerance
            
        except Exception as e:
            logger.debug(f"计算点距离时出错: {str(e)}")
            return False

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
        """根据模式查找复合实体"""
        try:
            result = []
            visited = set()
            
            logger.info(f"开始查找模式: {pattern}")
            
            for entity in self.entities:
                if entity in visited:
                    continue
                    
                try:
                    connected = self._get_connected_components(entity, visited)
                    if self._match_pattern(connected, pattern):
                        composite = CompositeEntity(
                            name=f"Composite_{len(result)}",
                            entities=list(connected)
                        )
                        result.append(composite)
                        
                        # 记录匹配成功的信息
                        bbox = composite.get_bounding_box()
                        geometry = self._calculate_geometry(bbox)
                        logger.info(
                            f"找到匹配组: ID={composite.name}, "
                            f"实体数={len(connected)}, "
                            f"类型={set(e.entity_type for e in connected)}, "
                            f"尺寸={geometry['width']:.2f}x{geometry['height']:.2f}"
                        )
                        
                except Exception as e:
                    logger.warning(
                        f"处理实体组失败: 起始实体={entity.entity_type}({entity.id}), "
                        f"图层={entity.layer}, "
                        f"错误={str(e)}"
                    )
                    continue
            
            logger.info(f"模式匹配完成: 找到 {len(result)} 个匹配组")
            return result
            
        except Exception as e:
            logger.error(f"模式匹配失败: {str(e)}")
            return []

    def _get_connected_components(self, start_entity: EntityInfo, visited: Set) -> Set[EntityInfo]:
        """获取与起始实体相连的所有实体"""
        try:
            connected = set()
            to_visit = [start_entity]
            
            while to_visit:
                current = to_visit.pop()
                if current not in connected:
                    connected.add(current)
                    visited.add(current)
                    
                    # 获取相连的实体
                    connected_ids = self.connections.get(current.id, set())
                    neighbors = [
                        e for e in self.entities 
                        if e.id in connected_ids and e not in connected
                    ]
                    to_visit.extend(neighbors)
            
            return connected
            
        except Exception as e:
            logger.error(f"Failed to get connected components: {str(e)}")
            return set()
    
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
            geometry = self._calculate_geometry(bbox)
            
            if (geometry['width'] is None or 
                geometry['height'] is None or 
                geometry['width'] > pattern['max_size'][0] or 
                geometry['height'] > pattern['max_size'][1]):
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
    
    def get_block_features(self, block_name: str) -> Optional[Dict[str, Any]]:
        """获取指定 block 的特征信息"""
        try:
            if not block_name:
                logger.warning("Block name is empty")
                return None
                
            if block_name not in self.blocks:
                logger.warning(f"Block '{block_name}' not found in drawing")
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
            
            # 遍历 block 中的实体
            for entity in block:
                try:
                    entity_info = EntityInfo(
                        dxf_entity=entity,
                        entity_type=entity.dxftype(),
                        layer=getattr(entity.dxf, 'layer', 'UNKNOWN')  # 安全获取图层
                    )
                    temp_composite.add_entity(entity_info)
                    
                    # 收集实体信息
                    entity_features = self._get_safe_entity_info(entity_info)
                    if entity_features:
                        features['entities'].append(entity_features)
                        features['entity_types'].add(entity.dxftype())
                        features['layers'].add(entity_features['layer'])
                        features['entity_count'] += 1
                        
                except AttributeError as e:
                    logger.warning(f"Invalid entity in block '{block_name}': {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing entity in block '{block_name}': {str(e)}")
                    continue
            
            # 计算边界框
            bbox = temp_composite.get_bounding_box()
            geometry = self._calculate_geometry(bbox)
            features.update(geometry)
            
            return features
            
        except Exception as e:
            logger.error(f"获取block特征失败: {str(e)}")
            return None
    
    def _get_safe_entity_info(self, entity_info: EntityInfo) -> Optional[Dict[str, Any]]:
        """安全地获取实体的详细信息"""
        try:
            entity = entity_info.dxf_entity
            info = {
                'type': entity_info.entity_type,
                'layer': entity_info.layer,
            }
            
            if entity_info.entity_type == 'INSERT':
                info.update({
                    'block_name': entity_info.block_name,
                    'rotation': getattr(entity_info, 'rotation', 0.0),
                    'scale': getattr(entity_info, 'scale', (1.0, 1.0, 1.0)),
                    'position': getattr(entity_info, 'position', (0.0, 0.0, 0.0)),
                })
                # 安全地获取 block 特征
                block_features = self.get_block_features(entity_info.block_name)
                if block_features:
                    info['block_features'] = block_features
                
            elif entity_info.entity_type == 'LINE':
                info.update({
                    'start': tuple(getattr(entity.dxf, 'start', (0, 0, 0))),
                    'end': tuple(getattr(entity.dxf, 'end', (0, 0, 0))),
                })
                
            elif entity_info.entity_type == 'CIRCLE':
                info.update({
                    'center': tuple(getattr(entity.dxf, 'center', (0, 0, 0))),
                    'radius': getattr(entity.dxf, 'radius', 0.0),
                })
                
            elif entity_info.entity_type == 'TEXT':
                info.update({
                    'text': getattr(entity.dxf, 'text', ''),
                    'position': tuple(getattr(entity.dxf, 'insert', (0, 0, 0))),
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting entity info for {entity_info.entity_type}: {str(e)}")
            return None
    
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
        
        print(f"\n正在查找匹配模式: {pattern.name}")
        print(f"模式特征:")
        print(f"- 实体数量: {pattern.entity_count}")
        print(f"- 实体类型: {pattern.entity_types}")
        print(f"- 宽度范围: {pattern.width_range}")
        print(f"- 高度范围: {pattern.height_range}")
        print(f"- 图层: {pattern.layers}")
        
        for entity in self.entities:
            if entity.entity_type == 'INSERT':
                print(f"\n检查 block: {entity.block_name}")
                
                # 获取 block 定义
                block = self.blocks.get(entity.block_name)
                if not block:
                    print(f"- 未找到 block 定义")
                    continue
                    
                # 创建临时的 CompositeEntity 来存储 block 中的实体
                temp_composite = CompositeEntity(name=entity.block_name)
                
                # 收集 block 内部的实体信息
                block_features = {
                    'entity_count': 0,
                    'entity_types': set(),
                    'layers': set()
                }
                
                # 遍历 block 内部的实体
                for block_entity in block:
                    entity_info = EntityInfo(
                        dxf_entity=block_entity,
                        entity_type=block_entity.dxftype(),
                        layer=block_entity.dxf.layer
                    )
                    temp_composite.add_entity(entity_info)
                    
                    # 更新特征
                    block_features['entity_count'] += 1
                    block_features['entity_types'].add(block_entity.dxftype())
                    block_features['layers'].add(block_entity.dxf.layer)
                
                print(f"- 实体数量: {block_features['entity_count']}")
                print(f"- 实体类型: {block_features['entity_types']}")
                print(f"- 图层: {block_features['layers']}")
                
                # 计算边界框和其他几何特征
                bbox = temp_composite.get_bounding_box()
                if bbox:
                    width = bbox[1][0] - bbox[0][0]
                    height = bbox[1][1] - bbox[0][1]
                    block_features.update({
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height if height != 0 else None
                    })
                    
                    # 应用缩放因子
                    block_features['width'] *= entity.scale[0]
                    block_features['height'] *= entity.scale[1]
                    
                    print(f"- 宽度: {block_features['width']}")
                    print(f"- 高度: {block_features['height']}")
                    print(f"- 宽高比: {block_features['aspect_ratio']}")
                    
                    # 检查是否匹配模式
                    if pattern.matches(block_features):
                        print("✓ 匹配成功!")
                        matching_blocks.append(entity)
                    else:
                        print("✗ 匹配失败")
                else:
                    print("- 无法计算边界框")
        
        print(f"\n找到 {len(matching_blocks)} 个匹配的 blocks")
        return matching_blocks
    
    def find_similar_entity_groups(self, pattern: BlockPattern, tolerance: float = 0.1) -> List[List[EntityInfo]]:
        """查找与模式相似的实体组"""
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
    
    def _match_entity_group_to_pattern(self, entities: List[EntityInfo], pattern: BlockPattern, tolerance: float) -> bool:
        """检查一组实体是否匹配指定的模式"""
        # 创建临时的复合实体来计算特征
        temp_composite = CompositeEntity(name="temp")
        for entity in entities:
            temp_composite.add_entity(entity)
            
        # 获取这组实体的特征
        bbox = temp_composite.get_bounding_box()
        if bbox is None:
            return False
            
        # 计算特征
        width = bbox[1][0] - bbox[0][0]
        height = bbox[1][1] - bbox[0][1]
        aspect_ratio = width / height if height != 0 else None
        
        # 检查实体数量
        if len(entities) != pattern.entity_count:
            return False
            
        # 检查实体类型
        entity_types = {e.entity_type for e in entities}
        if not pattern.entity_types.issubset(entity_types):
            return False
            
        # 检查尺寸
        if not (pattern.width_range[0] <= width <= pattern.width_range[1]):
            return False
            
        if not (pattern.height_range[0] <= height <= pattern.height_range[1]):
            return False
            
        # 检查宽高比
        if aspect_ratio is not None and pattern.aspect_ratio_range[0] is not None:
            if not (pattern.aspect_ratio_range[0] <= aspect_ratio <= pattern.aspect_ratio_range[1]):
                return False
                
        # 检查拓扑结构（可选，根据需要添加更多的拓扑检查）
        if not self._check_topology(entities, pattern):
            return False
            
        return True
    

    def _calculate_geometry(self, bbox: Optional[tuple]) -> Dict[str, Optional[float]]:
        """计算几何特征
        
        Args:
            bbox: 边界框，格式为 ((min_x, min_y), (max_x, max_y))
            
        Returns:
            包含几何特征的字典：
            - width: 宽度
            - height: 高度
            - center: 中心点 (x, y)
            - aspect_ratio: 宽高比
        """
        if not bbox:
            return {
                'width': None,
                'height': None,
                'center': None,
                'aspect_ratio': None
            }
        
        try:
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]
            center = (
                (bbox[0][0] + bbox[1][0]) / 2,
                (bbox[0][1] + bbox[1][1]) / 2
            )
            aspect_ratio = width / height if abs(height) > 1e-10 else None
            
            return {
                'width': width,
                'height': height,
                'center': center,
                'aspect_ratio': aspect_ratio
            }
        except Exception as e:
            logger.error(f"几何特征计算失败: {str(e)}")
            return {
                'width': None,
                'height': None,
                'center': None,
                'aspect_ratio': None
            }

    def _check_topology(self, entities: List[EntityInfo], pattern: 'BlockPattern') -> bool:
        """检查实体组的拓扑结构是否与模式匹配
        
        Args:
            entities: 要检查的实体列表 (EntityInfo对象列表)
            pattern: 要匹配的模式
            
        Returns:
            bool: 如果拓扑结构匹配返回True，否则返回False
        """
        try:
            if not entities or not pattern:
                return False
            
            composite = CompositeEntity("temp", list(entities))
            bbox = composite.get_bounding_box()
            if not bbox:
                return False
            
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]
            
            # 使用配置的容差进行检查
            entity_types = defaultdict(int)
            for entity in entities:
                entity_types[entity.entity_type] += 1
                
            # 检查实体数量
            if abs(len(entities) - pattern.entity_count) > self.tolerances['count_tolerance']:
                return False
                
            # 检查实体类型
            main_types = {t for t, count in entity_types.items() if count > 1}
            if not any(t in main_types for t in pattern.entity_types):
                return False
                
            # 检查尺寸
            size_tolerance = self.tolerances['size_tolerance']
            width_tolerance = (pattern.width_range[1] - pattern.width_range[0]) * size_tolerance
            height_tolerance = (pattern.height_range[1] - pattern.height_range[0]) * size_tolerance
            
            if not (pattern.width_range[0] - width_tolerance <= width <= pattern.width_range[1] + width_tolerance):
                return False
                
            if not (pattern.height_range[0] - height_tolerance <= height <= pattern.height_range[1] + height_tolerance):
                return False
                
            # 检查宽高比
            if pattern.aspect_ratio_range:
                aspect_ratio = width / height if height != 0 else float('inf')
                ratio_tolerance = (pattern.aspect_ratio_range[1] - pattern.aspect_ratio_range[0]) * self.tolerances['ratio_tolerance']
                if not (pattern.aspect_ratio_range[0] - ratio_tolerance <= aspect_ratio <= pattern.aspect_ratio_range[1] + ratio_tolerance):
                    return False
                    
            # 检查连接关系
            connections = self._find_connections(entities)
            if len(connections) < 1:
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"检查拓扑结构时出错: {str(e)}")
            return False

    def _find_connections(self, entities: List[EntityInfo]) -> List[tuple]:
        """查找实体之间的连接点"""
        try:
            connections = []
            processed = set()
            
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities[i+1:], i+1):
                    if (e1.id, e2.id) in processed:
                        continue
                    
                    # 获取端点
                    points1 = self._get_entity_endpoints(e1)
                    points2 = self._get_entity_endpoints(e2)
                    
                    # 查找接近的点（增加容差）
                    for p1 in points1:
                        for p2 in points2:
                            if self._points_are_close(p1, p2, tolerance=0.5):  # 增加容差到0.5
                                connections.append(((p1[0] + p2[0])/2, (p1[1] + p2[1])/2))
                                
                    processed.add((e1.id, e2.id))
                    processed.add((e2.id, e1.id))
                
            return connections
            
        except Exception as e:
            logger.debug(f"查找连接点时出错: {str(e)}")
            return []

    def _get_entity_points(self, entity: Any) -> List[tuple]:
        """获取实体的特征点（端点、中心点等）
        
        Args:
            entity: DXF实体
            
        Returns:
            List[tuple]: 点坐标列表
        """
        try:
            points = []
            
            if hasattr(entity, 'start_point'):
                points.append((entity.start_point[0], entity.start_point[1]))
            if hasattr(entity, 'end_point'):
                points.append((entity.end_point[0], entity.end_point[1]))
            if hasattr(entity, 'center'):
                points.append((entity.center[0], entity.center[1]))
            
            return points
            
        except Exception as e:
            logger.error(f"获取实体点时出错: {str(e)}")
            return []

if __name__ == "__main__":
    # 1. 从源文件提取block模式
    source_network = EntityNetwork("单元素研究_阀门/单个模块-一个阀门-在0图层.dxf")
    block_patterns = source_network.extract_block_patterns()
    
    print("\n源文件中的模式:")
    for pattern in block_patterns:
        print(f"\nBlock模式 '{pattern.name}' 的特征:")
        print(f"- 实体数量: {pattern.entity_count}")
        print(f"- 实体类型: {pattern.entity_types}")
        print(f"- 宽度范围: {pattern.width_range}")
        print(f"- 高度范围: {pattern.height_range}")
        print(f"- 宽高比范围: {pattern.aspect_ratio_range}")
        print(f"- 图层: {pattern.layers}")
        print("-" * 50)
        
        # 2. 在目标文件中查找匹配的 blocks
        # target_network = EntityNetwork("图例和流程图_仪表管件设备均为模块/2308PM-04-T3-2429.dxf")
        # target_network = EntityNetwork("图例和流程图_仪表管件设备均为普通线条/2308PM-04-T3-2429.dxf")
        target_network = EntityNetwork("Drawing1.dxf")
        all_matching_groups = []
        
        for pattern in block_patterns:
            # 查找block引用
            matching_blocks = target_network.find_matching_blocks(pattern)
            all_matching_groups.extend([(block,) for block in matching_blocks])
            
            # 查找相似的实体组
            similar_groups = target_network.find_similar_entity_groups(pattern)
            all_matching_groups.extend(similar_groups)
        
        print(f"\n找到 {len(all_matching_groups)} 个匹配项:")
        for group in all_matching_groups:
            if len(group) == 1 and group[0].entity_type == 'INSERT':
                block = group[0]
                print(f"- Block引用: 位置={block.position}, 旋转={block.rotation}")
            else:
                bbox = CompositeEntity("temp", list(group)).get_bounding_box()
                center = (
                    (bbox[0][0] + bbox[1][0]) / 2,
                    (bbox[0][1] + bbox[1][1]) / 2
                )
                print(f"- 实体组: 中心位置={center}, 实体数量={len(group)}")
    else:
        print("未能提取到任何block模式")