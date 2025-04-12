"""
DXF文件解析模块
提供DXF文件格式的解析功能，不直接依赖ezdxf
支持多种DXF解析后端
"""

import os
import math
from typing import List, Tuple, Dict, Any, Optional, Set
import uuid

from src.core.data_structures import (
    Point, BoundingBox, EntityType, Entity, LineEntity, 
    CircleEntity, ArcEntity, TextEntity, Block, 
    BlockReference, AttributeInfo, PolylineEntity, LwPolylineEntity
)
from src.parsers.parser_interface import CADFileParser


class DXFParseError(Exception):
    """DXF解析错误"""
    pass


class DXFBackendBase:
    """DXF解析后端基类"""
    
    def __init__(self):
        self.doc = None
        self.modelspace = None
        self.blocks_dict = {}
    
    def load_file(self, file_path: str) -> bool:
        """加载DXF文件"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_entities(self) -> List[Dict]:
        """获取模型空间中的所有实体"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_blocks(self) -> Dict[str, Dict]:
        """获取所有块定义"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_block_entities(self, block_name: str) -> List[Dict]:
        """获取块中的实体"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_block_inserts(self) -> List[Dict]:
        """获取所有块引用"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_attributes(self, owner_handle: str) -> List[Dict]:
        """获取指定所有者的属性"""
        raise NotImplementedError("子类必须实现此方法")


try:
    import ezdxf
    
    class EzdxfBackend(DXFBackendBase):
        """ezdxf解析后端"""
        
        def load_file(self, file_path: str) -> bool:
            """加载DXF文件"""
            try:
                self.doc = ezdxf.readfile(file_path)
                self.modelspace = self.doc.modelspace()
                
                # 加载块
                self.blocks_dict = {}
                for block in self.doc.blocks:
                    self.blocks_dict[block.name] = block
                
                return True
            except Exception as e:
                raise DXFParseError(f"Failed to load DXF file: {e}")
        
        def get_entities(self) -> List[Dict]:
            """获取模型空间中的所有实体"""
            if not self.modelspace:
                return []
            
            entities = []
            for entity in self.modelspace:
                entity_dict = self._convert_entity_to_dict(entity)
                if entity_dict:
                    entities.append(entity_dict)
            
            return entities
        
        def get_blocks(self) -> Dict[str, Dict]:
            """获取所有块定义"""
            if not self.doc:
                return {}
            
            blocks = {}
            for block in self.doc.blocks:
                blocks[block.name] = {
                    'name': block.name,
                    'handle': block.dxf.handle,
                    'base_point': (0, 0, 0) if not hasattr(block.dxf, 'base_point') else tuple(block.dxf.base_point)
                }
            
            return blocks
        
        def get_block_entities(self, block_name: str) -> List[Dict]:
            """获取块中的实体"""
            if block_name not in self.blocks_dict:
                return []
            
            block = self.blocks_dict[block_name]
            entities = []
            
            for entity in block:
                if entity.dxftype() not in ('ATTDEF', 'SEQEND'):
                    entity_dict = self._convert_entity_to_dict(entity)
                    if entity_dict:
                        entities.append(entity_dict)
            
            return entities
        
        def get_block_inserts(self) -> List[Dict]:
            """获取所有块引用"""
            if not self.modelspace:
                return []
            
            inserts = []
            for insert in self.modelspace.query('INSERT'):
                insert_dict = {
                    'type': 'INSERT',
                    'handle': insert.dxf.handle,
                    'layer': insert.dxf.layer,
                    'block_name': insert.dxf.name,
                    'position': (insert.dxf.insert.x, insert.dxf.insert.y, insert.dxf.insert.z),
                    'scale': (insert.dxf.xscale, insert.dxf.yscale, insert.dxf.zscale),
                    'rotation': insert.dxf.rotation
                }
                inserts.append(insert_dict)
            
            return inserts
        
        def get_attributes(self, owner_handle: str) -> List[Dict]:
            """获取指定所有者的属性"""
            attributes = []
            
            for entity in self.modelspace:
                if (entity.dxftype() == 'ATTRIB' and 
                    hasattr(entity.dxf, 'owner') and 
                    entity.dxf.owner == owner_handle):
                    
                    attr_dict = {
                        'type': 'ATTRIB',
                        'handle': entity.dxf.handle,
                        'tag': entity.dxf.tag,
                        'text': entity.dxf.text,
                        'position': tuple(entity.dxf.insert),
                        'height': entity.dxf.height,
                        'rotation': entity.dxf.rotation,
                        'layer': entity.dxf.layer,
                        'style': entity.dxf.style
                    }
                    attributes.append(attr_dict)
            
            return attributes
        
        def _convert_entity_to_dict(self, entity) -> Dict:
            """将ezdxf实体转换为字典"""
            entity_type = entity.dxftype()
            
            try:
                if entity_type == 'LINE':
                    return {
                        'type': 'LINE',
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'start': tuple(entity.dxf.start),
                        'end': tuple(entity.dxf.end)
                    }
                elif entity_type == 'CIRCLE':
                    return {
                        'type': 'CIRCLE',
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'center': tuple(entity.dxf.center),
                        'radius': entity.dxf.radius
                    }
                elif entity_type == 'ARC':
                    return {
                        'type': 'ARC',
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'center': tuple(entity.dxf.center),
                        'radius': entity.dxf.radius,
                        'start_angle': entity.dxf.start_angle,
                        'end_angle': entity.dxf.end_angle
                    }
                elif entity_type in ('TEXT', 'MTEXT'):
                    return {
                        'type': entity_type,
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'text': entity.dxf.text,
                        'position': tuple(entity.dxf.insert),
                        'height': entity.dxf.height,
                        'rotation': entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0.0
                    }
                elif entity_type in ('LWPOLYLINE', 'POLYLINE'):
                    points = []
                    if entity_type == 'LWPOLYLINE':
                        points = [
                            (p[0], p[1], p[2] if len(p) > 2 else 0.0)
                            for p in entity.get_points()
                            if isinstance(p, (list, tuple)) and len(p) >= 2
                        ]
                    else:  # POLYLINE
                        points = [tuple(p) for p in entity.points()]
                    
                    return {
                        'type': entity_type,
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'points': points,
                        'closed': entity.is_closed if hasattr(entity, 'is_closed') else False
                    }
                elif entity_type == 'INSERT':
                    return {
                        'type': 'INSERT',
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer,
                        'block_name': entity.dxf.name,
                        'position': tuple(entity.dxf.insert),
                        'scale': (entity.dxf.xscale, entity.dxf.yscale, entity.dxf.zscale),
                        'rotation': entity.dxf.rotation
                    }
                else:
                    # 基本信息
                    return {
                        'type': entity_type,
                        'handle': entity.dxf.handle,
                        'layer': entity.dxf.layer
                    }
            except Exception as e:
                print(f"Error converting entity {entity_type}: {e}")
                return {}

except ImportError:
    # ezdxf不可用时的替代方案
    pass


# 尝试导入其他可能的DXF解析库
try:
    # 示例：使用dxfgrabber作为备选
    import dxfgrabber
    
    class DxfGrabberBackend(DXFBackendBase):
        """dxfgrabber解析后端"""
        
        def load_file(self, file_path: str) -> bool:
            """加载DXF文件"""
            try:
                self.doc = dxfgrabber.readfile(file_path)
                self.modelspace = [e for e in self.doc.entities]
                
                # 加载块
                self.blocks_dict = {}
                for block in self.doc.blocks:
                    self.blocks_dict[block.name] = block
                
                return True
            except Exception as e:
                raise DXFParseError(f"Failed to load DXF file: {e}")
        
        # 其他方法的实现...
        # 这些实现应该适配dxfgrabber库的接口

except ImportError:
    # dxfgrabber不可用
    pass


class SimpleDXFParser:
    """简单的DXF解析器（不依赖第三方库）"""
    
    def __init__(self):
        self.entities = []
        self.blocks = {}
        self.inserts = []
    
    def load_file(self, file_path: str) -> bool:
        """加载DXF文件"""
        try:
            with open(file_path, 'r') as f:
                contents = f.readlines()
            
            # 这里实现一个非常基础的DXF解析
            # 真正的实现会更复杂，这里只是一个框架
            
            i = 0
            while i < len(contents):
                line = contents[i].strip()
                
                if line == "ENTITIES":
                    i = self._parse_entities_section(contents, i + 1)
                elif line == "BLOCKS":
                    i = self._parse_blocks_section(contents, i + 1)
                else:
                    i += 1
            
            return True
        
        except Exception as e:
            raise DXFParseError(f"Failed to load DXF file: {e}")
    
    def _parse_entities_section(self, contents, start_index):
        """解析ENTITIES部分"""
        i = start_index
        
        while i < len(contents):
            line = contents[i].strip()
            
            if line == "ENDSEC":
                return i + 1
            
            if line in ("LINE", "CIRCLE", "ARC", "TEXT", "INSERT"):
                entity_type = line
                entity = {"type": entity_type}
                
                # 解析实体属性
                i = self._parse_entity_attributes(contents, i + 1, entity)
                
                if entity_type == "INSERT":
                    self.inserts.append(entity)
                else:
                    self.entities.append(entity)
            else:
                i += 1
        
        return i
    
    def _parse_blocks_section(self, contents, start_index):
        """解析BLOCKS部分"""
        i = start_index
        current_block = None
        
        while i < len(contents):
            line = contents[i].strip()
            
            if line == "ENDSEC":
                return i + 1
            
            if line == "BLOCK":
                # 开始一个新块
                current_block = {"entities": []}
                i = self._parse_block_header(contents, i + 1, current_block)
            elif line == "ENDBLK":
                # 结束当前块
                if current_block and "name" in current_block:
                    self.blocks[current_block["name"]] = current_block
                current_block = None
                i += 1
            elif current_block is not None and line in ("LINE", "CIRCLE", "ARC", "TEXT"):
                # 块内的实体
                entity_type = line
                entity = {"type": entity_type}
                
                # 解析实体属性
                i = self._parse_entity_attributes(contents, i + 1, entity)
                
                current_block["entities"].append(entity)
            else:
                i += 1
        
        return i
    
    def _parse_entity_attributes(self, contents, start_index, entity):
        """解析实体属性"""
        i = start_index
        
        while i < len(contents):
            line = contents[i].strip()
            
            if line in ("LINE", "CIRCLE", "ARC", "TEXT", "INSERT", "BLOCK", "ENDBLK", "ENDSEC"):
                return i
            
            # 解析组码和值
            if i + 1 < len(contents):
                group_code = int(line)
                value = contents[i + 1].strip()
                
                # 根据组码设置属性
                self._set_entity_attribute(entity, group_code, value)
                
                i += 2
            else:
                i += 1
        
        return i
    
    def _parse_block_header(self, contents, start_index, block):
        """解析块头部"""
        i = start_index
        
        while i < len(contents):
            line = contents[i].strip()
            
            if line in ("LINE", "CIRCLE", "ARC", "TEXT", "INSERT", "ENDBLK", "ENDSEC"):
                return i
            
            # 解析组码和值
            if i + 1 < len(contents):
                group_code = int(line)
                value = contents[i + 1].strip()
                
                # 设置块属性
                if group_code == 2:
                    block["name"] = value
                elif group_code == 10:
                    block["x"] = float(value)
                elif group_code == 20:
                    block["y"] = float(value)
                elif group_code == 30:
                    block["z"] = float(value)
                
                i += 2
            else:
                i += 1
        
        return i
    
    def _set_entity_attribute(self, entity, group_code, value):
        """根据组码设置实体属性"""
        entity_type = entity["type"]
        
        if group_code == 8:
            entity["layer"] = value
        elif group_code == 5:
            entity["handle"] = value
        
        # LINE实体
        if entity_type == "LINE":
            if group_code == 10:
                entity.setdefault("start", [0, 0, 0])[0] = float(value)
            elif group_code == 20:
                entity.setdefault("start", [0, 0, 0])[1] = float(value)
            elif group_code == 30:
                entity.setdefault("start", [0, 0, 0])[2] = float(value)
            elif group_code == 11:
                entity.setdefault("end", [0, 0, 0])[0] = float(value)
            elif group_code == 21:
                entity.setdefault("end", [0, 0, 0])[1] = float(value)
            elif group_code == 31:
                entity.setdefault("end", [0, 0, 0])[2] = float(value)
        
        # CIRCLE实体
        elif entity_type == "CIRCLE":
            if group_code == 10:
                entity.setdefault("center", [0, 0, 0])[0] = float(value)
            elif group_code == 20:
                entity.setdefault("center", [0, 0, 0])[1] = float(value)
            elif group_code == 30:
                entity.setdefault("center", [0, 0, 0])[2] = float(value)
            elif group_code == 40:
                entity["radius"] = float(value)
        
        # ARC实体
        elif entity_type == "ARC":
            if group_code == 10:
                entity.setdefault("center", [0, 0, 0])[0] = float(value)
            elif group_code == 20:
                entity.setdefault("center", [0, 0, 0])[1] = float(value)
            elif group_code == 30:
                entity.setdefault("center", [0, 0, 0])[2] = float(value)
            elif group_code == 40:
                entity["radius"] = float(value)
            elif group_code == 50:
                entity["start_angle"] = float(value)
            elif group_code == 51:
                entity["end_angle"] = float(value)
        
        # TEXT实体
        elif entity_type == "TEXT":
            if group_code == 10:
                entity.setdefault("position", [0, 0, 0])[0] = float(value)
            elif group_code == 20:
                entity.setdefault("position", [0, 0, 0])[1] = float(value)
            elif group_code == 30:
                entity.setdefault("position", [0, 0, 0])[2] = float(value)
            elif group_code == 1:
                entity["text"] = value
            elif group_code == 40:
                entity["height"] = float(value)
            elif group_code == 50:
                entity["rotation"] = float(value)
        
        # INSERT实体
        elif entity_type == "INSERT":
            if group_code == 2:
                entity["block_name"] = value
            elif group_code == 10:
                entity.setdefault("position", [0, 0, 0])[0] = float(value)
            elif group_code == 20:
                entity.setdefault("position", [0, 0, 0])[1] = float(value)
            elif group_code == 30:
                entity.setdefault("position", [0, 0, 0])[2] = float(value)
            elif group_code == 41:
                entity.setdefault("scale", [1, 1, 1])[0] = float(value)
            elif group_code == 42:
                entity.setdefault("scale", [1, 1, 1])[1] = float(value)
            elif group_code == 43:
                entity.setdefault("scale", [1, 1, 1])[2] = float(value)
            elif group_code == 50:
                entity["rotation"] = float(value)
    
    def get_entities(self) -> List[Dict]:
        """获取解析的实体"""
        return self.entities
    
    def get_blocks(self) -> Dict[str, Dict]:
        """获取解析的块"""
        return self.blocks
    
    def get_inserts(self) -> List[Dict]:
        """获取解析的块引用"""
        return self.inserts


class DXFParser(CADFileParser):
    """DXF文件解析器"""
    
    def __init__(self):
        """初始化DXF解析器"""
        # 尝试使用可用的后端
        self.backend = None
        
        # 先尝试ezdxf
        try:
            import ezdxf
            self.backend = EzdxfBackend()
            print("Using ezdxf backend for DXF parsing")
            return
        except ImportError:
            pass
        
        # 再尝试dxfgrabber
        try:
            import dxfgrabber
            self.backend = DxfGrabberBackend()
            print("Using dxfgrabber backend for DXF parsing")
            return
        except ImportError:
            pass
        
        # 最后使用简单解析器
        self.backend = SimpleDXFParser()
        print("Using simple backend for DXF parsing")
    
    def supports_format(self, file_extension: str) -> bool:
        """检查是否支持指定格式"""
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        return file_extension.lower() in ['dxf']
    
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """解析DXF文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # 使用后端加载文件
            self.backend.load_file(file_path)
            
            # 解析所有实体
            entities = self._parse_entities()
            
            # 解析所有块定义
            blocks = self._parse_blocks()
            
            # 解析所有块引用
            block_instances = self._parse_block_instances()
            
            return entities, block_instances, {"blocks_dict": self.backend.blocks_dict}
        
        except Exception as e:
            raise DXFParseError(f"Error parsing DXF file: {e}")
    
    def _parse_entities(self) -> List[Entity]:
        """解析所有实体"""
        entities = []
        
        entity_dicts = self.backend.get_entities()
        
        for entity_dict in entity_dicts:
            entity = self._create_entity_from_dict(entity_dict)
            if entity:
                entities.append(entity)
        
        return entities
    
    def _parse_blocks(self) -> List[Block]:
        """解析所有块定义"""
        blocks = []
        
        blocks_dict = self.backend.get_blocks()
        
        for block_name, block_dict in blocks_dict.items():
            # 排除一些特殊块
            if block_name.startswith('*') or block_name in ('_ArchTick', '_ARCHTICK'):
                continue
            
            # 获取块中的实体
            entity_dicts = self.backend.get_block_entities(block_name)
            entities = []
            
            for entity_dict in entity_dicts:
                entity = self._create_entity_from_dict(entity_dict)
                if entity:
                    entities.append(entity)
            
            # 创建块
            if entities:
                block = Block(
                    id=block_dict.get('handle', self.generate_unique_id('BLOCK_')),
                    name=block_name,
                    entities=entities,
                    is_arrow=False  # 稍后检查是否为箭头
                )
                
                # 检查是否为箭头
                block.is_arrow = block.check_is_arrow()
                
                blocks.append(block)
        
        return blocks
    
    def _parse_block_instances(self) -> List[Block]:
        """解析所有块引用（INSERT实体）"""
        block_instances = []
        
        inserts = self.backend.get_block_inserts()
        blocks_dict = self.backend.get_blocks()
        
        for insert in inserts:
            block_name = insert.get('block_name', '')
            
            if block_name not in blocks_dict:
                continue
            
            # 获取块定义中的实体
            entity_dicts = self.backend.get_block_entities(block_name)
            
            # 变换实体
            transformed_entities = []
            for entity_dict in entity_dicts:
                transformed_entity_dict = self._transform_entity_dict(
                    entity_dict,
                    insert.get('position', (0, 0, 0)),
                    insert.get('scale', (1, 1, 1)),
                    insert.get('rotation', 0)
                )
                
                entity = self._create_entity_from_dict(transformed_entity_dict)
                if entity:
                    transformed_entities.append(entity)
            
            # 获取属性
            attributes = []
            if hasattr(self.backend, 'get_attributes'):
                attr_dicts = self.backend.get_attributes(insert.get('handle', ''))
                for attr_dict in attr_dicts:
                    attribute = AttributeInfo(
                        tag=attr_dict.get('tag', ''),
                        value=attr_dict.get('text', ''),
                        position=attr_dict.get('position', (0, 0, 0)),
                        height=attr_dict.get('height', 1.0),
                        rotation=attr_dict.get('rotation', 0.0),
                        layer=attr_dict.get('layer', ''),
                        style=attr_dict.get('style', '')
                    )
                    attributes.append(attribute)
            
            # 创建块引用
            block_ref = BlockReference(
                id=insert.get('handle', self.generate_unique_id('INSERT_')),
                name=block_name,
                position=Point.from_tuple(insert.get('position', (0, 0, 0))),
                rotation=insert.get('rotation', 0),
                scale=insert.get('scale', (1, 1, 1)),
                attributes=attributes
            )
            
            # 创建块实例
            block_instance = Block(
                id=f"{block_name}_{block_ref.id}",
                name=block_name,
                entities=transformed_entities,
                reference=block_ref,
                is_arrow=False  # 稍后检查是否为箭头
            )
            
            # 检查是否为箭头
            block_instance.is_arrow = block_instance.check_is_arrow()
            
            block_instances.append(block_instance)
        
        return block_instances
    
    def _create_entity_from_dict(self, entity_dict: Dict) -> Optional[Entity]:
        """从字典创建实体"""
        entity_type = entity_dict.get('type', '')
        
        try:
            if entity_type == 'LINE':
                return LineEntity(
                    id=entity_dict.get('handle', self.generate_unique_id('LINE_')),
                    entity_type=EntityType.LINE,
                    layer=entity_dict.get('layer', ''),
                    start_point=Point.from_tuple(entity_dict.get('start', (0, 0, 0))),
                    end_point=Point.from_tuple(entity_dict.get('end', (0, 0, 0)))
                )
            elif entity_type == 'CIRCLE':
                return CircleEntity(
                    id=entity_dict.get('handle', self.generate_unique_id('CIRCLE_')),
                    entity_type=EntityType.CIRCLE,
                    layer=entity_dict.get('layer', ''),
                    center=Point.from_tuple(entity_dict.get('center', (0, 0, 0))),
                    radius=entity_dict.get('radius', 0.0)
                )
            elif entity_type == 'ARC':
                return ArcEntity(
                    id=entity_dict.get('handle', self.generate_unique_id('ARC_')),
                    entity_type=EntityType.ARC,
                    layer=entity_dict.get('layer', ''),
                    center=Point.from_tuple(entity_dict.get('center', (0, 0, 0))),
                    radius=entity_dict.get('radius', 0.0),
                    start_angle=entity_dict.get('start_angle', 0.0),
                    end_angle=entity_dict.get('end_angle', 0.0)
                )
            elif entity_type in ('TEXT', 'MTEXT'):
                return TextEntity(
                    id=entity_dict.get('handle', self.generate_unique_id('TEXT_')),
                    entity_type=EntityType.TEXT if entity_type == 'TEXT' else EntityType.MTEXT,
                    layer=entity_dict.get('layer', ''),
                    text=entity_dict.get('text', ''),
                    position=Point.from_tuple(entity_dict.get('position', (0, 0, 0))),
                    height=entity_dict.get('height', 1.0),
                    rotation=entity_dict.get('rotation', 0.0)
                )
            elif entity_type in ('LWPOLYLINE', 'POLYLINE'):
                # 正确处理多段线，创建 PolylineEntity/LwPolylineEntity 并传递顶点
                points = entity_dict.get('points', [])
                # 调试：打印 points 内容
                vertices = []
                for pt in points:
                    try:
                        vertices.append(Point.from_tuple(pt))
                    except Exception:
                        pass
                is_closed = entity_dict.get('closed', False)
                if entity_type == 'LWPOLYLINE':
                    return LwPolylineEntity(
                        id=entity_dict.get('handle', self.generate_unique_id('LWPOLYLINE_')),
                        entity_type=EntityType.LWPOLYLINE,
                        layer=entity_dict.get('layer', ''),
                        vertices=vertices,
                        is_closed=is_closed
                    )
                else:
                    return PolylineEntity(
                        id=entity_dict.get('handle', self.generate_unique_id('POLYLINE_')),
                        entity_type=EntityType.POLYLINE,
                        layer=entity_dict.get('layer', ''),
                        vertices=vertices,
                        is_closed=is_closed
                    )
            else:
                # 其他实体类型
                return Entity(
                    id=entity_dict.get('handle', self.generate_unique_id('ENTITY_')),
                    entity_type=EntityType.UNKNOWN,
                    layer=entity_dict.get('layer', '')
                )
        except Exception as e:
            print(f"Error creating entity of type {entity_type}: {e}")
            return None
    
    def _transform_entity_dict(self, entity_dict: Dict, position: Tuple, scale: Tuple, rotation: float) -> Dict:
        """变换实体字典"""
        entity_type = entity_dict.get('type', '')
        result = entity_dict.copy()
        
        # 旋转角度（弧度）
        angle_rad = math.radians(rotation)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        if entity_type == 'LINE':
            # 变换起点
            start = list(entity_dict.get('start', (0, 0, 0)))
            start[0] *= scale[0]
            start[1] *= scale[1]
            
            # 旋转
            x_rot = start[0] * cos_angle - start[1] * sin_angle
            y_rot = start[0] * sin_angle + start[1] * cos_angle
            
            # 平移
            start[0] = x_rot + position[0]
            start[1] = y_rot + position[1]
            
            # 变换终点
            end = list(entity_dict.get('end', (0, 0, 0)))
            end[0] *= scale[0]
            end[1] *= scale[1]
            
            # 旋转
            x_rot = end[0] * cos_angle - end[1] * sin_angle
            y_rot = end[0] * sin_angle + end[1] * cos_angle
            
            # 平移
            end[0] = x_rot + position[0]
            end[1] = y_rot + position[1]
            
            result['start'] = tuple(start)
            result['end'] = tuple(end)
        
        elif entity_type == 'CIRCLE':
            # 变换中心点
            center = list(entity_dict.get('center', (0, 0, 0)))
            center[0] *= scale[0]
            center[1] *= scale[1]
            
            # 旋转
            x_rot = center[0] * cos_angle - center[1] * sin_angle
            y_rot = center[0] * sin_angle + center[1] * cos_angle
            
            # 平移
            center[0] = x_rot + position[0]
            center[1] = y_rot + position[1]
            
            # 缩放半径（取x和y缩放的平均值）
            radius = entity_dict.get('radius', 0.0) * (scale[0] + scale[1]) / 2
            
            result['center'] = tuple(center)
            result['radius'] = radius
        
        elif entity_type == 'ARC':
            # 变换中心点
            center = list(entity_dict.get('center', (0, 0, 0)))
            center[0] *= scale[0]
            center[1] *= scale[1]
            
            # 旋转
            x_rot = center[0] * cos_angle - center[1] * sin_angle
            y_rot = center[0] * sin_angle + center[1] * cos_angle
            
            # 平移
            center[0] = x_rot + position[0]
            center[1] = y_rot + position[1]
            
            # 缩放半径
            radius = entity_dict.get('radius', 0.0) * (scale[0] + scale[1]) / 2
            
            # 调整角度
            start_angle = entity_dict.get('start_angle', 0.0) + rotation
            end_angle = entity_dict.get('end_angle', 0.0) + rotation
            
            result['center'] = tuple(center)
            result['radius'] = radius
            result['start_angle'] = start_angle
            result['end_angle'] = end_angle
        
        elif entity_type in ('TEXT', 'MTEXT'):
            # 变换位置
            pos = list(entity_dict.get('position', (0, 0, 0)))
            pos[0] *= scale[0]
            pos[1] *= scale[1]
            
            # 旋转
            x_rot = pos[0] * cos_angle - pos[1] * sin_angle
            y_rot = pos[0] * sin_angle + pos[1] * cos_angle
            
            # 平移
            pos[0] = x_rot + position[0]
            pos[1] = y_rot + position[1]
            
            # 缩放高度
            height = entity_dict.get('height', 1.0) * scale[1]
            
            # 调整旋转角度
            text_rotation = entity_dict.get('rotation', 0.0) + rotation
            
            result['position'] = tuple(pos)
            result['height'] = height
            result['rotation'] = text_rotation
        
        elif entity_type in ('LWPOLYLINE', 'POLYLINE'):
            # 变换所有点
            points = []
            for point in entity_dict.get('points', []):
                pt = list(point)
                
                # 缩放
                pt[0] *= scale[0]
                pt[1] *= scale[1]
                
                # 旋转
                x_rot = pt[0] * cos_angle - pt[1] * sin_angle
                y_rot = pt[0] * sin_angle + pt[1] * cos_angle
                
                # 平移
                pt[0] = x_rot + position[0]
                pt[1] = y_rot + position[1]
                
                points.append(tuple(pt))
            
            result['points'] = points
        
        return result