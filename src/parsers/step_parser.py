"""
STEP文件解析模块
提供STEP文件格式的解析功能
支持多种STEP解析后端
"""

import os
import math
from typing import List, Tuple, Dict, Any, Optional, Set
import uuid

from src.core.data_structures import (
    Point, BoundingBox, EntityType, Entity, LineEntity, 
    CircleEntity, ArcEntity, TextEntity, Block, 
    BlockReference, AttributeInfo
)
from src.parsers.parser_interface import CADFileParser


class STEPParseError(Exception):
    """STEP解析错误"""
    pass


class STEPBackendBase:
    """STEP解析后端基类"""
    
    def __init__(self):
        self.doc = None
    
    def load_file(self, file_path: str) -> bool:
        """加载STEP文件"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_shapes(self) -> List[Dict]:
        """获取所有形状"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_assemblies(self) -> List[Dict]:
        """获取所有装配体"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_components(self, assembly_id: str) -> List[Dict]:
        """获取装配体中的组件"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_edges(self, shape_id: str) -> List[Dict]:
        """获取形状的边线"""
        raise NotImplementedError("子类必须实现此方法")


try:
    # 尝试导入OCC (Open CASCADE Technology)
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Shape, topods_Edge, topods_Vertex
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse
    from OCC.Core.gp import gp_Pnt
    
    class OCCBackend(STEPBackendBase):
        """OCC解析后端"""
        
        def load_file(self, file_path: str) -> bool:
            """加载STEP文件"""
            try:
                step_reader = STEPControl_Reader()
                status = step_reader.ReadFile(file_path)
                
                if status != IFSelect_RetDone:
                    raise STEPParseError("Failed to read STEP file")
                
                step_reader.TransferRoots()
                self.doc = step_reader.Model()
                
                return True
            except Exception as e:
                raise STEPParseError(f"Failed to load STEP file: {e}")
        
        def get_shapes(self) -> List[Dict]:
            """获取所有形状"""
            shapes = []
            
            for i in range(1, self.doc.NbEntities() + 1):
                entity = self.doc.Value(i)
                if entity.IsKind("SHAPE"):
                    shape = {
                        "id": str(entity.GetHashCode()),
                        "type": entity.DynamicType().Name(),
                        "name": entity.Name().ToCString() if hasattr(entity, "Name") else ""
                    }
                    shapes.append(shape)
            
            return shapes
        
        def get_assemblies(self) -> List[Dict]:
            """获取所有装配体"""
            assemblies = []
            
            for i in range(1, self.doc.NbEntities() + 1):
                entity = self.doc.Value(i)
                if entity.IsKind("PRODUCT_DEFINITION"):
                    assembly = {
                        "id": str(entity.GetHashCode()),
                        "name": entity.Name().ToCString() if hasattr(entity, "Name") else ""
                    }
                    assemblies.append(assembly)
            
            return assemblies
        
        def get_components(self, assembly_id: str) -> List[Dict]:
            """获取装配体中的组件"""
            components = []
            
            for i in range(1, self.doc.NbEntities() + 1):
                entity = self.doc.Value(i)
                if entity.IsKind("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
                    if str(entity.RelatingProduct().GetHashCode()) == assembly_id:
                        component = {
                            "id": str(entity.RelatedProduct().GetHashCode()),
                            "name": entity.RelatedProduct().Name().ToCString() if hasattr(entity.RelatedProduct(), "Name") else "",
                            "transform": self._get_transformation(entity)
                        }
                        components.append(component)
            
            return components
        
        def get_edges(self, shape_id: str) -> List[Dict]:
            """获取形状的边线"""
            edges = []
            
            # 查找形状
            shape = None
            for i in range(1, self.doc.NbEntities() + 1):
                entity = self.doc.Value(i)
                if entity.IsKind("SHAPE") and str(entity.GetHashCode()) == shape_id:
                    shape = entity
                    break
            
            if not shape:
                return []
            
            # 提取边线
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            edge_index = 0
            
            while explorer.More():
                edge = topods_Edge(explorer.Current())
                edge_index += 1
                
                # 分析边线类型
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()
                
                # 获取起点和终点
                p1 = BRep_Tool.Pnt(topods_Vertex(TopExp_Explorer(edge, TopAbs_VERTEX).Current()))
                explorer_end = TopExp_Explorer(edge, TopAbs_VERTEX)
                explorer_end.Next()  # 移动到第二个顶点
                p2 = BRep_Tool.Pnt(topods_Vertex(explorer_end.Current()))
                
                start_point = (p1.X(), p1.Y(), p1.Z())
                end_point = (p2.X(), p2.Y(), p2.Z())
                
                # 处理不同类型的曲线
                if curve_type == GeomAbs_Line:
                    edge_data = {
                        "id": f"edge_{shape_id}_{edge_index}",
                        "type": "LINE",
                        "start": start_point,
                        "end": end_point
                    }
                    edges.append(edge_data)
                
                elif curve_type == GeomAbs_Circle:
                    circle = curve_adaptor.Circle()
                    center = circle.Location()
                    radius = circle.Radius()
                    
                    edge_data = {
                        "id": f"edge_{shape_id}_{edge_index}",
                        "type": "CIRCLE" if curve_adaptor.LastParameter() - curve_adaptor.FirstParameter() >= 2 * math.pi else "ARC",
                        "center": (center.X(), center.Y(), center.Z()),
                        "radius": radius,
                        "start_angle": curve_adaptor.FirstParameter(),
                        "end_angle": curve_adaptor.LastParameter()
                    }
                    edges.append(edge_data)
                
                elif curve_type == GeomAbs_Ellipse:
                    # 处理椭圆
                    ellipse = curve_adaptor.Ellipse()
                    center = ellipse.Location()
                    major_radius = ellipse.MajorRadius()
                    minor_radius = ellipse.MinorRadius()
                    
                    edge_data = {
                        "id": f"edge_{shape_id}_{edge_index}",
                        "type": "ELLIPSE",
                        "center": (center.X(), center.Y(), center.Z()),
                        "major_radius": major_radius,
                        "minor_radius": minor_radius,
                        "start_angle": curve_adaptor.FirstParameter(),
                        "end_angle": curve_adaptor.LastParameter()
                    }
                    edges.append(edge_data)
                
                else:
                    # 简化处理其他曲线类型
                    edge_data = {
                        "id": f"edge_{shape_id}_{edge_index}",
                        "type": "EDGE",
                        "start": start_point,
                        "end": end_point
                    }
                    edges.append(edge_data)
                
                explorer.Next()
            
            return edges
        
        def _get_transformation(self, entity) -> Dict:
            """获取变换信息"""
            # 这只是一个简化的示例
            # 实际实现需要从STEP文件的AXIS2_PLACEMENT_3D中提取变换数据
            return {
                "translation": (0, 0, 0),
                "rotation": (0, 0, 0),
                "scale": (1, 1, 1)
            }

except ImportError:
    # OCC不可用
    pass


try:
    # 尝试导入FreeCAD (更轻量级的STEP解析)
    import FreeCAD
    import Part
    
    class FreeCADBackend(STEPBackendBase):
        """FreeCAD解析后端"""
        
        def load_file(self, file_path: str) -> bool:
            """加载STEP文件"""
            try:
                self.doc = FreeCAD.open(file_path)
                return True
            except Exception as e:
                raise STEPParseError(f"Failed to load STEP file: {e}")
        
        def get_shapes(self) -> List[Dict]:
            """获取所有形状"""
            shapes = []
            
            for obj in self.doc.Objects:
                if hasattr(obj, "Shape"):
                    shape = {
                        "id": str(obj.Name),
                        "type": obj.TypeId,
                        "name": obj.Label
                    }
                    shapes.append(shape)
            
            return shapes
        
        def get_assemblies(self) -> List[Dict]:
            """获取所有装配体"""
            assemblies = []
            
            for obj in self.doc.Objects:
                if hasattr(obj, "Group") and obj.Group:
                    assembly = {
                        "id": str(obj.Name),
                        "name": obj.Label
                    }
                    assemblies.append(assembly)
            
            return assemblies
        
        def get_components(self, assembly_id: str) -> List[Dict]:
            """获取装配体中的组件"""
            components = []
            
            for obj in self.doc.Objects:
                if obj.Name == assembly_id and hasattr(obj, "Group"):
                    for child in obj.Group:
                        component = {
                            "id": str(child.Name),
                            "name": child.Label,
                            "transform": self._get_transformation(child)
                        }
                        components.append(component)
            
            return components
        
        def get_edges(self, shape_id: str) -> List[Dict]:
            """获取形状的边线"""
            edges = []
            
            for obj in self.doc.Objects:
                if obj.Name == shape_id and hasattr(obj, "Shape"):
                    shape = obj.Shape
                    
                    for i, edge in enumerate(shape.Edges):
                        if edge.Curve.TypeId == 'Part::GeomLine':
                            # 线段
                            edge_data = {
                                "id": f"edge_{shape_id}_{i}",
                                "type": "LINE",
                                "start": tuple(edge.Vertexes[0].Point),
                                "end": tuple(edge.Vertexes[1].Point)
                            }
                            edges.append(edge_data)
                        
                        elif edge.Curve.TypeId == 'Part::GeomCircle':
                            # 圆或圆弧
                            center = edge.Curve.Center
                            radius = edge.Curve.Radius
                            
                            is_full_circle = (edge.LastParameter - edge.FirstParameter) >= 2 * math.pi
                            
                            edge_data = {
                                "id": f"edge_{shape_id}_{i}",
                                "type": "CIRCLE" if is_full_circle else "ARC",
                                "center": (center.x, center.y, center.z),
                                "radius": radius,
                                "start_angle": edge.FirstParameter,
                                "end_angle": edge.LastParameter
                            }
                            edges.append(edge_data)
                        
                        elif edge.Curve.TypeId == 'Part::GeomEllipse':
                            # 椭圆
                            center = edge.Curve.Center
                            major_radius = edge.Curve.MajorRadius
                            minor_radius = edge.Curve.MinorRadius
                            
                            edge_data = {
                                "id": f"edge_{shape_id}_{i}",
                                "type": "ELLIPSE",
                                "center": (center.x, center.y, center.z),
                                "major_radius": major_radius,
                                "minor_radius": minor_radius,
                                "start_angle": edge.FirstParameter,
                                "end_angle": edge.LastParameter
                            }
                            edges.append(edge_data)
                        
                        else:
                            # 其他类型的曲线
                            if len(edge.Vertexes) >= 2:
                                edge_data = {
                                    "id": f"edge_{shape_id}_{i}",
                                    "type": "EDGE",
                                    "start": tuple(edge.Vertexes[0].Point),
                                    "end": tuple(edge.Vertexes[1].Point)
                                }
                                edges.append(edge_data)
            
            return edges
        
        def _get_transformation(self, obj) -> Dict:
            """获取变换信息"""
            if hasattr(obj, "Placement"):
                placement = obj.Placement
                return {
                    "translation": tuple(placement.Base),
                    "rotation": tuple(placement.Rotation.getYawPitchRoll()),
                    "scale": (1, 1, 1)  # FreeCAD中缩放通常是单独的属性
                }
            else:
                return {
                    "translation": (0, 0, 0),
                    "rotation": (0, 0, 0),
                    "scale": (1, 1, 1)
                }

except ImportError:
    # FreeCAD不可用
    pass


class SimpleSTEPParser:
    """简单的STEP解析器（实现有限的功能）"""
    
    def __init__(self):
        self.shapes = []
        self.assemblies = []
        self.components = {}
        self.edges = {}
    
    def load_file(self, file_path: str) -> bool:
        """加载STEP文件（简化版，仅读取基本结构）"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # 简单解析STEP文件结构
            in_data_section = False
            current_id = None
            current_type = None
            
            for line in lines:
                line = line.strip()
                
                # 查找数据部分的开始
                if line == "DATA;":
                    in_data_section = True
                    continue
                
                # 查找数据部分的结束
                if line == "ENDSEC;":
                    in_data_section = False
                    continue
                
                # 解析数据行
                if in_data_section and line.startswith('#'):
                    # 提取ID和类型
                    parts = line.split('=')
                    if len(parts) >= 2:
                        current_id = parts[0].strip()
                        type_info = parts[1].strip()
                        
                        if type_info.startswith('ADVANCED_FACE'):
                            # 形状
                            self.shapes.append({
                                "id": current_id,
                                "type": "ADVANCED_FACE",
                                "name": f"Face_{current_id[1:]}"  # 移除#
                            })
                        
                        elif type_info.startswith('MANIFOLD_SOLID_BREP'):
                            # 实体
                            self.shapes.append({
                                "id": current_id,
                                "type": "MANIFOLD_SOLID_BREP",
                                "name": f"Solid_{current_id[1:]}"
                            })
                        
                        elif type_info.startswith('EDGE_CURVE'):
                            # 边线
                            # 提取顶点和曲线信息
                            vertices = []
                            curve_type = "UNKNOWN"
                            
                            if ',' in type_info:
                                params = type_info.split(',')
                                if len(params) >= 4:
                                    vertices = [params[1].strip(), params[2].strip()]
                                    curve_ref = params[3].strip()
                                    
                                    # 尝试确定曲线类型
                                    if 'LINE' in curve_ref:
                                        curve_type = "LINE"
                                    elif 'CIRCLE' in curve_ref:
                                        curve_type = "CIRCLE"
                            
                            # 为简化起见，使用随机点
                            if vertices:
                                # 每个形状存储自己的边线列表
                                shape_id = f"shape_{len(self.edges)}"
                                if shape_id not in self.edges:
                                    self.edges[shape_id] = []
                                
                                self.edges[shape_id].append({
                                    "id": current_id,
                                    "type": curve_type,
                                    "start": (0, 0, 0),  # 简化
                                    "end": (10, 10, 0)   # 简化
                                })
                        
                        elif type_info.startswith('PRODUCT_DEFINITION'):
                            # 装配体
                            self.assemblies.append({
                                "id": current_id,
                                "name": f"Assembly_{current_id[1:]}"
                            })
                        
                        elif type_info.startswith('NEXT_ASSEMBLY_USAGE_OCCURRENCE'):
                            # 组件关系
                            if ',' in type_info:
                                params = type_info.split(',')
                                if len(params) >= 3:
                                    assembly_ref = params[1].strip()
                                    component_ref = params[2].strip()
                                    
                                    if assembly_ref not in self.components:
                                        self.components[assembly_ref] = []
                                    
                                    self.components[assembly_ref].append({
                                        "id": component_ref,
                                        "name": f"Component_{component_ref[1:]}",
                                        "transform": {
                                            "translation": (0, 0, 0),  # 简化
                                            "rotation": (0, 0, 0),     # 简化
                                            "scale": (1, 1, 1)         # 简化
                                        }
                                    })
            
            return True
        
        except Exception as e:
            raise STEPParseError(f"Failed to load STEP file: {e}")
    
    def get_shapes(self) -> List[Dict]:
        """获取所有形状"""
        return self.shapes
    
    def get_assemblies(self) -> List[Dict]:
        """获取所有装配体"""
        return self.assemblies
    
    def get_components(self, assembly_id: str) -> List[Dict]:
        """获取装配体中的组件"""
        return self.components.get(assembly_id, [])
    
    def get_edges(self, shape_id: str) -> List[Dict]:
        """获取形状的边线"""
        return self.edges.get(shape_id, [])


class STEPParser(CADFileParser):
    """STEP文件解析器"""
    
    def __init__(self):
        """初始化STEP解析器"""
        # 尝试使用可用的后端
        self.backend = None
        
        # 先尝试OCC
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            self.backend = OCCBackend()
            print("Using OCC backend for STEP parsing")
            return
        except ImportError:
            pass
        
        # 再尝试FreeCAD
        try:
            import FreeCAD
            self.backend = FreeCADBackend()
            print("Using FreeCAD backend for STEP parsing")
            return
        except ImportError:
            pass
        
        # 最后使用简单解析器
        self.backend = SimpleSTEPParser()
        print("Using simple backend for STEP parsing")
    
    def supports_format(self, file_extension: str) -> bool:
        """检查是否支持指定格式"""
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        return file_extension.lower() in ['step', 'stp']
    
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """解析STEP文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # 使用后端加载文件
            self.backend.load_file(file_path)
            
            # 获取形状作为块
            shapes = self.backend.get_shapes()
            blocks = []
            
            for shape in shapes:
                # 获取形状的边线
                edges = self.backend.get_edges(shape['id'])
                
                # 将边线转换为实体
                entities = []
                for edge in edges:
                    entity = self._create_entity_from_dict(edge)
                    if entity:
                        entities.append(entity)
                
                # 创建块
                if entities:
                    block = Block(
                        id=shape['id'],
                        name=shape.get('name', f"Shape_{len(blocks)}"),
                        entities=entities
                    )
                    blocks.append(block)
            
            # 获取装配体作为块引用
            assemblies = self.backend.get_assemblies()
            block_instances = []
            
            for assembly in assemblies:
                # 获取装配体中的组件
                components = self.backend.get_components(assembly['id'])
                
                for component in components:
                    # 查找对应的块
                    block = next((b for b in blocks if b.id == component['id']), None)
                    
                    if block:
                        # 创建块引用
                        transform = component.get('transform', {})
                        translation = transform.get('translation', (0, 0, 0))
                        rotation = transform.get('rotation', (0, 0, 0))
                        scale = transform.get('scale', (1, 1, 1))
                        
                        block_ref = BlockReference(
                            id=f"{assembly['id']}_{component['id']}",
                            name=block.name,
                            position=Point.from_tuple(translation),
                            rotation=rotation[2],  # 取Z轴旋转
                            scale=scale,
                            attributes=[]
                        )
                        
                        # 变换实体
                        transformed_entities = self._transform_entities(
                            block.entities,
                            translation,
                            rotation,
                            scale
                        )
                        
                        # 创建块实例
                        block_instance = Block(
                            id=f"{block.name}_{block_ref.id}",
                            name=block.name,
                            entities=transformed_entities,
                            reference=block_ref
                        )
                        
                        block_instances.append(block_instance)
            
            # 收集所有实体
            entities = []
            for block in blocks:
                for entity in block.entities:
                    if isinstance(entity, LineEntity):
                        entities.append(entity)
            
            return entities, block_instances + blocks, {"shapes": shapes, "assemblies": assemblies}
        
        except Exception as e:
            raise STEPParseError(f"Error parsing STEP file: {e}")
    
    def _create_entity_from_dict(self, edge_dict: Dict) -> Optional[Entity]:
        """从字典创建实体"""
        edge_type = edge_dict.get('type', '')
        
        try:
            if edge_type == 'LINE':
                return LineEntity(
                    id=edge_dict.get('id', self.generate_unique_id('LINE_')),
                    entity_type=EntityType.LINE,
                    layer='0',  # STEP文件通常没有图层概念
                    start_point=Point.from_tuple(edge_dict.get('start', (0, 0, 0))),
                    end_point=Point.from_tuple(edge_dict.get('end', (0, 0, 0)))
                )
            elif edge_type == 'CIRCLE':
                return CircleEntity(
                    id=edge_dict.get('id', self.generate_unique_id('CIRCLE_')),
                    entity_type=EntityType.CIRCLE,
                    layer='0',
                    center=Point.from_tuple(edge_dict.get('center', (0, 0, 0))),
                    radius=edge_dict.get('radius', 0.0)
                )
            elif edge_type == 'ARC':
                return ArcEntity(
                    id=edge_dict.get('id', self.generate_unique_id('ARC_')),
                    entity_type=EntityType.ARC,
                    layer='0',
                    center=Point.from_tuple(edge_dict.get('center', (0, 0, 0))),
                    radius=edge_dict.get('radius', 0.0),
                    start_angle=edge_dict.get('start_angle', 0.0),
                    end_angle=edge_dict.get('end_angle', 0.0)
                )
            else:
                # 对于未知类型或复杂的曲线，创建一条线段
                if 'start' in edge_dict and 'end' in edge_dict:
                    return LineEntity(
                        id=edge_dict.get('id', self.generate_unique_id('EDGE_')),
                        entity_type=EntityType.LINE,
                        layer='0',
                        start_point=Point.from_tuple(edge_dict.get('start', (0, 0, 0))),
                        end_point=Point.from_tuple(edge_dict.get('end', (0, 0, 0)))
                    )
                else:
                    return None
        except Exception as e:
            print(f"Error creating entity of type {edge_type}: {e}")
            return None
    
    def _transform_entities(self, entities: List[Entity], 
                          translation: Tuple[float, float, float],
                          rotation: Tuple[float, float, float],
                          scale: Tuple[float, float, float]) -> List[Entity]:
        """变换实体列表"""
        transformed = []
        
        # 提取旋转值（弧度）
        rot_x, rot_y, rot_z = rotation
        
        # 创建旋转矩阵（简化为仅考虑Z轴旋转）
        cos_z = math.cos(rot_z)
        sin_z = math.sin(rot_z)
        
        for entity in entities:
            if isinstance(entity, LineEntity):
                # 变换起点
                start_x = entity.start_point.x * scale[0]
                start_y = entity.start_point.y * scale[1]
                start_z = entity.start_point.z * scale[2]
                
                # 旋转（Z轴）
                rotated_start_x = start_x * cos_z - start_y * sin_z
                rotated_start_y = start_x * sin_z + start_y * cos_z
                
                # 平移
                final_start = Point(
                    rotated_start_x + translation[0],
                    rotated_start_y + translation[1],
                    start_z + translation[2]
                )
                
                # 变换终点
                end_x = entity.end_point.x * scale[0]
                end_y = entity.end_point.y * scale[1]
                end_z = entity.end_point.z * scale[2]
                
                # 旋转（Z轴）
                rotated_end_x = end_x * cos_z - end_y * sin_z
                rotated_end_y = end_x * sin_z + end_y * cos_z
                
                # 平移
                final_end = Point(
                    rotated_end_x + translation[0],
                    rotated_end_y + translation[1],
                    end_z + translation[2]
                )
                
                # 创建变换后的线段
                transformed.append(LineEntity(
                    id=f"{entity.id}_transformed",
                    entity_type=entity.entity_type,
                    layer=entity.layer,
                    start_point=final_start,
                    end_point=final_end
                ))
            
            elif isinstance(entity, CircleEntity):
                # 变换圆心
                center_x = entity.center.x * scale[0]
                center_y = entity.center.y * scale[1]
                center_z = entity.center.z * scale[2]
                
                # 旋转（Z轴）
                rotated_center_x = center_x * cos_z - center_y * sin_z
                rotated_center_y = center_x * sin_z + center_y * cos_z
                
                # 平移
                final_center = Point(
                    rotated_center_x + translation[0],
                    rotated_center_y + translation[1],
                    center_z + translation[2]
                )
                
                # 计算缩放后的半径
                scaled_radius = entity.radius * (scale[0] + scale[1]) / 2
                
                # 创建变换后的圆
                transformed.append(CircleEntity(
                    id=f"{entity.id}_transformed",
                    entity_type=entity.entity_type,
                    layer=entity.layer,
                    center=final_center,
                    radius=scaled_radius
                ))
            
            elif isinstance(entity, ArcEntity):
                # 变换圆心
                center_x = entity.center.x * scale[0]
                center_y = entity.center.y * scale[1]
                center_z = entity.center.z * scale[2]
                
                # 旋转（Z轴）
                rotated_center_x = center_x * cos_z - center_y * sin_z
                rotated_center_y = center_x * sin_z + center_y * cos_z
                
                # 平移
                final_center = Point(
                    rotated_center_x + translation[0],
                    rotated_center_y + translation[1],
                    center_z + translation[2]
                )
                
                # 计算缩放后的半径
                scaled_radius = entity.radius * (scale[0] + scale[1]) / 2
                
                # 调整弧的角度
                start_angle = entity.start_angle + rot_z
                end_angle = entity.end_angle + rot_z
                
                # 创建变换后的弧
                transformed.append(ArcEntity(
                    id=f"{entity.id}_transformed",
                    entity_type=entity.entity_type,
                    layer=entity.layer,
                    center=final_center,
                    radius=scaled_radius,
                    start_angle=start_angle,
                    end_angle=end_angle
                ))
            
            else:
                # 其他类型的实体简单复制
                transformed.append(entity)
        
        return transformed