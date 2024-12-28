import unittest
import logging
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import math
import ezdxf
from ezdxf.math import Vec3, Matrix44
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
from Entity import EntityNetwork, CompositeEntity, BlockPattern
from dxf2png import dxf_to_image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEntityNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置目录
        cls.single_blocks_dir = Path("single_blocks")
        cls.test_output_dir = Path("test_output")
        cls.test_output_dir.mkdir(exist_ok=True)
        
        # 设置容差配置
        cls.tolerances = {
            'point_tolerance': 0.5,    # 点距离容差
            'count_tolerance': 1,      # 实体数量容差
            'size_tolerance': 0.2,     # 尺寸容差（20%）
            'ratio_tolerance': 0.2,    # 宽高比容差（20%）
        }
        
        # 加载所有标准元件的模式
        cls.standard_patterns = {}
        for dxf_file in cls.single_blocks_dir.glob("*.dxf"):
            try:
                network = EntityNetwork(str(dxf_file), tolerances=cls.tolerances)
                patterns = network.extract_block_patterns()
                if patterns:
                    cls.standard_patterns[dxf_file.stem] = patterns
                    logger.info(f"从 {dxf_file.name} 提取了 {len(patterns)} 个标准模式")
            except Exception as e:
                logger.error(f"处理标准元件 {dxf_file.name} 失败: {str(e)}")

    def test_original_blocks(self):
        """测试原始元件的识别"""
        for filename, patterns in self.standard_patterns.items():
            with self.subTest(f"测试原始元件: {filename}"):
                # 加载原始文件
                network = EntityNetwork(
                    str(self.single_blocks_dir / f"{filename}.dxf"),
                    tolerances=self.tolerances
                )
                
                # 测试每个模式
                for pattern in patterns:
                    # 使用 find_matching_blocks 而不是 find_similar_entity_groups
                    matches = network.find_matching_blocks(pattern)
                    
                    # 验证至少能找到一个匹配
                    self.assertTrue(
                        len(matches) > 0,
                        f"无法识别原始元件 {filename}"
                    )

    def test_rotated_blocks(self):
        """测试旋转后的元件识别"""
        angles = [0, 90, 180, 270]  # 测试90度的整数倍旋转
        
        for filename, patterns in self.standard_patterns.items():
            for angle in angles:
                with self.subTest(f"测试旋转 {angle}° 的元件: {filename}"):
                    # 加载并旋转文件
                    source_doc = ezdxf.readfile(str(self.single_blocks_dir / f"{filename}.dxf"))
                    rotated_doc = self._rotate_entities(source_doc, angle)
                    
                    if not rotated_doc:
                        self.skipTest(f"无法创建旋转后的文档: {filename}")
                        continue
                        
                    # 保存旋转后的文件
                    output_dxf = self.test_output_dir / f"{filename}_rotated_{angle}.dxf"
                    rotated_doc.saveas(output_dxf)
                    
                    # 转换为图像
                    output_png = self.test_output_dir / f"{filename}_rotated_{angle}.png"
                    dxf_to_image(str(output_dxf), str(output_png))
                    logger.info(f"已保存旋转图像: {output_png}")
                    
                    # 测试识别
                    network = EntityNetwork(str(output_dxf), tolerances=self.tolerances)
                    
                    # 测试每个模式
                    for pattern in patterns:
                        matches = network.find_matching_blocks(pattern)
                        logger.info(f"旋转 {angle}° 后找到 {len(matches)} 个匹配的blocks")
                        
                        self.assertTrue(
                            len(matches) > 0,
                            f"无法识别旋转 {angle}° 后的元件 {filename}"
                        )

    def test_decomposed_blocks(self):
        """测试分解后的元件识别"""
        for filename, patterns in self.standard_patterns.items():
            with self.subTest(f"测试分解后的元件: {filename}"):
                # 加载并分解文件
                source_doc = ezdxf.readfile(str(self.single_blocks_dir / f"{filename}.dxf"))
                decomposed_doc = self._decompose_block(source_doc)
                
                if not decomposed_doc:
                    self.skipTest(f"无法创建分解后的文档: {filename}")
                    continue
                    
                # 保存分解后的文件
                output_path = self.test_output_dir / f"{filename}_decomposed.dxf"
                decomposed_doc.saveas(output_path)
                
                # 测试识别
                network = EntityNetwork(str(output_path), tolerances=self.tolerances)
                
                # 测试每个模式
                for pattern in patterns:
                    matches = network.find_similar_entity_groups(pattern)
                    
                    # 验证至少能找到一个匹配
                    self.assertTrue(
                        len(matches) > 0,
                        f"无法识别分解后的元件 {filename}"
                    )

    @classmethod
    def _rotate_entities(cls, doc: ezdxf.document.Drawing, angle: float) -> ezdxf.document.Drawing:
        """旋转DXF文档中的所有实体"""
        try:
            # 创建新文档
            new_doc = ezdxf.new()
            new_msp = new_doc.modelspace()
            
            # 复制图层定义
            for layer in doc.layers:
                if layer.dxf.name not in new_doc.layers:
                    new_doc.layers.new(name=layer.dxf.name, dxfattribs=layer.dxfattribs())
            
            # 直接复制块定义，不做旋转
            for block in doc.blocks:
                if block.name not in new_doc.blocks:
                    new_block = new_doc.blocks.new(name=block.name)
                    for entity in block:
                        new_block.add_entity(entity.copy())
            
            # 复制并旋转模型空间中的实体
            angle_rad = math.radians(angle)
            rotation_matrix = Matrix44.z_rotate(angle_rad)
            
            for entity in doc.modelspace():
                new_entity = new_msp.add_entity(entity.copy())
                if hasattr(new_entity, 'transform'):
                    new_entity.transform(rotation_matrix)
            
            return new_doc
                
        except Exception as e:
            logger.error(f"旋转实体失败: {str(e)}")
            return None

    @classmethod
    def _decompose_block(cls, doc: ezdxf.document.Drawing) -> ezdxf.document.Drawing:
        """将块引用分解为基本实体"""
        try:
            # 创建新文档
            new_doc = ezdxf.new()
            new_msp = new_doc.modelspace()
            
            # 复制图层定义
            for layer in doc.layers:
                if layer.dxf.name not in new_doc.layers:
                    new_doc.layers.new(name=layer.dxf.name, dxfattribs=layer.dxfattribs())
            
            # 遍历所有实体
            for entity in doc.modelspace():
                if entity.dxftype() == 'INSERT':
                    # 获取块定义
                    block = doc.blocks[entity.dxf.name]
                    
                    # 计算变换矩阵
                    transform = Matrix44()
                    # 应用缩放
                    transform *= Matrix44.scale(
                        entity.dxf.xscale,
                        entity.dxf.yscale,
                        entity.dxf.zscale
                    )
                    # 应用旋转
                    if entity.dxf.rotation:
                        transform *= Matrix44.z_rotate(math.radians(entity.dxf.rotation))
                    # 应用平移
                    transform *= Matrix44.translate(entity.dxf.insert.x, entity.dxf.insert.y, 0)
                    
                    # 复制并变换块中的每个实体
                    for block_entity in block:
                        if block_entity.dxftype() not in {'ATTDEF', 'ATTRIB'}:
                            new_entity = new_msp.add_entity(block_entity.copy())
                            if hasattr(new_entity, 'transform'):
                                new_entity.transform(transform)
                else:
                    # 直接复制非块引用实体
                    new_msp.add_entity(entity.copy())
            
            return new_doc
                
        except Exception as e:
            logger.error(f"分解块失败: {str(e)}")
            return None

if __name__ == '__main__':
    unittest.main(verbosity=2)