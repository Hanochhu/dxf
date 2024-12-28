import unittest
import os
from Entity import EntityNetwork, BlockPattern, CompositeEntity, EntityInfo
import ezdxf
from pathlib import Path
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

class TestEntityNetwork(unittest.TestCase):
    @classmethod
    def _import_single_blocks(cls, blocks_dir: Path):
        """导入单个元件DXF文件并提取特征
        
        Args:
            blocks_dir: 包含单个元件DXF文件的目录路径
        
        Returns:
            Dict[str, Dict]: 每个块的特征字典，键为文件名
        """
        block_features = {}
        
        if not blocks_dir.exists():
            print(f"警告: 目录不存在 {blocks_dir}")
            return block_features
        
        # 遍历目录中的所有DXF文件
        for dxf_file in blocks_dir.glob("*.dxf"):
            try:
                # 导入DXF文件
                network = EntityNetwork(str(dxf_file))
                
                # 提取块模式
                patterns = network.extract_block_patterns()
                
                if patterns:
                    # 获取第一个块的特征
                    pattern = patterns[0]
                    features = network.get_block_features(pattern.name)
                    if features:
                        block_features[dxf_file.stem] = features
                        print(f"成功提取特征: {dxf_file.name}")
                    else:
                        print(f"未能获取块特征: {dxf_file.name}")
                else:
                    print(f"未找到块模式: {dxf_file.name}")
                    
            except Exception as e:
                print(f"处理文件失败 {dxf_file.name}: {str(e)}")
        
        return block_features

    @classmethod
    def setUpClass(cls):
        """设置测试环境，导入单个元件DXF文件"""
        # 创建预览图目录
        cls.preview_dir = Path("previews")
        if cls.preview_dir.exists():
            for file in cls.preview_dir.glob("*"):
                file.unlink()
        else:
            cls.preview_dir.mkdir(exist_ok=True)
        print(f"创建预览图目录: {cls.preview_dir.absolute()}")
        
        # 创建测试文件目录
        cls.test_dir = Path("test_dxf_files")
        if cls.test_dir.exists():
            for file in cls.test_dir.glob("*"):
                file.unlink()
        else:
            cls.test_dir.mkdir(exist_ok=True)
        print(f"创建测试文件目录: {cls.test_dir.absolute()}")
        
        # 定义变体参数
        variants = {
            'basic': {'rotation': 0, 'scale': 1.0, 'exploded': False},
            'rotated': {'rotation': 90, 'scale': 1.0, 'exploded': False},
            'scaled': {'rotation': 0, 'scale': 2.0, 'exploded': False},
            'rotated_scaled': {'rotation': 90, 'scale': 2.0, 'exploded': False},
            'exploded': {'rotation': 0, 'scale': 1.0, 'exploded': True},
            'exploded_rotated': {'rotation': 90, 'scale': 1.0, 'exploded': True},
            'exploded_scaled': {'rotation': 0, 'scale': 2.0, 'exploded': True}
        }
        
        # 定义单个元件目录
        cls.single_blocks_dir = Path("extracted_blocks")
        # 定义全局缩放比例
        global_scale=20.0

        # 导入单个元件DXF文件（如果存在）
        if cls.single_blocks_dir.exists():
            print("\n开始导入单个元件DXF文件...")
            cls.imported_features = cls._import_single_blocks(cls.single_blocks_dir)
            print(f"共导入 {len(cls.imported_features)} 个元件特征")
            
            # 为每个导入的元件生成变体和预览图
            cls.test_files = {}
            for block_name, features in cls.imported_features.items():
                try:
                    print(f"\n处理元件: {block_name}")
                    source_dxf = cls.single_blocks_dir / f"{block_name}.dxf"
                    
                    # 生成变体文件 - 使用 global_scale=20.0
                    variants_file = cls.test_dir / f"{block_name}_variants.dxf"
                    cls._create_test_valve_dxf(variants_file, str(source_dxf), variants, global_scale=global_scale)
                    cls.test_files[block_name] = variants_file
                    
                    # 验证变体文件是否生成
                    if variants_file.exists():
                        print(f"变体文件已生成: {variants_file}")
                        
                        # 生成变体文件的预览图
                        variants_preview = cls.preview_dir / f"{block_name}.png"
                        cls._create_preview(variants_file, variants_preview)
                        
                        # 验证预览图是否生成
                        if variants_preview.exists():
                            print(f"变体预览图已生成: {variants_preview}")
                        else:
                            print(f"错误: 变体预览图未生成: {variants_preview}")
                    else:
                        print(f"错误: 变体文件未生成: {variants_file}")
                        
                except Exception as e:
                    print(f"处理文件失败 {block_name}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
        else:
            cls.imported_features = {}
            print("\n未找到单个元件目录，跳过导入")

    @staticmethod
    def _create_test_valve_dxf(filename, source_dxf, variants, global_scale=20.0):
        """创建单个元件的所有变体测试文件"""
        print(f"\n开始创建变体文件: {filename}")
        print(f"源文件: {source_dxf}")
        
        # 读取源文件以获取块定义
        source_doc = ezdxf.readfile(source_dxf)
        print("成功读取源文件")
        
        # 创建新文件
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # 复制所有块定义
        block_count = 0
        source_block = None
        for block in source_doc.blocks:
            if not block.name.startswith('*'):  # 跳过特殊块
                target_block = doc.blocks.new(name=block.name)
                for entity in block:
                    target_block.add_entity(entity.copy())
                block_count += 1
                source_block = block  # 保存源块引用
                print(f"复制块: {block.name}")
        print(f"复制了 {block_count} 个块定义")
        
        # 创建图层
        doc.layers.new('BLOCK', dxfattribs={'color': 1})
        doc.layers.new('TEXT', dxfattribs={'color': 2})
        
        # 定义变体位置（网格布局）
        positions = {
            # 第一行：基本变体
            'basic':         (0, 0),
            'rotated':       (80, 0),
            'scaled':        (160, 0),
            'rotated_scaled':(240, 0),
            # 第二行：分解变体
            'exploded':      (0, -80),
            'exploded_rotated': (80, -80),
            'exploded_scaled':  (160, -80)
        }
        
        # 添加标题
        msp.add_text(
            'Normal Blocks', 
            dxfattribs={
                'layer': 'TEXT',
                'height': 5.0,
                'style': 'STANDARD',
                'insert': (0, 20)
            }
        )
        msp.add_text(
            'Exploded Blocks', 
            dxfattribs={
                'layer': 'TEXT',
                'height': 5.0,
                'style': 'STANDARD',
                'insert': (0, -60)
            }
        )
        
        # 添加所有变体
        for variant_name, pos in positions.items():
            if variant_name in variants:
                params = variants[variant_name]
                print(f"\n处理变体: {variant_name}")
                
                # 添加变体名称文本
                msp.add_text(
                    variant_name, 
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': 3.0,
                        'style': 'STANDARD',
                        'insert': (pos[0], pos[1] + 10)
                    }
                )
                
                if not params['exploded']:
                    # 添加块引用
                    for block in doc.blocks:
                        if not block.name.startswith('*'):
                            msp.add_blockref(block.name, pos, dxfattribs={
                                'layer': 'BLOCK',
                                'rotation': params['rotation'],
                                'xscale': params['scale']*global_scale,
                                'yscale': params['scale']*global_scale
                            })
                            print(f"添加了块引用变体: {variant_name}")
                            break
                else:
                    # 分解块
                    if source_block:
                        print(f"开始分解块: {variant_name}")
                        from math import radians, cos, sin
                        
                        # 计算变换参数
                        angle = radians(params['rotation'])
                        scale = params['scale'] * global_scale
                        dx, dy = pos[0], pos[1]
                        
                        # 复制并变换每个实体
                        for entity in source_block:
                            try:
                                new_entity = entity.copy()
                                
                                # 根据实体类型进行变换
                                if hasattr(new_entity, 'dxf'):
                                    if new_entity.dxftype() == 'LINE':
                                        # 变换起点
                                        start = new_entity.dxf.start
                                        x = start[0] * scale * cos(angle) - start[1] * scale * sin(angle) + dx
                                        y = start[0] * scale * sin(angle) + start[1] * scale * cos(angle) + dy
                                        new_entity.dxf.start = (x, y)
                                        
                                        # 变换终点
                                        end = new_entity.dxf.end
                                        x = end[0] * scale * cos(angle) - end[1] * scale * sin(angle) + dx
                                        y = end[0] * scale * sin(angle) + end[1] * scale * cos(angle) + dy
                                        new_entity.dxf.end = (x, y)
                                        
                                    elif new_entity.dxftype() in ['CIRCLE', 'ARC']:
                                        # 变换圆心
                                        center = new_entity.dxf.center
                                        x = center[0] * scale * cos(angle) - center[1] * scale * sin(angle) + dx
                                        y = center[0] * scale * sin(angle) + center[1] * scale * cos(angle) + dy
                                        new_entity.dxf.center = (x, y)
                                        new_entity.dxf.radius *= scale
                                        
                                        if new_entity.dxftype() == 'ARC':
                                            new_entity.dxf.start_angle += params['rotation']
                                            new_entity.dxf.end_angle += params['rotation']
                                            
                                    elif new_entity.dxftype() == 'LWPOLYLINE':
                                        # 变换多段线的所有顶点
                                        new_points = []
                                        for vertex in new_entity.get_points():
                                            x = vertex[0] * scale * cos(angle) - vertex[1] * scale * sin(angle) + dx
                                            y = vertex[0] * scale * sin(angle) + vertex[1] * scale * cos(angle) + dy
                                            # 保持原有的凸度和宽度属性（如果存在）
                                            if len(vertex) > 2:
                                                new_points.append((x, y) + vertex[2:])
                                            else:
                                                new_points.append((x, y))
                                        new_entity.set_points(new_points)
                                        
                                        # 如果有宽度属性，也需要缩放
                                        if hasattr(new_entity.dxf, 'const_width'):
                                            new_entity.dxf.const_width *= scale
                                
                                new_entity.dxf.layer = 'BLOCK'
                                msp.add_entity(new_entity)
                                
                            except Exception as e:
                                print(f"处理实体失败: {str(e)}")
                                
                        print(f"完成分解块: {variant_name}")
        
        print(f"保存文件到: {filename}")
        doc.saveas(filename)
        print("文件保存成功")

    @staticmethod
    def _create_preview(dxf_path: Path, output_path: Path, size=(800, 600)):
        """为DXF文件创建预览图"""
        print(f"\n开始生成预览图: {output_path}")
        print(f"源DXF文件: {dxf_path}")
        
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp)
        
        ax.set_axis_off()
        ax.autoscale_view()
        ax.set_aspect('equal')
        plt.margins(0.2)
        
        ax.set_facecolor('white')
        for artist in ax.get_children():
            if hasattr(artist, 'set_color'):
                artist.set_color('black')
                if hasattr(artist, 'set_linewidth'):
                    artist.set_linewidth(2.0)
        
        print(f"保存预览图到: {output_path}")
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if output_path.exists():
            print(f"预览图生成成功: {output_path}")
            print(f"文件大小: {output_path.stat().st_size} bytes")
        else:
            print(f"错误: 预览图未生成: {output_path}")

    @classmethod
    def tearDownClass(cls):
        """清理测试环境，但保留预览图"""
        # # 删除测试文件
        # for file_path in cls.test_files.values():
        #     if file_path.exists():
        #         file_path.unlink()
        # # 删除测试文件目录
        # cls.test_dir.rmdir()
        # print(f"\n预览图保留在目录: {cls.preview_dir.absolute()}")

    def test_load_entities(self):
        """测试实体加载功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        # 测试第一个导入的元件文件
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        self.assertGreater(len(network.entities), 0)

    def test_extract_block_patterns(self):
        """测试块模式提取功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        patterns = network.extract_block_patterns()
        
        # 验证提取的模式数量
        self.assertGreater(len(patterns), 0)
        
        # 验证模式的特征
        for pattern in patterns:
            self.assertIsInstance(pattern, BlockPattern)
            self.assertGreater(pattern.entity_count, 0)
            self.assertGreater(len(pattern.entity_types), 0)
            
            # 验证实体类型是否合理
            valid_types = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'POLYLINE'}
            self.assertTrue(any(t in valid_types for t in pattern.entity_types),
                           f"未找到有效的实体类型: {pattern.entity_types}")
            
            print(f"\n提取的块模式特征:")
            print(f"- 实体数量: {pattern.entity_count}")
            print(f"- 实体类型: {pattern.entity_types}")

    def test_find_matching_blocks(self):
        """测试块匹配功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        patterns = network.extract_block_patterns()
        
        for pattern in patterns:
            matches = network.find_matching_blocks(pattern)
            self.assertGreaterEqual(len(matches), 0)
            
            # 验证匹配的特征
            for match in matches:
                # 验证是块引用
                self.assertEqual(match.entity_type, 'INSERT')
                
                # 获取块的特征并验证
                block_features = network.get_block_features(match.block_name)
                self.assertIsNotNone(block_features)
                self.assertEqual(set(block_features['entity_types']), set(pattern.entity_types))
                
                print(f"\n匹配块的特征:")
                print(f"- 块名称: {match.block_name}")
                print(f"- 实体类型: {block_features['entity_types']}")

    def test_find_similar_entity_groups(self):
        """测试相似实体组查找功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        patterns = network.extract_block_patterns()
        
        for pattern in patterns:
            groups = network.find_similar_entity_groups(pattern)
            self.assertGreaterEqual(len(groups), 0)
            
            # 验证每个组的特征
            for group in groups:
                # 验证组中的实体数量
                self.assertEqual(len(group.entities), pattern.entity_count)
                
                # 验证组中的实体类型
                group_types = set(entity.entity_type for entity in group.entities)
                self.assertEqual(group_types, set(pattern.entity_types))
                
                # 验证组的边界框
                self.assertIsNotNone(group.bounding_box)
                
                print(f"\n相似组的特征:")
                print(f"- 实体数量: {len(group.entities)}")
                print(f"- 实体类型: {group_types}")
                print(f"- 边界框: {group.bounding_box}")

    def test_get_block_features(self):
        """测试块特征提取功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        
        # 获取所有块的特征
        for pattern in network.extract_block_patterns():
            features = network.get_block_features(pattern.name)
            self.assertIsNotNone(features)
            
            # 验证特征的完整性
            self.assertIn('entity_count', features)
            self.assertIn('entity_types', features)
            self.assertIn('bounding_box', features)
            
            # 验证特征的有效性
            self.assertGreater(features['entity_count'], 0)
            self.assertGreater(len(features['entity_types']), 0)
            self.assertIsNotNone(features['bounding_box'])
            
            # 验证边界框格式
            self.assertEqual(len(features['bounding_box']), 2)
            self.assertEqual(len(features['bounding_box'][0]), 2)  # min point (x,y)
            self.assertEqual(len(features['bounding_box'][1]), 2)  # max point (x,y)
            
            print(f"\n块的特征:")
            print(f"- 实体数量: {features['entity_count']}")
            print(f"- 实体类型: {features['entity_types']}")
            print(f"- 边界框: {features['bounding_box']}")

    def test_composite_entity_creation(self):
        """测试复合实体创建功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        
        # 创建复合实体（使用所有可用的实体）
        entities = network.entities
        composite = network.create_composite_entity("test_composite", entities)
        
        self.assertIsInstance(composite, CompositeEntity)
        self.assertEqual(len(composite.entities), len(entities))
        self.assertEqual(composite.name, "test_composite")

    def test_bounding_box_calculation(self):
        """测试边界框计算功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        block_name = next(iter(self.imported_features))
        network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
        
        # 测试单个块的边界框
        for entity in network.entities:
            if entity.entity_type == 'INSERT':
                features = network.get_block_features(entity.block_name)
                self.assertIsNotNone(features['bounding_box'])

    def test_preview_generation(self):
        """测试预览图生成"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            preview_path = self.preview_dir / f"{block_name}.png"
            self.assertTrue(preview_path.exists(), 
                          f"预览图 {preview_path} 未生成")
            self.assertGreater(preview_path.stat().st_size, 0, 
                             f"预览图 {preview_path} 是空文件")

    def test_imported_blocks(self):
        """测试导入的单个元件块"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name, features in self.imported_features.items():
            # 测试基本特征
            self.assertIsNotNone(features)
            self.assertIn('name', features)
            self.assertIn('entity_count', features)
            self.assertIn('entity_types', features)
            self.assertIn('bounding_box', features)
            
            # 测试实体数量
            self.assertGreater(features['entity_count'], 0)
            
            # 测试实体类型
            self.assertGreater(len(features['entity_types']), 0)
            
            # 测试边界框
            self.assertIsNotNone(features['bounding_box'])
            self.assertEqual(len(features['bounding_box']), 2)  # 两个点定义边界框
            
            print(f"\n测试块 {block_name} 的特征:")
            print(f"- 实体数量: {features['entity_count']}")
            print(f"- 实体类型: {features['entity_types']}")
            print(f"- 边界框: {features['bounding_box']}")

    def test_find_variants(self):
        """测试使用原始特征查找变体文件中的各个变体位置"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
            
        # 预期的变体位置
        expected_positions = {
            'basic':            (0, 0),
            'rotated':          (80, 0),
            'scaled':           (160, 0),
            'rotated_scaled':   (240, 0),
            'exploded':         (0, -80),
            'exploded_rotated': (80, -80),
            'exploded_scaled':  (160, -80)
        }
        
        for block_name, original_features in self.imported_features.items():
            print(f"\n测试元件 {block_name} 的变体定位:")
            
            # 获取变体文件
            variant_file = self.test_files[block_name]
            network = EntityNetwork(str(variant_file))
            
            # 从原始特征创建模式
            pattern = BlockPattern.from_block_features(original_features)
            self.assertIsNotNone(pattern, f"无法从原始特征创建模式: {block_name}")
            
            # 查找所有匹配的块引用和实体组
            blocks = network.find_matching_blocks(pattern)
            similar_groups = network.find_similar_entity_groups(pattern)
            
            # 验证找到的变体数量
            total_variants = len([v for v in expected_positions.keys() 
                                if not v.startswith('exploded')])
            total_exploded = len([v for v in expected_positions.keys() 
                                if v.startswith('exploded')])
            
            self.assertEqual(
                len(blocks), 
                total_variants,
                f"块引用变体数量不匹配: 期望 {total_variants}, 实际 {len(blocks)}"
            )
            self.assertEqual(
                len(similar_groups),
                total_exploded,
                f"分解变体数量不匹配: 期望 {total_exploded}, 实际 {len(similar_groups)}"
            )
            
            # 验证每个变体的位置
            found_positions = set()
            
            # 检查块引用变体
            for block in blocks:
                pos = (round(block.position[0]), round(block.position[1]))
                found_positions.add(pos)
                
                # 查找对应的变体名称
                variant_name = next(
                    (name for name, exp_pos in expected_positions.items()
                     if abs(exp_pos[0] - pos[0]) < 0.1 and 
                        abs(exp_pos[1] - pos[1]) < 0.1 and
                        not name.startswith('exploded')),
                    None
                )
                
                self.assertIsNotNone(
                    variant_name,
                    f"在位置 {pos} 找到的块引用不匹配任何预期变体"
                )
                print(f"找到变体 {variant_name} 在位置 {pos}")
            
            # 检查分解变体
            for group in similar_groups:
                bbox = CompositeEntity("temp", group).get_bounding_box()
                self.assertIsNotNone(bbox, "实体组没有有效的边界框")
                
                center = (
                    round((bbox[0][0] + bbox[1][0]) / 2),
                    round((bbox[0][1] + bbox[1][1]) / 2)
                )
                found_positions.add(center)
                
                # 查找对应的变体名称
                variant_name = next(
                    (name for name, exp_pos in expected_positions.items()
                     if abs(exp_pos[0] - center[0]) < 0.1 and 
                        abs(exp_pos[1] - center[1]) < 0.1 and
                        name.startswith('exploded')),
                    None
                )
                
                self.assertIsNotNone(
                    variant_name,
                    f"在位置 {center} 找到的实体组不匹配任何预期变体"
                )
                print(f"找到分解变体 {variant_name} 在位置 {center}")
            
            # 验证是否找到了所有变体
            expected_pos_set = set(expected_positions.values())
            self.assertEqual(
                len(found_positions),
                len(expected_positions),
                f"找到的变体数量不正确: 期望 {len(expected_positions)}, 实际 {len(found_positions)}"
            )
            self.assertEqual(
                found_positions,
                {(x, y) for x, y in expected_positions.values()},
                "找到的变体位置集合与预期不匹配"
            )

if __name__ == '__main__':
    unittest.main(verbosity=0)