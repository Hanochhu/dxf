import unittest
import os
from Entity import EntityNetwork, BlockPattern, CompositeEntity, EntityInfo
import ezdxf
from pathlib import Path
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from math import cos, sin, radians

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
                print(f"\n处理文件: {dxf_file.name}")
                
                # 导入DXF文件
                network = EntityNetwork(str(dxf_file))
                print(f"- 成功创建 EntityNetwork")
                
                # 提取块模式
                patterns = network.extract_block_patterns()
                print(f"- 找到 {len(patterns)} 个块模式")
                
                if patterns:
                    # 获取第一个块的特征
                    pattern = patterns[0]
                    print(f"- 第一个块模式: {pattern.name}")
                    
                    features = network.get_block_features(pattern.name)
                    if features:
                        block_features[dxf_file.stem] = features
                        print(f"√ 成功提取特征")
                        print(f"  - 实体数量: {features['entity_count']}")
                        print(f"  - 实体类型: {features['entity_types']}")
                        print(f"  - 边界框: {features['bounding_box']}")
                    else:
                        print(f"× 未能获取块特征")
                else:
                    print(f"× 未找到块模式")
                    
            except Exception as e:
                print(f"× 处理文件失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        print(f"\n总共导入 {len(block_features)} 个元件特征")
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
        # print(f"创建预览图目录: {cls.preview_dir.absolute()}")
        
        # 创建测试文件目录
        cls.test_dir = Path("test_dxf_files")
        if cls.test_dir.exists():
            for file in cls.test_dir.glob("*"):
                file.unlink()
        else:
            cls.test_dir.mkdir(exist_ok=True)
        # print(f"创建测试文件目录: {cls.test_dir.absolute()}")
        
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
            # print("\n开始导入单个元件DXF文件...")
            cls.imported_features = cls._import_single_blocks(cls.single_blocks_dir)
            # print(f"共导入 {len(cls.imported_features)} 个元件特征")
            
            # 为每个导入的元件生成变体和预览图
            cls.test_files = {}
            failed_blocks = set()  # 用于记录处理失败的块
            
            for block_name, features in cls.imported_features.items():
                try:
                    # print(f"\n处理元件: {block_name}")
                    source_dxf = cls.single_blocks_dir / f"{block_name}.dxf"
                    
                    # 检查源文件是否存在
                    if not source_dxf.exists():
                        print(f"源文件不存在: {source_dxf}")
                        failed_blocks.add(block_name)
                        continue
                    
                    # 生成变体文件 - 使用 global_scale=20.0
                    variants_file = cls.test_dir / f"{block_name}_variants.dxf"
                    cls._create_test_valve_dxf(variants_file, str(source_dxf), variants, global_scale=global_scale)
                    cls.test_files[block_name] = variants_file
                    
                    # 验证变体文件是否生成
                    if variants_file.exists():
                        # print(f"变体文件已生成: {variants_file}")
                        
                        # 生成变体文件的预览图
                        variants_preview = cls.preview_dir / f"{block_name}.png"
                        cls._create_preview(variants_file, variants_preview)
                        
                        # 验证预览图是否生成
                        if variants_preview.exists():
                            # print(f"变体预览图已生成: {variants_preview}")
                            pass
                        # else:
                            # print(f"错误: 预览图未生成: {variants_preview}")
                    # else:
                        # print(f"错误: 变体文件未生成: {variants_file}")
                        
                except Exception as e:
                    print(f"处理文件失败 {block_name}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    failed_blocks.add(block_name)
            
            # 迭代结束后再删除失败的块
            for block_name in failed_blocks:
                del cls.imported_features[block_name]
        else:
            cls.imported_features = {}
            # print("\n未找到单个元件目录，跳过导入")

        cls.test_stats = {}  # Initialize statistics dictionary

    @staticmethod
    def _create_test_valve_dxf(filename, source_dxf, variants, global_scale=20.0):
        """创建单个元件的所有变体测试文件"""
        # print(f"\n开始创建变体文件: {filename}")
        # print(f"源文件: {source_dxf}")
        
        # 读取源文件以获取块定义
        source_doc = ezdxf.readfile(source_dxf, encoding='utf-8')
        # print("成功读取源文件")
        
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
                # print(f"复制块: {block.name}")
        # print(f"复制了 {block_count} 个块定义")
        
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
                # print(f"\n处理变体: {variant_name}")
                
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
                            # print(f"添加了块引用变体: {variant_name}")
                            break
                else:
                    # 分解块
                    if source_block:
                        # print(f"开始分解块: {variant_name}")
                        
                        # 计算变换参数
                        angle = radians(params['rotation'])
                        scale = params['scale'] * global_scale
                        dx, dy = pos[0], pos[1]
                        
                        # 递归分解块中的所有实体
                        for entity in source_block:
                            _explode_entity(entity, doc, msp, angle, scale, dx, dy, params['rotation'])
        
        # print(f"保存文件到: {filename}")
        doc.saveas(filename)
        # print("文件保存成功")

    @staticmethod
    def _create_preview(dxf_path: Path, output_path: Path, size=(800, 600)):
        """为DXF文件创建预览图"""
        # print(f"\n开始生成预览图: {output_path}")
        # print(f"源DXF文件: {dxf_path}")
        
        doc = ezdxf.readfile(str(dxf_path), encoding='utf-8')
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
        
        # print(f"保存预览图到: {output_path}")
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if output_path.exists():
            # print(f"预览图生成成功: {output_path}")
            # print(f"文件大小: {output_path.stat().st_size} bytes")
            pass
        # else:
            # print(f"错误: 预览图未生成: {output_path}")

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

        instance = cls()
        instance._print_test_stats()  # 在所有测试结束后打印统计信息

    def _record_test_result(self, test_name, block_name, success=True, error=None):
        """Record test results for individual blocks"""
        if test_name not in self.test_stats:
            self.test_stats[test_name] = {
                'total': 0,
                'passed': 0,
                'failed': {},
                'errors': {}
            }
        
        self.test_stats[test_name]['total'] += 1
        if success:
            self.test_stats[test_name]['passed'] += 1
        else:
            self.test_stats[test_name]['failed'][block_name] = error
            self.test_stats[test_name]['errors'][block_name] = str(error)

    @classmethod
    def _print_test_stats(cls):
        """打印所有测试的详细统计信息"""
        print("\n详细测试统计信息:")
        for test_name, stats in cls.test_stats.items():
            print(f"\n测试: {test_name}")
            print(f"  总测试元件数: {stats['total']}")
            print(f"  通过测试的元件数: {stats['passed']}")
            if stats['failed']:
                print(f"  未通过测试的元件数: {len(stats['failed'])}")
                print("  未通过测试的元件及原因:")
                for block_name, error in stats['failed'].items():
                    print(f"    - {block_name}: {error}")

    def test_load_entities(self):
        """测试实体加载功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
                self.assertGreater(len(network.entities), 0)
                self._record_test_result('test_load_entities', block_name)
            except AssertionError as e:
                self._record_test_result('test_load_entities', block_name, success=False, error=e)

    def test_extract_block_patterns(self):
        """测试块模式提取功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
                patterns = network.extract_block_patterns()
                self.assertGreater(len(patterns), 0)
                self._record_test_result('test_extract_block_patterns', block_name)
            except AssertionError as e:
                self._record_test_result('test_extract_block_patterns', block_name, success=False, error=e)

    def test_find_matching_blocks(self):
        """测试块匹配功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
                patterns = network.extract_block_patterns()
                
                for pattern in patterns:
                    matches = network.find_matching_blocks(pattern)
                    self.assertGreater(len(matches), 0)
                    
                    # 验证匹配的特征
                    for match in matches:
                        # 验证是块引用
                        self.assertEqual(match.entity_type, 'INSERT')
                        
                        # 获取块的特征并验证
                        block_features = network.get_block_features(match.block_name)
                        self.assertIsNotNone(block_features)
                        self.assertEqual(set(block_features['entity_types']), set(pattern.entity_types))
                        
                        # print(f"\n匹配块的特征:")
                        # print(f"- 块名称: {match.block_name}")
                        # print(f"- 实体类型: {block_features['entity_types']}")
                
                self._record_test_result('test_find_matching_blocks', block_name)
            except AssertionError as e:
                self._record_test_result('test_find_matching_blocks', block_name, success=False, error=e)

    def test_find_similar_entity_groups(self):
        """测试相似实体组查找功能，包括分解成普通线条的场景"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                # 加载测试文件
                variant_file = self.test_files[block_name]
                network = EntityNetwork(str(variant_file))
                
                # 从原始特征创建模式
                original_features = self.imported_features[block_name]
                pattern = BlockPattern.from_block_features(original_features)
                self.assertIsNotNone(pattern, f"无法从原始特征创建模式: {block_name}")
                
                # 查找相似实体组
                groups = network.find_similar_entity_groups(pattern)
                
                # 打印调试信息
                print(f"\n调试信息 - 块: {block_name}")
                print(f"找到的相似组数量: {len(groups)}")
                
                # 验证至少找到一个组（分解后的实体组）
                self.assertGreater(len(groups), 0, f"块 {block_name} 未找到任何相似实体组")
                
                # 验证每个组的特征
                for i, group in enumerate(groups):
                    # 验证组中的实体数量
                    self.assertEqual(len(group.entities), pattern.entity_count,
                                   f"块 {block_name} 的组 {i + 1} 实体数量不匹配: 期望 {pattern.entity_count}, 实际 {len(group.entities)}")
                    
                    # 验证组中的实体类型
                    group_types = set(entity.entity_type for entity in group.entities)
                    self.assertEqual(group_types, set(pattern.entity_types),
                                   f"块 {block_name} 的组 {i + 1} 实体类型不匹配: 期望 {set(pattern.entity_types)}, 实际 {group_types}")
                    
                    # 验证组的边界框
                    self.assertIsNotNone(group.bounding_box,
                                       f"块 {block_name} 的组 {i + 1} 没有有效的边界框")
                    
                    # 打印组详细信息
                    print(f"\n组 {i + 1} 的详细信息:")
                    print(f"- 实体数量: {len(group.entities)}")
                    print(f"- 实体类型: {group_types}")
                    print(f"- 边界框: {group.bounding_box}")
                    print(f"- 实体列表:")
                    for entity in group.entities:
                        print(f"  - {entity.entity_type}: {entity.position}")
                
                self._record_test_result('test_find_similar_entity_groups', block_name)
            except AssertionError as e:
                self._record_test_result('test_find_similar_entity_groups', block_name, success=False, error=e)

    def test_get_block_features(self):
        """测试块特征提取功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
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
                    
                    # print(f"\n块的特征:")
                    # print(f"- 实体数量: {features['entity_count']}")
                    # print(f"- 实体类型: {features['entity_types']}")
                    # print(f"- 边界框: {features['bounding_box']}")
                
                self._record_test_result('test_get_block_features', block_name)
            except AssertionError as e:
                self._record_test_result('test_get_block_features', block_name, success=False, error=e)

    def test_composite_entity_creation(self):
        """测试复合实体创建功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
                
                # 创建复合实体（使用所有可用的实体）
                entities = network.entities
                composite = network.create_composite_entity("test_composite", entities)
                
                self.assertIsInstance(composite, CompositeEntity)
                self.assertEqual(len(composite.entities), len(entities))
                self.assertEqual(composite.name, "test_composite")
                
                self._record_test_result('test_composite_entity_creation', block_name)
            except AssertionError as e:
                self._record_test_result('test_composite_entity_creation', block_name, success=False, error=e)

    def test_bounding_box_calculation(self):
        """测试边界框计算功能"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                network = EntityNetwork(str(self.single_blocks_dir / f"{block_name}.dxf"))
                
                # 测试单个块的边界框
                for entity in network.entities:
                    if entity.entity_type == 'INSERT':
                        features = network.get_block_features(entity.block_name)
                        self.assertIsNotNone(features['bounding_box'])
                
                self._record_test_result('test_bounding_box_calculation', block_name)
            except AssertionError as e:
                self._record_test_result('test_bounding_box_calculation', block_name, success=False, error=e)

    def test_preview_generation(self):
        """测试预览图生成"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name in self.imported_features:
            try:
                preview_path = self.preview_dir / f"{block_name}.png"
                self.assertTrue(preview_path.exists(), 
                              f"预览图 {preview_path} 未生成")
                self.assertGreater(preview_path.stat().st_size, 0, 
                                 f"预览图 {preview_path} 是空文件")
                
                self._record_test_result('test_preview_generation', block_name)
            except AssertionError as e:
                self._record_test_result('test_preview_generation', block_name, success=False, error=e)

    def test_imported_blocks(self):
        """测试导入的单个元件块"""
        if not self.imported_features:
            self.skipTest("没有导入的元件可供测试")
        
        for block_name, features in self.imported_features.items():
            try:
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
                
                # print(f"\n测试块 {block_name} 的特征:")
                # print(f"- 实体数量: {features['entity_count']}")
                # print(f"- 实体类型: {features['entity_types']}")
                # print(f"- 边界框: {features['bounding_box']}")
                
                self._record_test_result('test_imported_blocks', block_name)
            except AssertionError as e:
                self._record_test_result('test_imported_blocks', block_name, success=False, error=e)

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
        
        # 统计结果
        total_blocks = len(self.imported_features)
        passed_blocks = 0
        failed_blocks = {}
        
        # 遍历所有元件进行测试
        for block_name, original_features in self.imported_features.items():
            try:
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
                    f"块 {block_name} 的块引用变体数量不匹配: 期望 {total_variants}, 实际 {len(blocks)}"
                )
                self.assertEqual(
                    len(similar_groups),
                    total_exploded,
                    f"块 {block_name} 的分解变体数量不匹配: 期望 {total_exploded}, 实际 {len(similar_groups)}"
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
                        f"块 {block_name} 在位置 {pos} 找到的块引用不匹配任何预期变体"
                    )
                
                # 检查分解变体
                for group in similar_groups:
                    bbox = CompositeEntity("temp", group).get_bounding_box()
                    self.assertIsNotNone(bbox, f"块 {block_name} 的实体组没有有效的边界框")
                    
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
                        f"块 {block_name} 在位置 {center} 找到的实体组不匹配任何预期变体"
                    )
                
                # 验证是否找到了所有变体
                expected_pos_set = set(expected_positions.values())
                self.assertEqual(
                    len(found_positions),
                    len(expected_positions),
                    f"块 {block_name} 找到的变体数量不正确: 期望 {len(expected_positions)}, 实际 {len(found_positions)}"
                )
                self.assertEqual(
                    found_positions,
                    {(x, y) for x, y in expected_positions.values()},
                    f"块 {block_name} 找到的变体位置集合与预期不匹配"
                )
                
                # 如果所有断言通过，增加通过计数
                passed_blocks += 1
                
                self._record_test_result('test_find_variants', block_name)
            except AssertionError as e:
                # 记录失败信息
                failed_blocks[block_name] = str(e)
                self._record_test_result('test_find_variants', block_name, success=False, error=e)
        
        # 移除直接输出，统一通过 _print_test_stats 输出

def _explode_entity(entity, doc, msp, angle, scale, dx, dy, rotation):
    """递归分解实体及其嵌套块
    
    Args:
        entity: 要分解的实体
        doc: DXF文档对象
        msp: 目标模型空间
        angle: 旋转角度（弧度）
        scale: 缩放比例
        dx, dy: 平移距离
        rotation: 旋转角度（度）
    """
    try:
        # 打印实体类型和属性
        print(f"处理实体: {entity.dxftype()}")
        if hasattr(entity, 'dxf'):
            print(f"  属性: {[attr for attr in dir(entity.dxf) if not attr.startswith('_')]}")
        
        new_entity = entity.copy()
        
        # 如果实体是 INSERT，递归分解嵌套块
        if new_entity.dxftype() == 'INSERT':
            nested_block_name = new_entity.dxf.name
            if nested_block_name in doc.blocks:
                # 获取块的插入点和变换参数
                insert_pos = new_entity.dxf.insert
                insert_rotation = new_entity.dxf.rotation if hasattr(new_entity.dxf, 'rotation') else 0
                insert_scale = new_entity.dxf.xscale if hasattr(new_entity.dxf, 'xscale') else 1.0
                
                # 计算新的变换参数
                # 1. 先计算插入点的新位置（考虑旋转和缩放）
                new_x = insert_pos[0] * scale * cos(angle) - insert_pos[1] * scale * sin(angle) + dx
                new_y = insert_pos[0] * scale * sin(angle) + insert_pos[1] * scale * cos(angle) + dy
                
                # 2. 计算组合旋转角度和缩放比例
                new_angle = angle + radians(insert_rotation)
                new_scale = scale * insert_scale
                
                # 递归分解嵌套块中的每个实体
                for nested_entity in doc.blocks.get(nested_block_name):
                    _explode_entity(nested_entity, doc, msp, new_angle, new_scale, new_x, new_y, rotation + insert_rotation)
            else:
                print(f"警告: 嵌套块 '{nested_block_name}' 不存在")
        else:
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
                        new_entity.dxf.start_angle += rotation
                        new_entity.dxf.end_angle += rotation
                        
                elif new_entity.dxftype() == 'LWPOLYLINE':
                    # 获取所有顶点数据（包括凸度和宽度）
                    vertices = list(new_entity.vertices())
                    # 变换多段线的所有顶点
                    new_points = []
                    for vertex in vertices:
                        # vertex 是一个 (x, y, [bulge], [start_width, end_width], [id]) 元组
                        x = vertex[0] * scale * cos(angle) - vertex[1] * scale * sin(angle) + dx
                        y = vertex[0] * scale * sin(angle) + vertex[1] * scale * cos(angle) + dy
                        
                        # 保持原有的凸度和宽度属性
                        point = [x, y]
                        if len(vertex) > 2:  # 有凸度
                            point.append(vertex[2])
                        if len(vertex) > 3:  # 有宽度
                            start_width = vertex[3] * scale if vertex[3] is not None else None
                            end_width = vertex[4] * scale if vertex[4] is not None else None
                            point.extend([start_width, end_width])
                        new_points.append(tuple(point))
                    
                    # 设置新的顶点
                    new_entity.set_points(new_points)
                    
                    # 如果有全局宽度，也需要缩放
                    if hasattr(new_entity.dxf, 'const_width'):
                        new_entity.dxf.const_width *= scale
                
                elif new_entity.dxftype() == 'ATTDEF':
                    try:
                        # 打印详细的ATTDEF属性信息
                        print(f"处理ATTDEF实体:")
                        print(f"  插入点: {new_entity.dxf.insert}")
                        print(f"  对齐点: {new_entity.dxf.align_point if hasattr(new_entity.dxf, 'align_point') else 'None'}")
                        print(f"  高度: {new_entity.dxf.height}")
                        print(f"  旋转: {new_entity.dxf.rotation if hasattr(new_entity.dxf, 'rotation') else 0}")
                        
                        # 变换插入点
                        insert = new_entity.dxf.insert
                        x = insert[0] * scale * cos(angle) - insert[1] * scale * sin(angle) + dx
                        y = insert[0] * scale * sin(angle) + insert[1] * scale * cos(angle) + dy
                        new_entity.dxf.insert = (x, y)
                        
                        # 如果有对齐点，也需要变换
                        if hasattr(new_entity.dxf, 'align_point') and new_entity.dxf.align_point is not None:
                            align = new_entity.dxf.align_point
                            if isinstance(align, (tuple, list)) and len(align) >= 2:
                                x = align[0] * scale * cos(angle) - align[1] * scale * sin(angle) + dx
                                y = align[0] * scale * sin(angle) + align[1] * scale * cos(angle) + dy
                                new_entity.dxf.align_point = (x, y)
                            else:
                                print(f"  警告: 对齐点格式无效: {align}")
                        
                        # 调整文本高度和旋转
                        new_entity.dxf.height *= scale
                        if hasattr(new_entity.dxf, 'rotation'):
                            new_entity.dxf.rotation += rotation
                        
                        # 如果有宽度，也需要缩放
                        if hasattr(new_entity.dxf, 'width'):
                            new_entity.dxf.width *= scale
                            
                    except Exception as e:
                        print(f"处理ATTDEF实体时出错: {str(e)}")
                        # 继续处理，不中断程序
                
                # 设置图层并添加实体
                new_entity.dxf.layer = 'BLOCK'
                msp.add_entity(new_entity)
            else:
                print(f"警告: 实体没有dxf属性: {new_entity.dxftype()}")
            
    except Exception as e:
        print(f"处理实体失败: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    unittest.main(verbosity=1)  # 设置 verbosity=1，只显示统计信息