import unittest
import ezdxf
from Entity import EntityNetwork, EntityInfo, AttributeInfo  # pylint: disable=no-name-in-module

# pylint: disable=no-member

class TestEntityLoading(unittest.TestCase):
    def setUp(self):
        # 创建测试DXF文档
        self.doc = ezdxf.new()
        self.msp = self.doc.modelspace()
        
        # 创建块定义
        block = self.doc.blocks.new('test_block')
        
        # 创建INSERT实体
        self.insert = self.msp.add_blockref('test_block', (0, 0))
        
        # 添加属性
        self.attrib1 = self.insert.add_attrib('TAG1', 'Value1', (10, 10))
        self.attrib2 = self.insert.add_attrib('TAG2', 'Value2', (20, 20))
        
        # 添加独立ATTRIB实体
        self.standalone_attrib = self.insert.add_attrib('TAG3', 'Value3', (30, 30))
        
        # 保存测试文件
        self.test_file = 'test_entity_loading.dxf'
        self.doc.saveas(self.test_file)
        
    def test_insert_entity_loading(self):
        network = EntityNetwork(self.test_file)
        entities = [e for e in network.entities if e.entity_type == 'INSERT']
        
        self.assertEqual(len(entities), 1)
        insert_entity = entities[0]
        
        # 验证属性数量
        self.assertEqual(len(insert_entity.attributes), 3)
        
        # 验证属性信息
        attrib1 = insert_entity.get_attribute('TAG1')
        self.assertIsNotNone(attrib1)
        self.assertEqual(attrib1.value, 'Value1')
        self.assertEqual(attrib1.position[:2], (10.0, 10.0))  # 只比较x,y坐标
        
        attrib2 = insert_entity.get_attribute('TAG2')
        self.assertIsNotNone(attrib2)
        self.assertEqual(attrib2.value, 'Value2')
        self.assertEqual(attrib2.position[:2], (20.0, 20.0))  # 只比较x,y坐标
        
    def test_standalone_attrib_loading(self):
        network = EntityNetwork(self.test_file)
        entities = [e for e in network.entities if e.entity_type == 'INSERT']
        insert_entity = entities[0]
        
        # 验证独立ATTRIB实体是否被正确加载
        attrib3 = insert_entity.get_attribute('TAG3')
        self.assertIsNotNone(attrib3)
        self.assertEqual(attrib3.value, 'Value3')
        self.assertEqual(attrib3.position[:2], (30.0, 30.0))  # 只比较x,y坐标
        
    def tearDown(self):
        # 清理测试文件
        import os
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()