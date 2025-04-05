import os
import sys
from pprint import pprint

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.dxf_parser import EzdxfBackend, DXFParseError

def test_ezdxf_backend(dxf_file_path):
    """测试EzdxfBackend的功能"""
    print("开始测试EzdxfBackend...")
    
    # 创建后端实例
    backend = EzdxfBackend()
    
    try:
        # 测试加载文件
        print("\n1. 测试加载DXF文件")
        result = backend.load_file(dxf_file_path)
        print(f"文件加载成功: {result}")
        
        # 测试获取实体
        print("\n2. 测试获取实体")
        entities = backend.get_entities()
        print(f"找到 {len(entities)} 个实体")
        if entities:
            print("第一个实体示例:")
            pprint(entities[0])
        
        # 测试获取块
        print("\n3. 测试获取块")
        blocks = backend.get_blocks()
        print(f"找到 {len(blocks)} 个块定义")
        if blocks:
            block_name = next(iter(blocks))
            print(f"块 '{block_name}' 示例:")
            pprint(blocks[block_name])
            
            # 测试获取块中的实体
            print(f"\n4. 测试获取块 '{block_name}' 中的实体")
            block_entities = backend.get_block_entities(block_name)
            print(f"找到 {len(block_entities)} 个实体")
            if block_entities:
                print("第一个块实体示例:")
                pprint(block_entities[0])
        
        # 测试获取块引用
        print("\n5. 测试获取块引用")
        inserts = backend.get_block_inserts()
        print(f"找到 {len(inserts)} 个块引用")
        if inserts:
            print("第一个块引用示例:")
            pprint(inserts[0])
            
            # 测试获取属性
            if hasattr(backend, 'get_attributes') and inserts[0].get('handle'):
                print(f"\n6. 测试获取块引用属性")
                attributes = backend.get_attributes(inserts[0]['handle'])
                print(f"找到 {len(attributes)} 个属性")
                if attributes:
                    print("第一个属性示例:")
                    pprint(attributes[0])
        
    except DXFParseError as e:
        print(f"解析错误: {e}")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dxf_file = sys.argv[1]
    else:
        dxf_file = input("请输入DXF文件路径: ")
    
    test_ezdxf_backend(dxf_file) 