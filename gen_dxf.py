import ezdxf
from Entity import EntityNetwork

def create_test_valves_dxf(filename: str):
    # 创建新的DXF文档
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # 创建图层
    doc.layers.new('VALVE', dxfattribs={'color': 1})
    
    # 创建VG2 - 圆形阀门
    vg2 = doc.blocks.new(name='VG2')
    vg2.add_circle((0, 0), radius=2.0)
    vg2.add_line((-3, 0), (-2, 0))
    vg2.add_line((2, 0), (3, 0))
    
    # 创建VG3 - 蝶阀
    vg3 = doc.blocks.new(name='VG3')
    vg3.add_circle((0, 0), radius=2.5)
    vg3.add_line((-2.5, 0), (2.5, 0))
    vg3.add_arc((0, 0), radius=1.5, start_angle=30, end_angle=150)
    
    # 创建VG4 - 组合阀门
    vg4 = doc.blocks.new(name='VG4')
    vg4.add_circle((0, 0), radius=2.0)
    vg4.add_line((-4, 0), (4, 0))
    vg4.add_line((0, -2), (0, 2))
    
    # 在模型空间中插入这些块
    # VG2
    msp.add_blockref('VG2', (0, 0))
    msp.add_blockref('VG2', (10, 0), dxfattribs={'rotation': 45})
    
    # VG3
    msp.add_blockref('VG3', (0, 10))
    msp.add_blockref('VG3', (10, 10), dxfattribs={'rotation': 90})
    
    # VG4
    msp.add_blockref('VG4', (0, 20))
    msp.add_blockref('VG4', (10, 20), dxfattribs={'xscale': 1.5})
    
    # 保存DXF文件
    doc.saveas(filename)

# 生成测试文件
create_test_valves_dxf('test_valves.dxf')

# 测试代码
network = EntityNetwork('test_valves.dxf')

# 提取模式
patterns = network.extract_block_patterns()

# 测试查找
for pattern in patterns:
    print(f"\n测试模式: {pattern.name}")
    matches = network.find_matching_blocks(pattern)
    print(f"找到 {len(matches)} 个匹配项")
    for match in matches:
        print(f"- 位置: {match.position}, 旋转: {match.rotation}")
