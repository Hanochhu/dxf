import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

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
    
    # 在模型空间中插入这些块，使用更大的间距以便清晰显示
    # VG2
    msp.add_blockref('VG2', (0, 0))
    msp.add_blockref('VG2', (20, 0), dxfattribs={'rotation': 45})
    
    # VG3
    msp.add_blockref('VG3', (0, 20))
    msp.add_blockref('VG3', (20, 20), dxfattribs={'rotation': 90})
    
    # VG4
    msp.add_blockref('VG4', (0, 40))
    msp.add_blockref('VG4', (20, 40), dxfattribs={'xscale': 1.5})
    
    # 保存DXF文件
    doc.saveas(filename)

def dxf_to_image(dxf_path: str, output_path: str, size=(800, 800)):
    """将DXF文件转换为图像文件"""
    # 读取DXF文件
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # 创建绘图上下文
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp)
    
    # 设置图像显示
    ax.set_axis_off()
    plt.margins(0, 0)
    
    # 自动调整视图以显示所有实体
    ax.autoscale_view()
    
    # 调整对比度
    ax.set_facecolor('white')
    for artist in ax.get_children():
        if hasattr(artist, 'set_color'):
            artist.set_color('black')
            if hasattr(artist, 'set_linewidth'):
                artist.set_linewidth(1.5)  # 稍微加粗线条以便更清晰
    
    # 保存图像
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    # 生成DXF文件
    create_test_valves_dxf('test_valves.dxf')
    
    # 转换为PNG图像
    dxf_to_image('test_valves.dxf', 'test_valves.png', size=(800, 800))
    
    print("已生成测试阀门图像：test_valves.png") 