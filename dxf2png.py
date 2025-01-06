import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

def dxf_to_image(dxf_path: str, output_path: str, size=(800, 600)):
    """将DXF文件转换为图像文件
    
    Args:
        dxf_path: DXF文件路径
        output_path: 输出图像路径
        size: 输出图像大小,默认 800x600
    """
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
    ax.set_facecolor('white')  # 设置白色背景
    for artist in ax.get_children():
        if hasattr(artist, 'set_color'):
            artist.set_color('black')  # 设置线条为黑色
            if hasattr(artist, 'set_linewidth'):
                artist.set_linewidth(1.0)  # 加粗线条
    
    # 保存图像
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    dxf_to_image("/Users/hhq/Downloads/单元素研究_阀门/单个模块-一个阀门-在0图层.dxf", "output.png")