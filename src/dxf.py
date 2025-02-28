import ezdxf
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class DXFFile:
    def __init__(self, filename):
        self.filename = filename
        self.entities = []
        self.blocks = {}

    def parse(self):
        # 使用 ezdxf 读取 DXF 文件
        doc = ezdxf.readfile(self.filename)
        
        # 解析块定义
        for block in doc.blocks:
            self.blocks[block.name] = block

        # 解析模型空间中的所有线、圆和块引用
        for entity in tqdm(doc.modelspace().query('LINE CIRCLE INSERT'), desc="Parsing Entities"):
            self.entities.append(entity)

    def generate_output(self, output_filename):
        with open(output_filename, 'w') as output_file:
            output_file.write("DXF File Information:\n")
            
            # 首先记录普通块的定义（排除 Model_Space）
            output_file.write("\nBlock Definitions:\n")
            for block_name, block in self.blocks.items():
                if block_name != '*Model_Space':  # 跳过 Model_Space
                    output_file.write(f"  Block '{block_name}':\n")
                    # 打印所有实体类型，帮助调试
                    entity_types = set(entity.dxftype() for entity in block)
                    if entity_types:
                        output_file.write(f"    Entity types in block: {entity_types}\n")
                    
                    for entity in block:
                        if entity.dxftype() == 'LINE':
                            start = entity.dxf.start
                            end = entity.dxf.end
                            output_file.write(f"    Line from {start} to {end}\n")
                        elif entity.dxftype() == 'CIRCLE':
                            center = entity.dxf.center
                            radius = entity.dxf.radius
                            output_file.write(f"    Circle at {center} with radius {radius}\n")
                        # ... 其他实体类型的处理 ...
                        elif entity.dxftype() == 'INSERT':
                            name = entity.dxf.name
                            insertion_point = entity.dxf.insert
                            output_file.write(f"    Nested block '{name}' at {insertion_point}\n")

            # 然后记录 Model_Space 的内容
            output_file.write("\nModel Space Contents:\n")
            if '*Model_Space' in self.blocks:
                model_space = self.blocks['*Model_Space']
                entity_types = set(entity.dxftype() for entity in model_space)
                if entity_types:
                    output_file.write(f"  Entity types in Model Space: {entity_types}\n")
                
                for entity in model_space:
                    if entity.dxftype() == 'LINE':
                        start = entity.dxf.start
                        end = entity.dxf.end
                        output_file.write(f"  Line from {start} to {end}\n")
                    elif entity.dxftype() == 'CIRCLE':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        output_file.write(f"  Circle at {center} with radius {radius}\n")
                    # ... 其他实体类型的处理 ...
                    elif entity.dxftype() == 'INSERT':
                        name = entity.dxf.name
                        insertion_point = entity.dxf.insert
                        output_file.write(f"  Block '{name}' at {insertion_point}\n")

class Entity(ABC):
    @abstractmethod
    def get_coordinates(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

class Line(Entity):
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point

    def get_coordinates(self):
        return self.start_point, self.end_point

    def __str__(self):
        return f"Line from {self.start_point} to {self.end_point}"

class Circle(Entity):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_coordinates(self):
        return self.center, self.radius

    def __str__(self):
        return f"Circle at {self.center} with radius {self.radius}"

class Drawing:
    def __init__(self, entities, blocks):
        self.entities = entities
        self.blocks = blocks

    def draw(self):
        plt.figure(figsize=(8, 8))  # 创建一个绘图窗口
        for entity in tqdm(self.entities, desc="Drawing Entities"):
            if entity.dxftype() == 'LINE':
                self.draw_line(entity)
            elif entity.dxftype() == 'CIRCLE':
                self.draw_circle(entity)
            elif entity.dxftype() == 'INSERT':
                self.draw_block(entity)  # 处理块引用
        plt.axis('equal')  # 设置坐标轴比例相等
        plt.title("DXF Entities Visualization")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        plt.show()  # 显示绘图

    def draw_line(self, line, insertion_point=(0, 0)):
        start = line.dxf.start + insertion_point  # 应用插入点
        end = line.dxf.end + insertion_point  # 应用插入点
        plt.plot([start.x, end.x], [start.y, end.y], color='blue')  # 绘制线段

    def draw_circle(self, circle, insertion_point=(0, 0)):
        center = circle.dxf.center + insertion_point  # 应用插入点
        radius = circle.dxf.radius
        circle_patch = plt.Circle((center.x, center.y), radius, color='red', fill=False)  # 创建圆形
        plt.gca().add_patch(circle_patch)  # 添加圆形到绘图中

    def draw_block(self, block):
        block_name = block.dxf.name
        insertion_point = block.dxf.insert
        print(f"Drawing block '{block_name}' at {insertion_point}")  # 打印块信息

        # 获取块定义并绘制其中的实体
        if block_name in self.blocks:
            block_definition = self.blocks[block_name]
            for entity in block_definition:
                if entity.dxftype() == 'LINE':
                    self.draw_line(entity, insertion_point)  # 传递插入点
                elif entity.dxftype() == 'CIRCLE':
                    self.draw_circle(entity, insertion_point)  # 传递插入点
                # 可以添加更多的实体类型处理

# 示例用法
if __name__ == "__main__":
    source_dxf = "extracted_blocks/VALLGA.dxf"
    module_dxf = "图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf"
    line_dxf = "图例和流程图_仪表管件设备均为普通线条/2308PM-09-T3-2900.dxf"
    # 示例用法
    dxf_file = DXFFile(module_dxf)
    dxf_file.parse()

    # 生成输出文件
    dxf_file.generate_output('output.txt')

    # 绘制图形
    drawing = Drawing(dxf_file.entities, dxf_file.blocks)
    drawing.draw()