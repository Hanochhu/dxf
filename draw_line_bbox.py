import ezdxf
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, LineString
import numpy as np
from Entity import *
from shapely.geometry import LineString, box, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString
class DXFVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.doc = ezdxf.readfile(file_path)
        self.msp = self.doc.modelspace()

        # 初始化数据存储
        self.lines = []
        self.insert_points_aa2 = []
        self.insert_points_other = []
        self.intersections = {}
        self.insert_point_closest_line = {}  # 存储 AA2 插入点与最接近的线段的关系

    def read_lines(self):
        """读取所有 LINE 实体并存储"""
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'brown', 'pink', 'cyan', 'magenta']  # 预定义颜色
        for idx, entity in enumerate(self.msp.query('LINE')):
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            # 为每条线段分配不同的颜色
            color = colors[idx % len(colors)]
            self.lines.append({"id": idx, "geometry": LineString([start, end]), "color": color})
        # 处理所有的 LWPOLYLINE 实体
        last_line_id = len(self.msp.query('LINE'))  # 获取最后一个 LINE 的 ID
        # 处理所有的 LWPOLYLINE 实体
        for idx, entity in enumerate(self.msp.query('LWPOLYLINE')):
            points = [(point[0], point[1]) for point in entity.vertices()]  # 修改这里，点是元组而非对象
            # 为每个多段线的线段分配不同的颜色
            color = colors[(idx + last_line_id) % len(colors)]  # 确保颜色不会重复
            for i in range(1, len(points)):
                start = points[i - 1]
                end = points[i]
                polyline_id = last_line_id + idx  # 设置 LWPOLYLINE 的 ID 为最后一个 LINE ID + 当前索引
                self.lines.append({"id": polyline_id, "geometry": LineString([start, end]), "color": color})
        
        
        return self.lines

    def find_intersections(self):
        """查找线段之间的交点"""
        intersections = {line["id"]: [] for line in self.lines}
        for i, line1 in enumerate(self.lines):
            for j, line2 in enumerate(self.lines):
                if i >= j:
                    continue  # 避免重复计算和自交点

                if line1["geometry"].intersects(line2["geometry"]):
                    point = line1["geometry"].intersection(line2["geometry"])
                    if not point.is_empty and point.geom_type == 'Point':
                        intersections[line1["id"]].append({"line_id": line2["id"], "point": (point.x, point.y)})
                        intersections[line2["id"]].append({"line_id": line1["id"], "point": (point.x, point.y)})

        self.intersections = intersections
        return self.intersections

    def read_inserts(self):
        """读取所有 INSERT 实体的插入点，并提取旋转角度"""
        for entity in self.msp:
            if entity.dxftype() == "INSERT":
                insert_point = entity.dxf.insert
                if entity.dxf.name == "AA2" or entity.dxf.name == "AA3" :
                    # 获取旋转角度，如果不存在则设为0
                    rotation_angle = getattr(entity.dxf, 'rotation', 0)  # 如果没有旋转角度，默认设为0
                    self.insert_points_aa2.append((insert_point.x, insert_point.y, rotation_angle,entity.dxf.name))
                else:
                    rotation_angle = getattr(entity.dxf, 'rotation', 0)  # 如果没有旋转角度，默认设为0
                    self.insert_points_other.append((insert_point.x, insert_point.y, rotation_angle,entity.dxf.name))
        return self.insert_points_aa2, self.insert_points_other

    @staticmethod
    def line_angle_with_axis(start, end, axis='x'):
        """计算线段与指定轴（x轴或y轴）的夹角"""
        dx = end[0] - start[0]  # 使用索引访问元组元素
        dy = end[1] - start[1]  # 使用索引访问元组元素
        if axis == 'x':
            return math.degrees(math.atan2(dy, dx))
        elif axis == 'y':
            return math.degrees(math.atan2(dx, dy))


    def find_closest_line_for_inserts(self):
        """查找每个 AA2 插入点最接近的线段（支持多个线段）"""
        for idx, point in enumerate(self.insert_points_aa2):
            point_geom = Point(point[0], point[1])
            rotation_angle = point[2]  # 获取旋转角度
            min_distance = float('inf')  # 初始化为一个很大的值
            closest_line_ids = []  # 存储所有最接近的线段 ID
            is_point_on_line = False  # 是否在线段上
    
            # 遍历所有线段，找到距离插入点最近的线段
            for line in self.lines:
                distance = point_geom.distance(line["geometry"])
                if distance < min_distance:
                    min_distance = distance
                    closest_line_ids = [line["id"]]  # 更新最接近的线段 ID 列表
                    # 检查点是否在线段上
                    if distance == 0:
                        is_point_on_line = True
                elif distance == min_distance:
                    closest_line_ids.append(line["id"])  # 如果距离相同，添加到结果列表
    
            # 确定与插入点旋转角度接近的方向
            filtered_line_ids = self.filter_lines_by_rotation(closest_line_ids, rotation_angle)
    
            # # 如果找到了符合条件的线段，使用它们
            # if filtered_line_ids:
            #     print(f"插入点 {point} 最接近的线段（根据旋转角度和夹角筛选）: {filtered_line_ids}")
            # else:
            #     print(f"插入点 {point} 找不到符合条件的线段")
    
            # 将筛选后的最接近的线段的 ID 和旋转角度一起记录
            self.insert_point_closest_line[idx] = {
                "line_ids": filtered_line_ids,  # 如果有多个最近的线段，存储所有的线段 ID
                "rotation_angle": rotation_angle,
                "is_point_on_line": is_point_on_line
            }
    
        return self.insert_point_closest_line

    def filter_lines_by_rotation(self, closest_line_ids, rotation_angle):
        """根据旋转角度过滤与坐标轴夹角小于10°的线段"""
        filtered_line_ids = []
        if rotation_angle % 360 < 10 or 170 < rotation_angle % 360 < 190:  # 旋转角度接近 0 或 180
            # 过滤与 x 轴夹角小于 10°的线段
            for line_id in closest_line_ids:
                line = self.lines[line_id]  # 获取线段
                angle = self.line_angle_with_axis(line["geometry"].coords[0], line["geometry"].coords[1], axis='x')
                if abs(angle) < 10 or abs(abs(angle) - 180) < 10:
                    filtered_line_ids.append(line_id)
        elif 80 < rotation_angle % 360 < 100 or 260 < rotation_angle % 360 < 280:  # 旋转角度接近 90 或 270
            # 过滤与 y 轴夹角小于 10°的线段
            for line_id in closest_line_ids:
                line = self.lines[line_id]  # 获取线段
                angle = self.line_angle_with_axis(line["geometry"].coords[0], line["geometry"].coords[1], axis='y')
                if abs(angle) < 10 or abs(abs(angle) - 180) < 10:
                    filtered_line_ids.append(line_id)

        return filtered_line_ids

    def build_line_info_dict(self):
        """通过 Insert Points AA2 Closest Line IDs with Rotation Angles 构建一个新的字典，
           键是 line_id，值是相关信息（如旋转角度、插入点信息等）
        """
        line_info_dict = {}
    
        # 遍历插入点与线段的关系
        for insert_idx, insert_info in self.insert_point_closest_line.items():
            rotation_angle = insert_info["rotation_angle"]
            is_point_on_line = insert_info["is_point_on_line"]
    
            # 遍历每个插入点关联的所有线段
            for line_id in insert_info["line_ids"]:
                if line_id not in line_info_dict:
                    line_info_dict[line_id] = {
                        "rotation_angles": [],  # 存储所有相关的旋转角度
                        "insert_points": [],    # 存储所有相关的插入点坐标
                        "is_point_on_line": is_point_on_line  # 插入点是否在线段上
                    }
    
                # 将该插入点的旋转角度添加到相应的线段信息中
                line_info_dict[line_id]["rotation_angles"].append(rotation_angle)
                line_info_dict[line_id]["insert_points"].append(self.insert_points_aa2[insert_idx])  # 使用插入点的坐标

        return line_info_dict

def visualize_rectangles(rectangles):
    fig, ax = plt.subplots(figsize=(10, 10))

    all_x = []
    all_y = []

    # 绘制所有矩形框
    for rect in rectangles:
        bbox = rect['bounding_box']
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # 绘制矩形框
        rect_patch = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect_patch)

        # 在矩形框中心显示块名称和ID
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        ax.text(center_x, center_y, f"{rect['name']}\nID: {rect['id']}",
                fontsize=8, color='blue', ha='center', va='center')

        # 保存坐标用于调整显示范围
        all_x.extend([x_min, x_max])
        all_y.extend([y_min, y_max])

    # 设置图像显示范围，避免范围过大
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualization of Rectangles')

    # 只保存一张图片，命名为 bbox_all.png
    plt.savefig('bbox_all.png')  # 保存为 bbox_all.png
    plt.close()  # 替换 plt.show()
    print("Saved image as bbox_all.png")

def visualize_rectangles_and_lines(rectangles, lines):
    fig, ax = plt.subplots(figsize=(10, 10))

    all_x = []
    all_y = []

    # 绘制矩形框
    for rect in rectangles:
        bbox = rect['bounding_box']
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # 绘制矩形框
        rect_patch = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect_patch)

        # 在矩形框中心显示块名称和ID
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        ax.text(center_x, center_y, f"{rect['name']}\nID: {rect['id']}",
                fontsize=8, color='blue', ha='center', va='center')

        # 保存坐标用于调整显示范围
        all_x.extend([x_min, x_max])
        all_y.extend([y_min, y_max])

    # 绘制线段
    for line in lines:
        # 提取线段坐标
        x, y = line['geometry'].xy
        color = line['color']
        
        # 绘制线段
        ax.plot(x, y, color=color, linewidth=2, label=f"Line ID: {line['id']}")
        
        # 保存坐标用于调整显示范围
        all_x.extend(x)
        all_y.extend(y)

    # 设置图像显示范围，避免范围过大
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualization of Rectangles and Lines')

    # 显示图例
    ax.legend()

    # 保存为图片，命名为 "line_and_bbox.png"
    plt.savefig('line_and_bbox.png')  # 保存为 line_and_bbox.png
    plt.close()  # 替换 plt.show()
    print("Saved image as line_and_bbox.png")
    

def main(file_path):
    # file_path = "./dxf/Drawing1.dxf"  # 替换为你的 DXF 文件路径
    target_dxf = file_path
    # 创建对象
    visualizer = DXFVisualizer(file_path)
    
    # 获取线段数据
    lines = visualizer.read_lines()
    # print("Lines:", lines)
    
    # 查找交点
    intersections = visualizer.find_intersections()
    # print("Intersections:", intersections)
    
    # 获取插入点数据
    insert_points_aa2, insert_points_other = visualizer.read_inserts()
    # print("Insert Points AA2:", insert_points_aa2)
    # print("Insert Points Other:", insert_points_other)
    
    # 查找每个 AA2 插入点最接近的直线
    insert_point_closest_line = visualizer.find_closest_line_for_inserts()
    # print("Insert Points AA2 Closest Line IDs with Rotation Angles:", insert_point_closest_line)
    
    # 构建线段信息字典
    line_info_dict = visualizer.build_line_info_dict()
    block_name = [i[-1] for i in insert_points_other] 
    block_name = list(set(block_name))
    total_intersected_lines = []
    total_bounding_boxes = []
    for i in block_name:
    
        source_dxf = f"extracted_blocks/{i}.dxf"
        
        matching_results = find_matching_entities(source_dxf, target_dxf)
        
        # 打印找到的匹配实体数量
        # print(f"Found {len(matching_results)} matching entities for block '{i}'.")
    
        # 用来存储矩形框（Bounding Boxes）
        bounding_boxes = []
    
        # 遍历每个匹配结果
        for idx, result in enumerate(matching_results):
    
            result['bounding_box'] = (
                (result['bounding_box'][0][0] - 1, result['bounding_box'][0][1] - 1),
                (result['bounding_box'][1][0] + 1, result['bounding_box'][1][1] + 1)
            )
            # print(f"ID: {idx}, Result: {result}")
            # 从字典读取bounding_box坐标
            bounding_box = result['bounding_box']
            
            # 创建矩形框
            rect = box(bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1])
            
            # 添加到bounding_boxes列表
            bounding_boxes.append({'id': idx, 'rect': rect})
            total_bounding_boxes.append({'id': idx, 'rect': rect,"name":i})
            # 记录相交的线段和交点
            intersected_lines = []
            
            # 遍历每一条线段
            for line_data in lines:
                line = line_data['geometry']
                # 查找交点
                intersection = line.intersection(rect)
                
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        # 交点是单个点，表示它与矩形框的某个边相交
                        position = 'on boundary'
                    elif isinstance(intersection, LineString):
                        # 如果交点是线段，说明这条线段穿过矩形框
                        position = 'crossing'
                    else:
                        position = 'outside'  # 其他情况认为是外部
            
                    # 添加信息
                    intersected_lines.append({
                        'line_id': line_data['id'],
                        'color': line_data['color'],
                        'intersection': intersection,
                        'position': position
                    })
                    total_intersected_lines.append({"block_id" : idx,
                        "block_name" : i ,
                        'line_id': line_data['id'],
                        'color': line_data['color'],
                        'intersection': intersection,
                        'position': position
                    })
    # 保存矩形框的信息
    rectangles_info = []
    
    # 遍历每个块的名称
    for i in block_name:
        source_dxf = f"extracted_blocks/{i}.dxf"
        matching_results = find_matching_entities(source_dxf, target_dxf)
        
        # print(f"Found {len(matching_results)} matching entities for block '{i}'.")
    
        # 遍历每个匹配结果并记录矩形框信息
        for idx, result in enumerate(matching_results):
            # print(f"ID: {idx}, Result: {result}")
            
            # 从字典读取bounding_box坐标
            bounding_box = result['bounding_box']
            
            if len(bounding_box) != 2 or len(bounding_box[0]) != 2 or len(bounding_box[1]) != 2:
                print(f"Skipping invalid bounding box: {bounding_box}")
                continue  # 跳过无效的bounding_box
            
            # 从两个点计算 x_min, y_min, x_max, y_max
            (x_min, y_min), (x_max, y_max) = bounding_box
            
            # 创建矩形框并保存信息
            rect_info = {
                'id': idx,
                'name': i,
                'bounding_box': (x_min, y_min, x_max, y_max)
            }
            rectangles_info.append(rect_info)
    
    # 调用可视化函数
    visualize_rectangles(rectangles_info) 
    Lines = lines
    visualize_rectangles_and_lines(rectangles_info, Lines)
if __name__ == "__main__":
    file_path = "./dxf/Drawing1.dxf"
    main(file_path)