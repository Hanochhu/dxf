from dxf_visualizer import DXFVisualizer
from Entity import *
from shapely.geometry import LineString, box, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box

def get_rectangles_info(insert_points_other, target_dxf):
    """
    提取与各个块相关的矩形框信息。

    参数：
    insert_points_other (list): 包含块信息的列表
    target_dxf (str): 目标 DXF 文件路径

    返回：
    list: 包含所有矩形框信息的列表，每个元素是一个字典，包含 id、name 和 bounding_box
    """
    # 保存矩形框的信息
    rectangles_info = []

    # 获取所有唯一的块名称
    block_names = [i[-1] for i in insert_points_other]
    block_names = list(set(block_names))

    # 遍历每个块的名称
    for block_name in block_names:
        source_dxf = f"extracted_blocks/{block_name}.dxf"
        matching_results = find_matching_entities(source_dxf, target_dxf)

        # 遍历每个匹配结果并记录矩形框信息
        for idx, result in enumerate(matching_results):
            bounding_box = result['bounding_box']

            # 跳过无效的bounding_box
            if len(bounding_box) != 2 or len(bounding_box[0]) != 2 or len(bounding_box[1]) != 2:
                continue

            # 从两个点计算 x_min, y_min, x_max, y_max
            (x_min, y_min), (x_max, y_max) = bounding_box

            # 创建矩形框并保存信息
            rect_info = {
                'id': idx,
                'name': block_name,
                'bounding_box': (x_min, y_min, x_max, y_max)
            }
            rectangles_info.append(rect_info)

    return rectangles_info


def process_intersections(insert_points_other, lines, file_path):
    """
    处理与各个块相关的矩形框和相交的线段。

    参数:
    insert_points_other (list): 包含块信息的列表
    lines (list): 存储所有线段的数据列表
    file_path (str): 目标 DXF 文件路径

    返回:
    tuple: 包含 total_intersected_lines 和 total_bounding_boxes 的元组
    """
    block_names = [i[-1] for i in insert_points_other]
    block_names = list(set(block_names))

    total_intersected_lines = []
    total_bounding_boxes = []
    intersecting_boxes = {}

    for block_name in block_names:
        source_dxf = f"extracted_blocks/{block_name}.dxf"
        matching_results = find_matching_entities(source_dxf, file_path)

        bounding_boxes = []

        for idx, result in enumerate(matching_results):
            result['bounding_box'] = (
                (result['bounding_box'][0][0] - 1, result['bounding_box'][0][1] - 1),
                (result['bounding_box'][1][0] + 1, result['bounding_box'][1][1] + 1)
            )
            bounding_box = result['bounding_box']
            rect = box(bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1])

            bounding_boxes.append({'id': idx, 'rect': rect})
            total_bounding_boxes.append({'id': idx, 'rect': rect, "name": block_name})

            # 记录相交的线段和交点
            intersected_lines = []
            for line_data in lines:
                line = line_data['geometry']
                intersection = line.intersection(rect)

                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        position = 'on boundary'
                    elif isinstance(intersection, LineString):
                        position = 'crossing'
                    else:
                        position = 'outside'  # 其他情况认为是外部

                    intersected_lines.append({
                        'line_id': line_data['id'],
                        'color': line_data['color'],
                        'intersection': intersection,
                        'position': position
                    })
                    total_intersected_lines.append({
                        "block_id": idx,
                        "block_name": block_name,
                        'line_id': line_data['id'],
                        'color': line_data['color'],
                        'intersection': intersection,
                        'position': position
                    })

        # 检测矩形框之间的相交
        for i in range(len(total_bounding_boxes)):
            for j in range(i + 1, len(total_bounding_boxes)):
                rect1 = total_bounding_boxes[i]['rect']
                rect2 = total_bounding_boxes[j]['rect']

                if rect1.intersects(rect2):
                    if (total_bounding_boxes[i]['name'], total_bounding_boxes[i]['id']) not in intersecting_boxes:
                        intersecting_boxes[(total_bounding_boxes[i]['name'], total_bounding_boxes[i]['id'])] = []
                    if (total_bounding_boxes[j]['name'], total_bounding_boxes[j]['id']) not in intersecting_boxes:
                        intersecting_boxes[(total_bounding_boxes[j]['name'], total_bounding_boxes[j]['id'])] = []

                    intersecting_boxes[(total_bounding_boxes[i]['name'], total_bounding_boxes[i]['id'])].append({
                        'name': total_bounding_boxes[j]['name'],
                        'id': total_bounding_boxes[j]['id']
                    })

                    intersecting_boxes[(total_bounding_boxes[j]['name'], total_bounding_boxes[j]['id'])].append({
                        'name': total_bounding_boxes[i]['name'],
                        'id': total_bounding_boxes[i]['id']
                    })

    return total_intersected_lines, total_bounding_boxes

def find_connected_blocks(block_name, block_id):
    """
    查找与指定块相关的所有块及其连接的 line_id。

    参数：
    block_name (str): 初始块的名称
    block_id (int): 初始块的 ID

    返回：
    dict: 包含找到的所有连接块及其对应 line_ids 的字典
    """
    # 查找与 block_name 和 block_id 相关的所有 line_id
    initial_line_ids = find_line_ids_by_block(block_name, block_id)

    if not initial_line_ids:
        print(f"No lines found for block {block_name}, block_id {block_id}.")
        return {}  # 如果没有找到任何相关的 line_id，返回空字典
    else:
        # 通过栈模拟递归查找与这些 line_id 相关联的块
        found_blocks = {}  # 存储每个找到的块和路径
        visited_lines = set()  # 使用 set 来记录已访问的 line_id

        for line_id in initial_line_ids:
            # 对于每个 line_id，查找是否与其他块连接
            connected_block_paths, visited_lines = find_connected_block_by_line(
                line_id, visited_lines, start_block_name=block_name, start_block_id=block_id, block_paths=found_blocks
            )

            if connected_block_paths:
                found_blocks = connected_block_paths  # 更新找到的块和路径

        return found_blocks  # 返回找到的所有连接块及其 line_id


def find_connected_block_by_line(line_id, visited_lines=None, start_block_name=None, start_block_id=None,
                                 block_paths=None):
    if visited_lines is None:
        visited_lines = set()  # 使用 set 来避免重复访问
    if block_paths is None:
        block_paths = {}  # 用于存储块和路径的映射

    # 使用栈来模拟递归
    stack = [(line_id, [], visited_lines)]  # stack 存储 (当前line_id, 当前路径, 已访问的线段)

    while stack:
        current_line_id, path, visited_lines = stack.pop()
        path = path + [current_line_id]  # 更新当前路径

        # 如果该 line_id 已经访问过，跳过
        if current_line_id in visited_lines:
            continue

        # 标记当前 line_id 为已访问
        visited_lines.add(current_line_id)

        # 查找与当前 line_id 相关联的块
        connected_block = find_block_by_line_id(current_line_id)

        # 如果找到了一个连接的块，并且它不是起始块
        if connected_block and (connected_block[0] != start_block_name or connected_block[1] != start_block_id):
            if connected_block not in block_paths:
                block_paths[connected_block] = []  # 初始化该块的路径列表
            block_paths[connected_block].append(path)  # 记录路径
            continue  # 找到目标块，跳过继续递归

        # 查找与当前 line_id 相交的所有线段
        intersecting_lines = intersections.get(current_line_id, [])

        # 将所有与当前线段相交的线段压入栈中
        for intersection in intersecting_lines:
            intersected_line_id = intersection['line_id']
            if intersected_line_id not in visited_lines:
                stack.append((intersected_line_id, path, visited_lines.copy()))  # 使用路径副本，避免影响其他路径

    # 返回最终的块路径和已访问的线段
    return block_paths, visited_lines


# 查找给定 line_id 所属的块
def find_block_by_line_id(line_id):
    for item in total_intersected_lines:
        if item['line_id'] == line_id:
            return (item['block_name'], item['block_id'])
    return None


# 查找给定 block_name 和 block_id 的所有相关线段
def find_line_ids_by_block(block_name, block_id):
    line_ids = []
    for item in total_intersected_lines:
        if item['block_name'] == block_name and item['block_id'] == block_id:
            line_ids.append(item['line_id'])
    return line_ids

# 可视化矩形框和线段
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
        # ax.text(center_x, center_y, f"{rect['name']}\nID: {rect['id']}",
        #         fontsize=8, color='blue', ha='center', va='center')

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
        # ax.plot(x, y, color='black', linewidth=2)
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
    plt.show()

# 可视化所有矩形框
def visualize_rectangles(rectangles):
    fig, ax = plt.subplots(figsize=(10, 10))

    all_x = []
    all_y = []

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
    plt.show()


if __name__ =="__main__":
    # 使用示例
    file_path = r"dxf/Drawing1.dxf"  # 替换为你的 DXF 文件路径

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
    # 打印结果
    # print(line_info_dict)

    ## total_intersected_lines 记录了block与line的相交关系，  total_bounding_boxes 记录的矩形框之间的相交关系
    total_intersected_lines, total_bounding_boxes = process_intersections(insert_points_other, lines, file_path)


    # rectangles_info  记录了每个矩形框的具体信息
    rectangles_info = get_rectangles_info(insert_points_other, file_path)

    block_name = 'mis007'
    block_id = 1
    connected_blocks = find_connected_blocks(block_name, block_id)
    print(connected_blocks)

    
    ## 以下变量你可能用得上
    # insert_point_closest_line记录了箭头与线段的关系，箭头 0°是左  90°是下
    # print(insert_point_closest_line)
    # intersections 记录了line与line的相交关系
    # total_intersected_lines 记录了block与line 的相交关系

