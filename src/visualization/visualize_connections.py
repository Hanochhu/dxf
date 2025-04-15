import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from src.system.cad_analysis_system import CADAnalysisSystem
from src.visualization.entity_renderer import EntityRenderer

def _sort_segments_to_polyline(segments, tolerance=1e-3):
    # 将一组线段按端点顺序串联成折线点序列
    if not segments:
        return []
    # 构建端点到线段的映射
    from collections import defaultdict
    point_map = defaultdict(list)
    for seg in segments:
        for pt in [seg.start_point, seg.end_point]:
            key = (round(pt.x / tolerance), round(pt.y / tolerance), round(pt.z / tolerance))
            point_map[key].append(seg)
    # 找到端点只出现一次的点，作为首尾
    endpoint_count = {}
    for seg in segments:
        for pt in [seg.start_point, seg.end_point]:
            key = (round(pt.x / tolerance), round(pt.y / tolerance), round(pt.z / tolerance))
            endpoint_count[key] = endpoint_count.get(key, 0) + 1
    endpoints = [k for k, v in endpoint_count.items() if v == 1]
    # 从任一端点出发，串联所有线段
    used = set()
    polyline = []
    if endpoints:
        # 有首尾
        start_key = endpoints[0]
    else:
        # 闭合环
        start_key = list(point_map.keys())[0]
    current_key = start_key
    last_pt = None
    while True:
        segs = [s for s in point_map[current_key] if id(s) not in used]
        if not segs:
            break
        seg = segs[0]
        used.add(id(seg))
        if not polyline:
            polyline.append(seg.start_point)
            polyline.append(seg.end_point)
        else:
            # 判断如何衔接
            if abs(seg.start_point.x - polyline[-1].x) < tolerance and abs(seg.start_point.y - polyline[-1].y) < tolerance:
                polyline.append(seg.end_point)
            else:
                polyline.append(seg.start_point)
        # 下一个端点
        if abs(seg.end_point.x - polyline[-1].x) < tolerance and abs(seg.end_point.y - polyline[-1].y) < tolerance:
            current_key = (round(seg.end_point.x / tolerance), round(seg.end_point.y / tolerance), round(seg.end_point.z / tolerance))
        else:
            current_key = (round(seg.start_point.x / tolerance), round(seg.start_point.y / tolerance), round(seg.start_point.z / tolerance))
    return polyline

def visualize_connections(dxf_path, output_path="connections.png"):
    # 初始化系统并分析文件
    system = CADAnalysisSystem()
    success = system.analyze_file(dxf_path)
    if not success:
        print("分析文件失败")
        return

    # 强制调大连接容差，便于端点聚合
    system.set_connection_parameters(block_connection_tolerance=20.0)
    renderer = EntityRenderer()

    # 调试：统计聚合连接的分布
    seg_count_hist = {}
    for conn in getattr(system, "connections", []):
        n = len(getattr(conn, "path_segments", []))
        seg_count_hist[n] = seg_count_hist.get(n, 0) + 1
    print("连接聚合段数分布：", seg_count_hist)

    fig, ax = plt.subplots(figsize=(48, 36))
    ax.set_aspect("equal")
    ax.set_title("CAD 块连接可视化", fontsize=16)

    # 渲染所有块引用
    for ref in getattr(system, "block_references", []):
        # 在BlockReference位置标注其id
        ax.text(ref.position.x, ref.position.y + 2, str(ref.id), fontsize=6, color='blue', ha='center', va='bottom', zorder=20)
        renderer.render_insert(ref, ax, color="#00ff00", linewidth=1.0, alpha=0.7, zorder=10)

    # 渲染所有连接（聚合折线高亮）
    for conn in getattr(system, "connections", []):
        segs = getattr(conn, "path_segments", [])
        if len(segs) > 1:
            # 聚合为折线
            polyline = _sort_segments_to_polyline(segs)
            if len(polyline) >= 2:
                xs = [pt.x for pt in polyline]
                ys = [pt.y for pt in polyline]
                ax.plot(xs, ys, color="#FFA500", linewidth=3.0, alpha=0.9, zorder=6, label="聚合连接" if "聚合连接" not in ax.get_legend_handles_labels()[1] else "")
        for seg in segs:
            renderer.render_line(seg, ax, color="#ff0000", linewidth=2.0, alpha=0.8, zorder=5)

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"连接可视化结果已保存为: {output_path}")

if __name__ == "__main__":
    # 修改为你的 dxf 文件路径
    dxf_path = "图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf"
    visualize_connections(dxf_path)
    # 统计连接数量
    system = CADAnalysisSystem()
    success = system.analyze_file(dxf_path)
    if not success:
        print("分析文件失败")
        exit()
    # 输出所有参与连接的BlockReference的id列表
    connected_blocks = system.connection_analyzer.get_connected_blockreferences(system.connections)
    # 输出所有具有链接关系的BlockReference对及其相连线段
    connection_info = []
    for conn in system.connections:
        if getattr(conn, "connection_type", None) != "unconnected":
            src_id = getattr(conn.source_ref, "id", None)
            tgt_id = getattr(conn.target_ref, "id", None)
            seg_ids = [getattr(seg, "id", None) for seg in getattr(conn, "path_segments", [])]
            connection_info.append((src_id, tgt_id, seg_ids))
    print("具有链接关系的BlockReference对及其相连线段：")
    for src_id, tgt_id, seg_ids in connection_info:
        print(f"{src_id} <-> {tgt_id} : {seg_ids}")

    block_ids = [b.id for b in connected_blocks]
    print("具有链接关系的BlockReference id 列表：", block_ids)

    print(f"连接数量: {len(system.connections)}")