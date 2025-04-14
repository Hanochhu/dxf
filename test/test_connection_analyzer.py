import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_structures import Point, BoundingBox, BlockReference, LineEntity
from src.connection.connection_analyzer import ConnectionAnalyzer

from src.core.data_structures import Block, EntityType

def make_blockref(id, min_pt, max_pt):
    bbox = BoundingBox(Point(*min_pt), Point(*max_pt))
    # 构造最小Block对象
    block = Block(
        id=f"block_{id}",
        name="block",
        entities=[],
        bounding_box=bbox
    )
    return BlockReference(
        id=id,
        name="test",
        position=Point(0,0,0),
        rotation=0,
        scale=(1,1,1),
        block=block,
        attributes=[]
    )

def make_lineentity(id, start, end, horizontal=True):
    # 只生成水平或竖直线段
    if horizontal:
        # 水平线段，y相等
        pt1 = (start[0], start[1], 0)
        pt2 = (end[0], start[1], 0)
    else:
        # 竖直线段，x相等
        pt1 = (start[0], start[1], 0)
        pt2 = (start[0], end[1], 0)
    return LineEntity(
        id=id,
        entity_type=None,
        layer="0",
        start_point=Point(*pt1),
        end_point=Point(*pt2)
    )

def test_blockreference_connection():
    # 两个BlockReference，边界框分别为(0,0,0)-(2,2,0)和(3,0,0)-(5,2,0)
    ref1 = make_blockref("ref1", (0,0,0), (2,2,0))
    ref2 = make_blockref("ref2", (3,0,0), (5,2,0))
    # LineEntity，边界框为(1,0,0)-(4,2,0)，与两个BlockReference都重叠
    # 水平线段，y=2，x从1到4，正好接触两个BlockReference的上边界
    line = make_lineentity("line1", (1,2,0), (4,2,0), horizontal=True)
    analyzer = ConnectionAnalyzer()
    connections = analyzer.analyze_blockreference_connections_via_lines([ref1, ref2], [line])
    print("连接数（应为1）:", len(connections))
    for conn in connections:
        print("连接:", conn.source_ref.id, "<->", conn.target_ref.id)
        for seg in conn.path_segments:
            print("  path_segment id:", seg.id, "起点:", seg.start_point.to_tuple(), "终点:", seg.end_point.to_tuple())

    # 换一条不重叠的线
    line2 = make_lineentity("line2", (10,10,0), (12,12,0))
    connections2 = analyzer.analyze_blockreference_connections_via_lines([ref1, ref2], [line2])
    print("连接数（应为0）:", len(connections2))

if __name__ == "__main__":
    test_blockreference_connection()

    # 可视化部分
    import matplotlib.pyplot as plt
    from src.visualization.entity_renderer import EntityRenderer
    from matplotlib.patches import Rectangle

    def visualize_blockrefs_and_connections(refs, lines, connections):
        fig, ax = plt.subplots(figsize=(8, 4))
        renderer = EntityRenderer()
        # 绘制BlockReference的bounding_box
        for ref in refs:
            bbox = ref.bounding_box
            if bbox:
                rect = Rectangle(
                    (bbox.min_point.x, bbox.min_point.y),
                    bbox.width,
                    bbox.height,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                    linestyle="--",
                    label=f"BlockRef {ref.id}"
                )
                ax.add_patch(rect)
                ax.text(
                    bbox.center.x, bbox.center.y, ref.id,
                    color="blue", fontsize=12, ha="center", va="center"
                )
        # 绘制LineEntity
        for line in lines:
            renderer.render_line(line, ax, color="orange", linewidth=2)
        # 绘制连接关系
        for conn in connections:
            # 计算source_ref的bounding_box与line的最近点
            ref1_bbox = conn.source_ref.bounding_box
            ref2_bbox = conn.target_ref.bounding_box
            line = conn.path_segments[0]
            # 取ref1/ref2的上边界中点（假设水平线接触上边界）
            ref1_top = ((ref1_bbox.min_point.x + ref1_bbox.max_point.x)/2, ref1_bbox.max_point.y)
            ref2_top = ((ref2_bbox.min_point.x + ref2_bbox.max_point.x)/2, ref2_bbox.max_point.y)
            # 取line的中点
            line_mid = ((line.start_point.x + line.end_point.x)/2, (line.start_point.y + line.end_point.y)/2)
            # 画ref1上边界中点到line中点的虚线
            ax.plot([ref1_top[0], line_mid[0]], [ref1_top[1], line_mid[1]], color="red", linewidth=1.5, alpha=0.7, zorder=10, linestyle="--")
            ax.plot([ref2_top[0], line_mid[0]], [ref2_top[1], line_mid[1]], color="red", linewidth=1.5, alpha=0.7, zorder=10, linestyle="--")
            # 标注接触点
            ax.scatter([ref1_top[0], ref2_top[0], line_mid[0]], [ref1_top[1], ref2_top[1], line_mid[1]], color="red", s=40, zorder=12)
            ax.text(line_mid[0], line_mid[1], "接触", color="red", fontsize=10, ha="center", va="center")
        ax.set_aspect("equal")
        ax.set_title("BlockReference连接关系可视化")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    # 复用上面的测试数据
    ref1 = make_blockref("ref1", (0,0,0), (2,2,0))
    ref2 = make_blockref("ref2", (3,0,0), (5,2,0))
    # 水平线段，y=2，x从1到4，正好接触两个BlockReference的上边界
    line = make_lineentity("line1", (1,2,0), (4,2,0), horizontal=True)
    analyzer = ConnectionAnalyzer()
    connections = analyzer.analyze_blockreference_connections_via_lines([ref1, ref2], [line])
    visualize_blockrefs_and_connections([ref1, ref2], [line], connections)