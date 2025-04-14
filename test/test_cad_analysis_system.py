import os
import sys
print("sys.path:", sys.path)

# 保证可以导入 src 目录下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.system.cad_analysis_system import CADAnalysisSystem
except Exception as e:
    print("导入 CADAnalysisSystem 失败:", e)
    raise

def test_cad_analysis_system_connection():
    dxf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf'))
    assert os.path.exists(dxf_path), f"测试用 dxf 文件不存在: {dxf_path}"

    system = CADAnalysisSystem()
    success = system.analyze_file(dxf_path)
    assert success, "CADAnalysisSystem 分析 dxf 文件失败"

    # 输出连接分析结果
    print("连接数量:", len(system.connections))
    for conn in system.connections[:10]:  # 只打印前10个连接
        print(f"连接: {conn}")

    # 可以根据实际需求添加更详细的断言
    assert len(system.connections) > 0, "未检测到任何连接"

if __name__ == "__main__":
    test_cad_analysis_system_connection()