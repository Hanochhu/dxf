import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
from Entity import EntityNetwork
import tempfile
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    USE_DND = True
except ImportError:
    USE_DND = False
    print("tkinterdnd2 not available, drag and drop will be disabled")

class EntityViewer(TkinterDnD.Tk if USE_DND else tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("DXF Entity Viewer")
        self.geometry("800x600")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建拖放区域
        self.drop_frame = ttk.LabelFrame(
            self.main_frame, 
            text="点击浏览按钮选择DXF文件或直接拖放文件到此处"
        )
        self.drop_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图片显示标签
        self.image_label = ttk.Label(self.drop_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 添加按钮框架
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # 添加按钮
        self.browse_button = ttk.Button(
            self.button_frame, 
            text="浏览文件", 
            command=self.browse_file
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # 初始化变量
        self.current_dxf_path = None
        self.current_image = None
        
        # 绑定拖放事件
        if USE_DND:
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        
    def handle_drop(self, event):
        """处理文件拖放"""
        try:
            # 获取拖放的文件路径
            file_path = event.data.strip('{}')
            
            if file_path.lower().endswith('.dxf'):
                self.load_dxf(file_path)
            else:
                messagebox.showerror("错误", "请拖放DXF文件")
        except Exception as e:
            messagebox.showerror("错误", f"处理拖放文件失败: {str(e)}")
    
    def browse_file(self):
        """打开文件浏览对话框"""
        file_path = filedialog.askopenfilename(
            title="选择DXF文件",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )
        if file_path:
            self.load_dxf(file_path)
    
    def load_dxf(self, file_path):
        """加载DXF文件并显示"""
        try:
            self.current_dxf_path = file_path
            
            # 创建EntityNetwork实例
            network = EntityNetwork(file_path)
            
            # 生成预览图片
            img_path = self.generate_preview(file_path)
            
            # 显示图片
            if img_path:
                self.display_image(img_path)
                
            # 显示加载成功消息
            messagebox.showinfo("成功", f"已成功加载文件: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载DXF文件失败: {str(e)}")
    
    def generate_preview(self, dxf_path):
        """生成DXF预览图片"""
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "preview.png")
            
            # 设置绘图区域
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp)
            
            # 调整视图
            ax.set_axis_off()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"生成预览图片失败: {str(e)}")
            return None
    
    def display_image(self, image_path):
        """在界面上显示图片"""
        try:
            # 打开图片
            image = Image.open(image_path)
            
            # 获取显示区域大小
            self.update_idletasks()  # 确保尺寸更新
            display_width = max(self.drop_frame.winfo_width() - 20, 100)
            display_height = max(self.drop_frame.winfo_height() - 20, 100)
            
            # 保持宽高比缩放
            image_ratio = image.width / image.height
            display_ratio = display_width / display_height
            
            if image_ratio > display_ratio:
                new_width = display_width
                new_height = int(display_width / image_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * image_ratio)
            
            # 调整图片大小
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # 更新显示
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # 保持引用
            
        except Exception as e:
            messagebox.showerror("错误", f"显示图片失败: {str(e)}")

def main():
    app = EntityViewer()
    
    # 创建高亮样式
    style = ttk.Style()
    style.configure('Highlight.TLabelframe', borderwidth=2, relief="solid")
    
    # 设置窗口最小尺寸
    app.minsize(400, 300)
    
    # 启动应用
    app.mainloop()

if __name__ == "__main__":
    main() 