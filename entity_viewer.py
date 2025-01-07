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
from draw_in_out import main as draw_in_out_main
from draw_line_bbox import main as draw_line_bbox_main

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
        
        # 创建按钮框架，放在顶部
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # 添加按钮
        self.browse_button = ttk.Button(
            self.button_frame, 
            text="浏览文件", 
            command=self.browse_file
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # 添加导航按钮
        self.prev_button = ttk.Button(
            self.button_frame,
            text="上一张",
            command=self.show_prev_image,
            state=tk.DISABLED
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            self.button_frame,
            text="下一张",
            command=self.show_next_image,
            state=tk.DISABLED
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # 添加自动播放按钮
        self.play_button = ttk.Button(
            self.button_frame,
            text="播放",
            command=self.toggle_autoplay
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # 添加缩放控制框架
        self.zoom_frame = ttk.Frame(self.button_frame)
        self.zoom_frame.pack(side=tk.RIGHT, padx=5)
        
        # 添加缩放标签
        self.zoom_label = ttk.Label(self.zoom_frame, text="缩放: ")
        self.zoom_label.pack(side=tk.LEFT)
        
        # 添加缩放滑块
        self.zoom_scale = ttk.Scale(
            self.zoom_frame,
            from_=50,
            to=150,
            orient=tk.HORIZONTAL,
            length=100,
            value=100,
            command=self.on_zoom_change
        )
        self.zoom_scale.pack(side=tk.LEFT)
        
        # 添加缩放百分比显示
        self.zoom_percent = ttk.Label(self.zoom_frame, text="100%")
        self.zoom_percent.pack(side=tk.LEFT, padx=(5, 0))
        
        # 创建固定大小的图片显示框架
        self.image_frame = ttk.Frame(self.main_frame, height=500, width=700)
        self.image_frame.pack(pady=10, padx=10)
        self.image_frame.pack_propagate(False)  # 防止框架大小被内容改变
        
        # 创建画布
        self.canvas = tk.Canvas(self.image_frame, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 创建图片显示标签
        self.image_label = ttk.Label(self.canvas)
        self.image_label.pack(expand=True)
        
        # 初始化变量
        self.current_dxf_path = None
        self.current_image = None
        self.image_files = []
        self.current_image_index = 0
        self.is_playing = False
        self.after_id = None
        
        # 添加缩放相关的实例变量
        self.current_zoom = 100
        self.base_width = 700
        self.base_height = 500
        
        # 绑定鼠标滚轮事件到所有相关控件
        for widget in (self.main_frame, self.image_frame, self.canvas, self.image_label):
            widget.bind('<MouseWheel>', self.on_mousewheel)       # Windows/Mac
            widget.bind('<Button-4>', self.on_mousewheel)         # Linux
            widget.bind('<Button-5>', self.on_mousewheel)         # Linux
            if USE_DND:
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind('<<Drop>>', self.handle_drop)
                except:
                    pass  # 某些控件可能不支持拖放
        
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
        """加载DXF文件并生成所有图片"""
        try:
            self.current_dxf_path = file_path
            
            # 显示加载提示
            self.image_label.configure(text="正在生成图片，请稍候...")
            self.update()
            
            # 清除之前的图片
            self.cleanup_previous_images()
            
            # 生成新的图片
            draw_line_bbox_main(file_path)  # 生成bbox相关的图片
            draw_in_out_main(file_path, 'mis007')  # 生成in_out相关的图片
            
            # 收集所有生成的图片
            self.image_files = []
            for file in os.listdir():
                if file.endswith('.png') and (file.startswith('bbox_') or 
                                            file.startswith('line_and_bbox') or 
                                            file.startswith('in_out_')):
                    self.image_files.append(file)
            
            self.image_files.sort()  # 排序图片文件
            
            if self.image_files:
                self.current_image_index = 0
                self.display_current_image()
                self.update_navigation_buttons()
                
            # 显示加载成功消息
            messagebox.showinfo("成功", f"已成功加载文件: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载DXF文件失败: {str(e)}")
    
    def cleanup_previous_images(self):
        """清理之前生成的图片"""
        for file in os.listdir():
            if file.endswith('.png') and (file.startswith('bbox_') or 
                                        file.startswith('line_and_bbox') or 
                                        file.startswith('in_out_')):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"无法删除文件 {file}: {e}")
    
    def display_current_image(self):
        """显示当前索引的图片"""
        if 0 <= self.current_image_index < len(self.image_files):
            self.display_image(self.image_files[self.current_image_index])
    
    def display_image(self, image_path):
        """在界面上显示图片"""
        try:
            # 打开图片
            image = Image.open(image_path)
            
            # 计算缩放后的显示尺寸
            display_width = int(self.base_width * (self.current_zoom / 100))
            display_height = int(self.base_height * (self.current_zoom / 100))
            
            # 更新图片框架大小
            self.image_frame.configure(width=display_width, height=display_height)
            
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
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("错误", f"显示图片失败: {str(e)}")
    
    def show_prev_image(self):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.update_navigation_buttons()
    
    def show_next_image(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        """更新导航按钮状态"""
        self.prev_button.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)
    
    def toggle_autoplay(self):
        """切换自动播放状态"""
        self.is_playing = not self.is_playing
        self.play_button.config(text="停止" if self.is_playing else "播放")
        
        if self.is_playing:
            self.autoplay()
        elif self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
    
    def autoplay(self):
        """自动播放图片"""
        if self.is_playing:
            if self.current_image_index < len(self.image_files) - 1:
                self.show_next_image()
                self.after_id = self.after(2000, self.autoplay)  # 2秒后显示下一张
            else:
                # 播放到最后一张时停止
                self.is_playing = False
                self.play_button.config(text="播放")
    
    def on_zoom_change(self, value):
        """处理缩放值变化"""
        try:
            zoom_value = float(value)
            self.current_zoom = zoom_value
            self.zoom_percent.configure(text=f"{int(zoom_value)}%")
            
            # 如果当前有显示的图片，重新显示以应用缩放
            if self.current_image_index < len(self.image_files):
                self.display_current_image()
        except ValueError:
            pass
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        current = self.zoom_scale.get()
        
        # 根据事件类型处理缩放
        if hasattr(event, 'delta'):  # Windows/Mac
            delta = event.delta
            if abs(delta) >= 120:  # Windows
                delta = delta // 120
            if delta > 0:
                new_value = min(current + 5, 150)  # 向上滚动，放大
            else:
                new_value = max(current - 5, 50)   # 向下滚动，缩小
        else:  # Linux
            if event.num == 4:
                new_value = min(current + 5, 150)  # 向上滚动，放大
            elif event.num == 5:
                new_value = max(current - 5, 50)   # 向下滚动，缩小
            else:
                return
        
        self.zoom_scale.set(new_value)
        self.on_zoom_change(new_value)
        return "break"  # 阻止事件继续传播

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