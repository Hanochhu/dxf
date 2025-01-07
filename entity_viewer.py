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
        
        # 初始化基础变量
        self.current_zoom = 100
        self.base_width = 700
        self.base_height = 500
        
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
        
        # 添加两个模式按钮的框架
        self.mode_frame = ttk.Frame(self.button_frame)
        self.mode_frame.pack(side=tk.LEFT, padx=5)
        
        # 添加显示模式按钮
        self.bbox_button = ttk.Button(
            self.mode_frame,
            text="显示边界框",
            command=lambda: self.switch_mode('bbox')
        )
        self.bbox_button.pack(side=tk.LEFT, padx=5)
        
        self.flow_button = ttk.Button(
            self.mode_frame,
            text="显示流向图",
            command=lambda: self.switch_mode('flow')
        )
        self.flow_button.pack(side=tk.LEFT, padx=5)
        
        # 添加block_name输入框架
        self.block_frame = ttk.Frame(self.button_frame)
        self.block_frame.pack(side=tk.LEFT, padx=5)
        
        # 添加标签
        self.block_label = ttk.Label(self.block_frame, text="Block名称:")
        self.block_label.pack(side=tk.LEFT)
        
        # 添加输入框
        self.block_entry = ttk.Entry(self.block_frame, width=10)
        self.block_entry.pack(side=tk.LEFT, padx=5)
        self.block_entry.insert(0, 'mis007')  # 设置默认值
        
        # 添加确认按钮
        self.block_confirm = ttk.Button(
            self.block_frame,
            text="确认",
            command=self.reload_with_block
        )
        self.block_confirm.pack(side=tk.LEFT)
        
        # 创建固定大小的图片显示框架
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # 创建一个容器框架来管理logo和图片区域
        self.container = ttk.Frame(self.image_frame)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # 创建logo画布，固定在左侧
        self.logo_canvas = tk.Canvas(
            self.container,
            width=200,
            height=200,
            highlightthickness=0
        )
        self.logo_canvas.pack(side=tk.LEFT, fill=tk.Y)  # 使用pack布局，固定在左侧
        
        # 添加logo
        try:
            logo_image = Image.open("logo.jpg")
            logo_size = (200, 200)
            logo_image = logo_image.resize(logo_size, Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            
            # 在logo画布上创建logo
            self.logo_canvas.create_image(
                0, 0,
                image=self.logo_photo,
                anchor='nw'
            )
        except Exception as e:
            print(f"无法加载logo: {e}")
        
        # 创建主画布用于显示图片，放在logo右侧
        self.canvas = tk.Canvas(
            self.container,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # 填充剩余空间
        
        # 创建图片显示框架
        self.image_container = ttk.Frame(self.canvas)
        
        # 创建图片显示标签
        self.image_label = ttk.Label(self.image_container)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 在主画布上创建窗口来显示图片
        self.image_area = self.canvas.create_window(
            0, 0,
            anchor='nw',
            window=self.image_container,
            width=500,  # 初始宽度
            height=500  # 初始高度
        )
        
        # 初始化其他变量
        self.current_dxf_path = None
        self.current_image = None
        self.image_files = []
        self.current_image_index = 0
        self.is_playing = False
        self.after_id = None
        
        # 添加缩放相关的实例变量
        self.current_mode = None
        self.bbox_images = []
        self.flow_images = []
        
        # 添加输出文件夹路径
        self.output_dir = "output_images"
        # 确保输出文件夹存在
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            is_new_file = file_path != self.current_dxf_path
            self.current_dxf_path = file_path
            
            # 获取当前block_name
            block_name = self.block_entry.get().strip()
            if not block_name:
                messagebox.showwarning("警告", "请输入Block名称")
                return
            
            # 显示加载提示
            self.image_label.configure(text="正在生成图片，请稍候...")
            self.update()
            
            # 如果是新文件，才生成边界框图片
            if is_new_file:
                self.cleanup_previous_images()
                try:
                    draw_line_bbox_main(file_path)
                except Exception as e:
                    messagebox.showerror("错误", f"生成边界框图片失败: {str(e)}")
                    return
            else:
                # 只清除流向图
                for file in os.listdir():
                    if file.endswith('.png') and file.startswith('in_out_'):
                        try:
                            os.remove(file)
                        except Exception as e:
                            print(f"无法删除文件 {file}: {e}")
            
            try:
                # 生成流向图
                draw_in_out_main(file_path, block_name)
            except Exception as e:
                messagebox.showerror("错误", f"生成流向图失败: {str(e)}\n可能是因为找不到指定的Block: {block_name}")
            
            # 分别收集两种类型的图片
            self.bbox_images = []
            self.flow_images = []
            
            for file in os.listdir(self.output_dir):
                if file.endswith('.png'):
                    if file.startswith('bbox_') or file.startswith('line_and_bbox'):
                        self.bbox_images.append(os.path.join(self.output_dir, file))
                    elif file.startswith('in_out_'):
                        self.flow_images.append(os.path.join(self.output_dir, file))
            
            # 排序图片文件
            self.bbox_images.sort()
            self.flow_images.sort()
            
            if not self.bbox_images and not self.flow_images:
                messagebox.showerror("错误", "未能生成任何图片")
                return
            
            # 默认显示边界框模式
            self.switch_mode('bbox')
            
            # 显示加载成功消息，包含生成图片的数量信息
            success_msg = f"已成功加载文件: {os.path.basename(file_path)}\n"
            success_msg += f"生成边界框图片: {len(self.bbox_images)}张\n"
            success_msg += f"生成流向图: {len(self.flow_images)}张"
            messagebox.showinfo("成功", success_msg)
            
        except Exception as e:
            messagebox.showerror("错误", f"加载DXF文件失败: {str(e)}")
    
    def cleanup_previous_images(self):
        """清理之前生成的图片"""
        # 清空图片列表
        self.bbox_images = []
        self.flow_images = []
        self.image_files = []
        
        # 清理输出文件夹中的所有图片
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith('.png'):
                    try:
                        os.remove(os.path.join(self.output_dir, file))
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
            
            # 计算缩放后的显示尺寸，考虑logo占用的空间
            logo_width = 240  # logo宽度 + 边距
            display_width = int((self.base_width - logo_width) * (self.current_zoom / 100))
            display_height = int(self.base_height * (self.current_zoom / 100))
            
            # 更新图片框架大小
            self.image_frame.configure(width=self.base_width, height=display_height)
            
            # 更新图片显示区域大小
            self.canvas.itemconfig(
                self.image_area,
                width=display_width,
                height=display_height
            )
            
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
    
    def switch_mode(self, mode):
        """切换显示模式"""
        self.current_mode = mode
        if mode == 'bbox':
            if not self.bbox_images:
                messagebox.showwarning("警告", "没有可用的边界框图片")
                return
            self.image_files = self.bbox_images
            self.bbox_button.state(['pressed'])
            self.flow_button.state(['!pressed'])
        else:
            if not self.flow_images:
                messagebox.showwarning("警告", "没有可用的流向图，可能是因为找不到指定的Block")
                return
            self.image_files = self.flow_images
            self.bbox_button.state(['!pressed'])
            self.flow_button.state(['pressed'])
        
        # 重置图片索引并显示
        if self.image_files:
            self.current_image_index = 0
            self.display_current_image()
            self.update_navigation_buttons()
    
    def reload_with_block(self):
        """使用新的block_name重新加载流向图"""
        if not self.current_dxf_path:
            messagebox.showwarning("警告", "请先加载DXF文件")
            return
        
        try:
            # 获取当前block_name
            block_name = self.block_entry.get().strip()
            if not block_name:
                messagebox.showwarning("警告", "请输入Block名称")
                return
            
            # 显示加载提示
            self.image_label.configure(text="正在生成流向图，请稍候...")
            self.update()
            
            # 只清除流向图
            for file in os.listdir(self.output_dir):
                if file.endswith('.png') and file.startswith('in_out_'):
                    try:
                        os.remove(os.path.join(self.output_dir, file))
                    except Exception as e:
                        print(f"无法删除文件 {file}: {e}")
            
            try:
                # 只重新生成流向图
                draw_in_out_main(self.current_dxf_path, block_name)
            except Exception as e:
                messagebox.showerror("错误", f"生成流向图失败: {str(e)}\n可能是因为找不到指定的Block: {block_name}")
                return
            
            # 重新收集流向图
            self.flow_images = []
            for file in os.listdir(self.output_dir):
                if file.endswith('.png') and file.startswith('in_out_'):
                    self.flow_images.append(os.path.join(self.output_dir, file))
            
            self.flow_images.sort()
            
            # 如果当前正在显示流向图，则更新显示
            if self.current_mode == 'flow':
                self.switch_mode('flow')
            
            # 显示成功消息
            messagebox.showinfo("成功", f"已重新生成流向图: {len(self.flow_images)}张")
            
        except Exception as e:
            messagebox.showerror("错误", f"重新加载失败: {str(e)}")

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