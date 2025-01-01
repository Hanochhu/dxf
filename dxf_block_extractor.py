import ezdxf
from pathlib import Path
import re
from typing import List, Dict, Optional
import logging

class DxfBlockExtractor:
    """DXF文件块提取器，用于从DXF文件中提取块并保存为单独的文件"""

    def __init__(self, source_file: str, output_dir: str):
        """初始化DXF块提取器
        
        Args:
            source_file: 源DXF文件路径
            output_dir: 输出目录路径
        """
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.doc = None
        self.extracted_blocks: Dict[str, Path] = {}
        
        # 配置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def _clean_filename(self, name: str) -> str:
        """清理文件名，移除非法字符
        
        Args:
            name: 原始文件名
            
        Returns:
            清理后的文件名
        """
        # 替换非法字符为下划线
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
        # 确保文件名不以点或空格开始/结束
        cleaned = cleaned.strip('. ')
        # 如果文件名为空，使用默认名称
        return cleaned if cleaned else 'block'

    def _extract_nested_blocks(self, block_name: str, target_doc):
        """递归提取嵌套块定义
        
        Args:
            block_name: 块名称
            target_doc: 目标DXF文档
        """
        if block_name not in self.doc.blocks:
            self.logger.warning(f"块定义 '{block_name}' 不存在")
            return
        
        # 如果块已提取，跳过
        if block_name in target_doc.blocks:
            return
        
        # 提取当前块
        source_block = self.doc.blocks.get(block_name)
        target_block = target_doc.blocks.new(block_name)
        
        # 复制块中的实体
        for entity in source_block:
            target_block.add_entity(entity.copy())
            
            # 如果实体是 INSERT，递归提取嵌套块
            if entity.dxftype() == 'INSERT':
                nested_block_name = entity.dxf.name
                self._extract_nested_blocks(nested_block_name, target_doc)

    def _create_block_file(self, block_name: str, entities: List) -> Optional[Path]:
        """为单个块创建DXF文件
        
        Args:
            block_name: 块名称
            entities: 块中的实体列表
            
        Returns:
            创建的文件路径，如果失败则返回None
        """
        try:
            # 创建新的DXF文件
            new_doc = ezdxf.new('R2010')
            msp = new_doc.modelspace()
            
            # 提取当前块及其嵌套块
            self._extract_nested_blocks(block_name, new_doc)
            
            # 在模型空间中插入块引用
            msp.add_blockref(block_name, (0, 0))
            
            # 生成文件名并保存
            clean_name = self._clean_filename(block_name)
            output_file = self.output_dir / f"{clean_name}.dxf"
            
            # 如果文件已存在，添加数字后缀
            counter = 1
            while output_file.exists():
                output_file = self.output_dir / f"{clean_name}_{counter}.dxf"
                counter += 1
            
            new_doc.saveas(output_file)
            self.logger.info(f"已保存块 '{block_name}' 到文件: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"创建块文件 '{block_name}' 时出错: {str(e)}")
            return None

    def extract_blocks(self, skip_special_blocks: bool = True) -> Dict[str, Path]:
        """提取所有块并保存为单独的文件
        
        Args:
            skip_special_blocks: 是否跳过特殊块（以*开头的块）
            
        Returns:
            Dict[str, Path]: 块名称到文件路径的映射
        """
        try:
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取源文件
            self.doc = ezdxf.readfile(self.source_file)
            self.logger.info(f"成功读取源文件: {self.source_file}")
            
            # 遍历所有块
            for block in self.doc.blocks:
                # 检查是否跳过特殊块
                if skip_special_blocks and block.name.startswith('*'):
                    self.logger.debug(f"跳过特殊块: {block.name}")
                    continue
                
                # 创建块文件
                output_file = self._create_block_file(block.name, block)
                if output_file:
                    self.extracted_blocks[block.name] = output_file
            
            self.logger.info(f"共提取了 {len(self.extracted_blocks)} 个块")
            return self.extracted_blocks
            
        except Exception as e:
            self.logger.error(f"提取块时出错: {str(e)}")
            return {}

    def get_block_info(self, block_name: str) -> Dict:
        """获取指定块的信息
        
        Args:
            block_name: 块名称
            
        Returns:
            包含块信息的字典
        """
        if not self.doc:
            return {}
        
        try:
            block = self.doc.blocks[block_name]
            
            # 统计实体类型
            entity_types = {}
            for entity in block:
                entity_type = entity.dxftype()
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            return {
                'name': block_name,
                'entity_count': len(block),
                'entity_types': entity_types,
                'output_file': self.extracted_blocks.get(block_name)
            }
            
        except Exception as e:
            self.logger.error(f"获取块 '{block_name}' 信息时出错: {str(e)}")
            return {}

    def get_all_blocks_info(self) -> List[Dict]:
        """获取所有已提取块的信息
        
        Returns:
            包含所有块信息的列表
        """
        return [self.get_block_info(name) for name in self.extracted_blocks.keys()] 
    
if __name__ == "__main__":
    # 创建提取器实例
    extractor = DxfBlockExtractor(
        source_file="图例和流程图_仪表管件设备均为模块/2308PM-09-T3-2900.dxf",
        output_dir="extracted_blocks"
    )

    # 提取所有块
    extracted_blocks = extractor.extract_blocks()

    # 获取所有块的详细信息
    blocks_info = extractor.get_all_blocks_info()

    # 打印提取结果
    for block_info in blocks_info:
        print(f"\n块名称: {block_info['name']}")
        print(f"实体数量: {block_info['entity_count']}")
        print(f"实体类型: {block_info['entity_types']}")
        print(f"输出文件: {block_info['output_file']}")