"""
文件解析模块：抽象接口
提供CAD文件解析的基础接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import uuid

from core.data_structures import Entity, Block


class CADFileParser(ABC):
    """CAD文件解析器基类"""
    
    @abstractmethod
    def parse_file(self, file_path: str) -> Tuple[List[Entity], List[Block], Dict[str, Any]]:
        """
        解析CAD文件，返回实体、块和附加信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            元组：(实体列表, 块列表, 附加信息字典)
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """
        检查解析器是否支持指定格式
        
        Args:
            file_extension: 文件扩展名（带或不带.）
            
        Returns:
            布尔值：是否支持该格式
        """
        pass
    
    def generate_unique_id(self, prefix: str = "") -> str:
        """
        生成唯一ID
        
        Args:
            prefix: ID前缀
            
        Returns:
            唯一ID字符串
        """
        return f"{prefix}{uuid.uuid4().hex}"


class CADBlockExporter(ABC):
    """CAD块导出器基类"""
    
    @abstractmethod
    def export_block(self, block: Block, file_path: str) -> bool:
        """
        将块导出为CAD文件
        
        Args:
            block: 块对象
            file_path: 输出文件路径
            
        Returns:
            布尔值：操作是否成功
        """
        pass


class CADFileWriter(ABC):
    """CAD文件写入器基类"""
    
    @abstractmethod
    def create_new_file(self, file_path: str) -> bool:
        """
        创建新的CAD文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            布尔值：操作是否成功
        """
        pass
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> bool:
        """
        添加实体到文件
        
        Args:
            entity: 实体对象
            
        Returns:
            布尔值：操作是否成功
        """
        pass
    
    @abstractmethod
    def add_block(self, block: Block) -> bool:
        """
        添加块到文件
        
        Args:
            block: 块对象
            
        Returns:
            布尔值：操作是否成功
        """
        pass
    
    @abstractmethod
    def save_file(self) -> bool:
        """
        保存文件
        
        Returns:
            布尔值：操作是否成功
        """
        pass


class UnknownFormatError(Exception):
    """未知文件格式错误"""
    pass


class ParserRegistry:
    """解析器注册中心"""
    
    def __init__(self):
        self.parsers = {}
    
    def register_parser(self, parser: CADFileParser):
        """
        注册解析器
        
        Args:
            parser: CAD文件解析器实例
        """
        for format_ext in self._get_supported_formats(parser):
            self.parsers[format_ext.lower()] = parser
    
    def get_parser(self, file_path: str) -> CADFileParser:
        """
        获取适用于指定文件的解析器
        
        Args:
            file_path: 文件路径
            
        Returns:
            CADFileParser: 适用的解析器
            
        Raises:
            UnknownFormatError: 如果没有找到适用的解析器
        """
        ext = self._get_file_extension(file_path)
        
        if ext in self.parsers:
            return self.parsers[ext]
        
        # 尝试匹配支持该格式的解析器
        for parser in set(self.parsers.values()):
            if parser.supports_format(ext):
                return parser
        
        raise UnknownFormatError(f"No parser found for file format: {ext}")
    
    def _get_file_extension(self, file_path: str) -> str:
        """
        获取文件扩展名（小写，不带点）
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件扩展名
        """
        import os
        ext = os.path.splitext(file_path)[1].lower()
        return ext[1:] if ext.startswith('.') else ext
    
    def _get_supported_formats(self, parser: CADFileParser) -> List[str]:
        """
        获取解析器支持的所有格式
        
        Args:
            parser: CAD文件解析器
            
        Returns:
            List[str]: 支持的格式列表
        """
        formats = []
        # 尝试一些常见格式
        common_formats = ['dxf', 'dwg', 'step', 'stp', 'iges', 'igs', 'stl']
        
        for fmt in common_formats:
            if parser.supports_format(fmt):
                formats.append(fmt)
        
        return formats