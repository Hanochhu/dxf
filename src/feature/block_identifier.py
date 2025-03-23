"""
块特征提取和识别模块
提供块特征的提取和基于特征的块识别功能
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import os

from core.data_structures import (
    EntityType, Entity, Point, BoundingBox, Block, BlockFeature
)


class BlockFeatureExtractor:
    """块特征提取器"""
    
    def extract_features(self, block: Block) -> np.ndarray:
        """
        提取块的特征向量
        
        Args:
            block: 块对象
            
        Returns:
            numpy.ndarray: 特征向量
        """
        return np.array(block.get_feature_vector())
    
    def extract_feature_dict(self, block: Block) -> Dict:
        """
        提取块的特征字典
        
        Args:
            block: 块对象
            
        Returns:
            Dict: 特征字典
        """
        feature_vector = block.get_feature_vector()
        
        # 计算边界框特征
        bbox_features = {}
        if block.bounding_box:
            bbox_features = {
                'width': block.bounding_box.width,
                'height': block.bounding_box.height,
                'aspect_ratio': block.bounding_box.aspect_ratio,
                'center': (block.bounding_box.center.x, block.bounding_box.center.y)
            }
        
        # 计算实体类型分布
        type_counts = block.count_entity_types()
        entity_types = {etype.value: count for etype, count in type_counts.items() if count > 0}
        
        # 实体分布百分比
        total_entities = sum(type_counts.values())
        type_percentages = {}
        if total_entities > 0:
            for etype, count in type_counts.items():
                if count > 0:
                    type_percentages[etype.value] = count / total_entities * 100.0
        
        # 构建特征字典
        return {
            'block_name': block.name,
            'entity_count': len(block.entities),
            'entity_types': entity_types,
            'entity_type_percentages': type_percentages,
            'bbox': bbox_features,
            'is_arrow': block.is_arrow,
            'unique_layers': len(set(e.layer for e in block.entities))
        }
    
    def analyze_block_pattern(self, block: Block) -> Dict:
        """
        分析块的模式特征
        
        Args:
            block: 块对象
            
        Returns:
            Dict: 块模式特征
        """
        features = self.extract_feature_dict(block)
        
        # 增强特征分析
        enhanced_features = {}
        
        # 1. 分析长宽比和形状类型
        if 'bbox' in features and 'aspect_ratio' in features['bbox']:
            aspect_ratio = features['bbox']['aspect_ratio']
            if aspect_ratio > 3.0:
                shape_type = "elongated"
            elif 0.8 <= aspect_ratio <= 1.2:
                shape_type = "square"
            else:
                shape_type = "rectangular"
            enhanced_features['shape_type'] = shape_type
        
        # 2. 分析实体组合模式
        entity_types = features.get('entity_types', {})
        if EntityType.CIRCLE.value in entity_types and entity_types[EntityType.CIRCLE.value] > 0:
            if EntityType.LINE.value in entity_types and entity_types[EntityType.LINE.value] > 0:
                enhanced_features['composite_pattern'] = "circle_with_lines"
            else:
                enhanced_features['composite_pattern'] = "circle_dominant"
        elif EntityType.LINE.value in entity_types and entity_types[EntityType.LINE.value] > 0:
            if len(entity_types) > 1:
                enhanced_features['composite_pattern'] = "complex_lines"
            else:
                enhanced_features['composite_pattern'] = "simple_lines"
        
        # 3. 分析可能的功能类型
        if block.is_arrow:
            enhanced_features['functional_type'] = "arrow"
        elif 'composite_pattern' in enhanced_features:
            if enhanced_features['composite_pattern'] == "circle_with_lines":
                if features['entity_count'] < 10:
                    enhanced_features['functional_type'] = "valve"
                else:
                    enhanced_features['functional_type'] = "instrument"
            elif enhanced_features['composite_pattern'] == "complex_lines":
                if 'shape_type' in enhanced_features and enhanced_features['shape_type'] == "square":
                    enhanced_features['functional_type'] = "equipment"
                else:
                    enhanced_features['functional_type'] = "connector"
        
        # 合并增强特征
        features['enhanced_features'] = enhanced_features
        
        return features


class BlockIdentifier:
    """块识别器"""
    
    def __init__(self):
        """初始化块识别器"""
        self.feature_extractor = BlockFeatureExtractor()
        self.block_features = {}  # 存储块特征模板
    
    def add_block_template(self, name: str, block: Block, tolerance: float = 0.2):
        """
        添加块模板供识别
        
        Args:
            name: 特征名称
            block: 块对象
            tolerance: 匹配容差
        """
        features = BlockFeature.from_sample_block(block, name, tolerance=tolerance)
        self.block_features[name] = features
    
    def add_block_feature(self, feature: BlockFeature):
        """
        添加块特征
        
        Args:
            feature: 块特征对象
        """
        self.block_features[feature.name] = feature
    
    def identify_block(self, block: Block) -> List[Tuple[str, float]]:
        """
        识别块，返回匹配的特征及置信度
        
        Args:
            block: 块对象
            
        Returns:
            List[Tuple[str, float]]: 匹配的特征名称和置信度列表
        """
        matches = []
        
        for name, feature in self.block_features.items():
            confidence = 0.0
            
            if feature.matches(block):
                # 计算匹配度
                match_score = self._calculate_match_score(block, feature)
                
                # 基础匹配度
                confidence = match_score * 0.8
                
                # 进一步微调置信度
                if block.name == feature.name:
                    confidence = min(confidence + 0.15, 0.95)
                
                matches.append((name, confidence))
        
        # 按置信度排序
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def _calculate_match_score(self, block: Block, feature: BlockFeature) -> float:
        """
        计算块与特征的匹配度
        
        Args:
            block: 块对象
            feature: 块特征
            
        Returns:
            float: 0到1之间的匹配度
        """
        # 1. 检查实体类型匹配度
        block_entity_types = set(entity.entity_type for entity in block.entities)
        required_types = feature.entity_types
        
        if not all(rtype in block_entity_types for rtype in required_types):
            return 0.3  # 基本匹配，但缺少必要的实体类型
        
        # 2. 实体数量匹配度
        entity_count = len(block.entities)
        count_range = feature.max_entity_count - feature.min_entity_count
        
        if count_range == 0:  # 避免除以零
            count_score = 1.0 if entity_count == feature.min_entity_count else 0.5
        else:
            distance = min(abs(entity_count - feature.min_entity_count), 
                         abs(entity_count - feature.max_entity_count))
            count_score = 1.0 - (distance / (count_range * 2))
            count_score = max(0.3, min(count_score, 1.0))  # 限制在0.3到1.0之间
        
        # 3. 边界框特征匹配度
        bbox_score = 0.5  # 默认分数
        
        if block.bounding_box:
            # 长宽比匹配度
            aspect_ratio = block.bounding_box.aspect_ratio
            ar_range = feature.max_aspect_ratio - feature.min_aspect_ratio
            
            if ar_range == 0:  # 避免除以零
                ar_score = 1.0 if aspect_ratio == feature.min_aspect_ratio else 0.5
            else:
                ar_distance = min(abs(aspect_ratio - feature.min_aspect_ratio), 
                               abs(aspect_ratio - feature.max_aspect_ratio))
                ar_score = 1.0 - (ar_distance / (ar_range * 2))
                ar_score = max(0.3, min(ar_score, 1.0))
            
            # 尺寸匹配度（简化）
            size_score = 0.7  # 默认大小匹配分数
            
            bbox_score = (ar_score * 0.7 + size_score * 0.3)
        
        # 综合计算匹配度
        total_score = (
            0.4 * (1.0 if all(rtype in block_entity_types for rtype in required_types) else 0.5) +  # 类型匹配权重
            0.3 * count_score +  # 数量匹配权重
            0.3 * bbox_score     # 边界框匹配权重
        )
        
        return total_score
    
    def is_arrow_block(self, block: Block) -> bool:
        """
        检查块是否是箭头
        
        Args:
            block: 块对象
            
        Returns:
            bool: 是否为箭头
        """
        # 直接检查
        if block.is_arrow:
            return True
        
        # 通过特征模板检查
        matches = self.identify_block(block)
        for name, confidence in matches:
            if "ARROW" in name.upper() and confidence > 0.7:
                return True
        
        # 如果没有直接判断，基于形状和组成元素进行试探性判断
        if block.bounding_box and block.bounding_box.aspect_ratio > 3.0:
            # 细长的形状
            type_counts = block.count_entity_types()
            line_count = type_counts[EntityType.LINE]
            polyline_count = type_counts[EntityType.POLYLINE] + type_counts[EntityType.LWPOLYLINE]
            
            # 细长且主要由线段组成的可能是箭头
            if line_count > 1 and line_count + polyline_count >= 3:
                return True
        
        return False
    
    def save_templates(self, file_path: str):
        """
        保存块特征模板到文件
        
        Args:
            file_path: 文件路径
        """
        templates = {name: feature.to_dict() for name, feature in self.block_features.items()}
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(templates, f, indent=2)
            return True
        except Exception as e:
            print(f"保存特征模板时出错: {e}")
            return False
    
    def load_templates(self, file_path: str):
        """
        从文件加载块特征模板
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 操作是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            self.block_features = {}
            for name, feature_dict in templates.items():
                self.block_features[name] = BlockFeature.from_dict(feature_dict)
            
            return True
        except Exception as e:
            print(f"加载特征模板时出错: {e}")
            return False


class BlockClusterAnalyzer:
    """块聚类分析器"""
    
    def __init__(self, feature_extractor: BlockFeatureExtractor = None):
        """
        初始化块聚类分析器
        
        Args:
            feature_extractor: 特征提取器（可选）
        """
        self.feature_extractor = feature_extractor or BlockFeatureExtractor()
    
    def cluster_blocks(self, blocks: List[Block], n_clusters: int = 0, method: str = 'kmeans') -> Dict[int, List[Block]]:
        """
        对块进行聚类
        
        Args:
            blocks: 块列表
            n_clusters: 聚类数量（0表示自动确定）
            method: 聚类方法，可选 'kmeans'、'hierarchy'、'dbscan'
            
        Returns:
            Dict[int, List[Block]]: 聚类结果，键为聚类ID，值为块列表
        """
        if len(blocks) == 0:
            return {}
        
        # 提取特征
        features = np.array([self.feature_extractor.extract_features(block) for block in blocks])
        
        # 归一化特征
        features = self._normalize_features(features)
        
        # 确定聚类数量
        if n_clusters <= 0:
            # 使用轮廓系数或肘部方法自动确定聚类数量
            n_clusters = self._estimate_clusters(features, max_clusters=min(10, len(blocks)))
        
        # 应用聚类
        if method == 'kmeans':
            labels = self._kmeans_clustering(features, n_clusters)
        elif method == 'hierarchy':
            labels = self._hierarchical_clustering(features, n_clusters)
        elif method == 'dbscan':
            labels = self._dbscan_clustering(features)
        else:
            # 默认使用K-means
            labels = self._kmeans_clustering(features, n_clusters)
        
        # 整理聚类结果
        clusters = {}
        for i, block in enumerate(blocks):
            cluster_id = labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(block)
        
        return clusters
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征
        
        Args:
            features: 特征数组
            
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 简单的最小-最大归一化
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        
        # 避免除以零
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (features - min_vals) / range_vals
        
        return normalized
    
    def _estimate_clusters(self, features: np.ndarray, max_clusters: int = 10) -> int:
        """
        估计最佳聚类数量
        
        Args:
            features: 特征数组
            max_clusters: 最大聚类数量
            
        Returns:
            int: 最佳聚类数量
        """
        try:
            # 尝试从scikit-learn导入需要的工具
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # 使用轮廓系数估计
            silhouette_scores = []
            for n in range(2, min(max_clusters + 1, len(features))):
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores.append((n, score))
            
            # 返回轮廓系数最高的聚类数量
            best_n, _ = max(silhouette_scores, key=lambda x: x[1])
            return best_n
        
        except ImportError:
            # 如果scikit-learn不可用，使用简单启发式方法
            n_samples = len(features)
            if n_samples <= 10:
                return max(2, n_samples // 2)
            elif n_samples <= 50:
                return 4
            else:
                return min(8, max_clusters)
    
    def _kmeans_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        K-means聚类
        
        Args:
            features: 特征数组
            n_clusters: 聚类数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(features)
        except ImportError:
            # 简单实现的K-means
            return self._simple_kmeans(features, n_clusters)
    
    def _hierarchical_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        层次聚类
        
        Args:
            features: 特征数组
            n_clusters: 聚类数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            return clustering.fit_predict(features)
        except ImportError:
            # 如果scikit-learn不可用，退回到简单的K-means
            return self._simple_kmeans(features, n_clusters)
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        DBSCAN聚类
        
        Args:
            features: 特征数组
            
        Returns:
            np.ndarray: 聚类标签
        """
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            return dbscan.fit_predict(features)
        except ImportError:
            # 如果scikit-learn不可用，退回到简单的K-means
            n_clusters = max(2, len(features) // 10)  # 简单估计聚类数量
            return self._simple_kmeans(features, n_clusters)
    
    def _simple_kmeans(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        简单的K-means实现（不依赖scikit-learn）
        
        Args:
            features: 特征数组
            n_clusters: 聚类数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        n_samples, n_features = features.shape
        
        # 随机初始化聚类中心
        np.random.seed(42)
        centroids = features[np.random.choice(n_samples, n_clusters, replace=False)]
        
        # 初始化标签
        labels = np.zeros(n_samples, dtype=int)
        
        # 最大迭代次数
        max_iterations = 100
        
        for _ in range(max_iterations):
            old_labels = labels.copy()
            
            # 为每个样本分配最近的中心
            for i in range(n_samples):
                distances = np.sqrt(np.sum((centroids - features[i])**2, axis=1))
                labels[i] = np.argmin(distances)
            
            # 如果标签没有变化，则收敛
            if np.all(labels == old_labels):
                break
            
            # 更新聚类中心
            for j in range(n_clusters):
                cluster_points = features[labels == j]
                if len(cluster_points) > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
        
        return labels


class BlockLayoutAnalyzer:
    """块布局分析器"""
    
    def analyze_spatial_distribution(self, blocks: List[Block]) -> Dict:
        """
        分析块的空间分布
        
        Args:
            blocks: 块列表
            
        Returns:
            Dict: 空间分布分析结果
        """
        if not blocks:
            return {
                "empty": True,
                "message": "No blocks to analyze"
            }
        
        # 获取所有块的中心点
        centers = []
        for block in blocks:
            if block.bounding_box and block.bounding_box.center:
                centers.append((
                    block.bounding_box.center.x,
                    block.bounding_box.center.y
                ))
        
        if not centers:
            return {
                "empty": True,
                "message": "No valid block centers found"
            }
        
        # 计算边界
        min_x = min(c[0] for c in centers)
        max_x = max(c[0] for c in centers)
        min_y = min(c[1] for c in centers)
        max_y = max(c[1] for c in centers)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 计算密度
        area = width * height if width > 0 and height > 0 else 1.0
        density = len(centers) / area
        
        # 划分网格进行密度分析
        grid_size = 5  # 5x5网格
        grid_density = self._calculate_grid_density(centers, min_x, max_x, min_y, max_y, grid_size)
        
        # 查找密度热点
        hotspots = []
        for i in range(grid_size):
            for j in range(grid_size):
                if grid_density[i][j] > density * 1.5:  # 密度高于平均值50%
                    grid_x = min_x + (max_x - min_x) * (i + 0.5) / grid_size
                    grid_y = min_y + (max_y - min_y) * (j + 0.5) / grid_size
                    hotspots.append({
                        "position": (grid_x, grid_y),
                        "density": grid_density[i][j]
                    })
        
        # 计算平均距离
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                distances.append(distance)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        return {
            "empty": False,
            "bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "width": width,
                "height": height
            },
            "block_count": len(blocks),
            "avg_density": density,
            "avg_distance": avg_distance,
            "hotspots": hotspots,
            "grid_density": grid_density
        }
    
    def _calculate_grid_density(self, centers, min_x, max_x, min_y, max_y, grid_size):
        """
        计算网格密度
        
        Args:
            centers: 中心点列表
            min_x, max_x, min_y, max_y: 边界
            grid_size: 网格大小
            
        Returns:
            List[List[float]]: 网格密度
        """
        grid_density = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        width = max_x - min_x
        height = max_y - min_y
        
        for cx, cy in centers:
            # 确定点所在的网格单元
            if width > 0:
                grid_x = min(grid_size - 1, int((cx - min_x) / width * grid_size))
            else:
                grid_x = 0
                
            if height > 0:
                grid_y = min(grid_size - 1, int((cy - min_y) / height * grid_size))
            else:
                grid_y = 0
            
            grid_density[grid_x][grid_y] += 1
        
        # 计算每个单元的密度
        cell_area = (width / grid_size) * (height / grid_size)
        if cell_area > 0:
            for i in range(grid_size):
                for j in range(grid_size):
                    grid_density[i][j] /= cell_area
        
        return grid_density
    
    def detect_patterns(self, blocks: List[Block]) -> List[Dict]:
        """
        检测块的排列模式
        
        Args:
            blocks: 块列表
            
        Returns:
            List[Dict]: 检测到的模式列表
        """
        if not blocks:
            return []
        
        patterns = []
        
        # 检测线性排列
        linear_patterns = self._detect_linear_arrangements(blocks)
        if linear_patterns:
            patterns.extend(linear_patterns)
        
        # 检测网格排列
        grid_patterns = self._detect_grid_arrangements(blocks)
        if grid_patterns:
            patterns.extend(grid_patterns)
        
        # 检测放射状排列
        radial_patterns = self._detect_radial_arrangements(blocks)
        if radial_patterns:
            patterns.extend(radial_patterns)
        
        return patterns
    
    def _detect_linear_arrangements(self, blocks: List[Block]) -> List[Dict]:
        """
        检测线性排列
        
        Args:
            blocks: 块列表
            
        Returns:
            List[Dict]: 线性排列模式列表
        """
        # 获取块中心点
        centers = []
        for block in blocks:
            if block.bounding_box and block.bounding_box.center:
                centers.append({
                    'block': block,
                    'center': (block.bounding_box.center.x, block.bounding_box.center.y)
                })
        
        if len(centers) < 3:
            return []
        
        linear_patterns = []
        
        # 尝试不同的起点
        for start_idx in range(len(centers)):
            start = centers[start_idx]
            
            # 对其他点按照与起点的角度排序
            points_with_angles = []
            for i, point in enumerate(centers):
                if i != start_idx:
                    dx = point['center'][0] - start['center'][0]
                    dy = point['center'][1] - start['center'][1]
                    angle = math.atan2(dy, dx)
                    distance = math.sqrt(dx*dx + dy*dy)
                    points_with_angles.append({
                        'block': point['block'],
                        'center': point['center'],
                        'angle': angle,
                        'distance': distance
                    })
            
            # 按角度分组
            angle_groups = {}
            for point in points_with_angles:
                # 将角度分组，容差为0.1弧度
                key = round(point['angle'] / 0.1) * 0.1
                if key not in angle_groups:
                    angle_groups[key] = []
                angle_groups[key].append(point)
            
            # 查找数量超过2的角度组（加上起点至少3个点）
            for angle, group in angle_groups.items():
                if len(group) >= 2:
                    # 按距离排序
                    group.sort(key=lambda x: x['distance'])
                    
                    # 检查距离是否近似等间距
                    distances = [p['distance'] for p in group]
                    avg_diff = sum(distances[i+1] - distances[i] for i in range(len(distances) - 1)) / (len(distances) - 1) if len(distances) > 1 else 0
                    
                    is_equidistant = True
                    for i in range(len(distances) - 1):
                        if abs((distances[i+1] - distances[i]) - avg_diff) > avg_diff * 0.2:  # 20%的容差
                            is_equidistant = False
                            break
                    
                    # 加入起点构建完整的线性排列
                    blocks_in_line = [start['block']] + [p['block'] for p in group]
                    
                    linear_patterns.append({
                        'type': 'linear',
                        'blocks': blocks_in_line,
                        'angle': angle,
                        'is_equidistant': is_equidistant
                    })
        
        return linear_patterns
    
    def _detect_grid_arrangements(self, blocks: List[Block]) -> List[Dict]:
        """
        检测网格排列
        
        Args:
            blocks: 块列表
            
        Returns:
            List[Dict]: 网格排列模式列表
        """
        # 获取块中心点
        centers = []
        for block in blocks:
            if block.bounding_box and block.bounding_box.center:
                centers.append({
                    'block': block,
                    'center': (block.bounding_box.center.x, block.bounding_box.center.y)
                })
        
        if len(centers) < 4:  # 至少需要四个点来形成网格
            return []
        
        # 这里实现一个简化的网格检测
        # 完整实现会更复杂，需要考虑更多的几何约束
        
        # 按x坐标聚类可能的列
        x_coords = [c['center'][0] for c in centers]
        x_clusters = self._cluster_coordinates(x_coords)
        
        # 按y坐标聚类可能的行
        y_coords = [c['center'][1] for c in centers]
        y_clusters = self._cluster_coordinates(y_coords)
        
        # 需要至少2行2列才能形成网格
        if len(x_clusters) < 2 or len(y_clusters) < 2:
            return []
        
        # 检查是否每个网格单元都有点
        grid = {}
        for point in centers:
            x_cluster = self._find_cluster(point['center'][0], x_clusters)
            y_cluster = self._find_cluster(point['center'][1], y_clusters)
            
            if x_cluster is not None and y_cluster is not None:
                key = (x_cluster, y_cluster)
                if key not in grid:
                    grid[key] = []
                grid[key].append(point['block'])
        
        # 检查填充率
        fill_count = sum(1 for _ in grid.keys())
        expected_count = len(x_clusters) * len(y_clusters)
        fill_rate = fill_count / expected_count if expected_count > 0 else 0
        
        if fill_rate >= 0.5:  # 至少填充了一半的单元
            return [{
                'type': 'grid',
                'rows': len(y_clusters),
                'columns': len(x_clusters),
                'fill_rate': fill_rate,
                'blocks': [block for cell_blocks in grid.values() for block in cell_blocks]
            }]
        
        return []
    
    def _cluster_coordinates(self, coords, tolerance=10.0):
        """
        聚类一维坐标
        
        Args:
            coords: 坐标列表
            tolerance: 聚类容差
            
        Returns:
            List[float]: 聚类中心列表
        """
        if not coords:
            return []
        
        # 排序坐标
        sorted_coords = sorted(coords)
        
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for i in range(1, len(sorted_coords)):
            if sorted_coords[i] - sorted_coords[i-1] <= tolerance:
                current_cluster.append(sorted_coords[i])
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [sorted_coords[i]]
        
        # 添加最后一个簇
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _find_cluster(self, value, clusters, tolerance=10.0):
        """
        找到值所属的聚类
        
        Args:
            value: 要查找的值
            clusters: 聚类中心列表
            tolerance: 匹配容差
            
        Returns:
            int: 聚类索引，如果没有找到则返回None
        """
        for i, cluster in enumerate(clusters):
            if abs(value - cluster) <= tolerance:
                return i
        return None
    
    def _detect_radial_arrangements(self, blocks: List[Block]) -> List[Dict]:
        """
        检测放射状排列
        
        Args:
            blocks: 块列表
            
        Returns:
            List[Dict]: 放射状排列模式列表
        """
        # 获取块中心点
        centers = []
        for block in blocks:
            if block.bounding_box and block.bounding_box.center:
                centers.append({
                    'block': block,
                    'center': (block.bounding_box.center.x, block.bounding_box.center.y)
                })
        
        if len(centers) < 4:  # 至少需要中心点和三个辐射点
            return []
        
        radial_patterns = []
        
        # 尝试每个点作为中心
        for center_idx, center_point in enumerate(centers):
            # 计算其他点到中心的距离和角度
            points_with_angles = []
            for i, point in enumerate(centers):
                if i != center_idx:
                    dx = point['center'][0] - center_point['center'][0]
                    dy = point['center'][1] - center_point['center'][1]
                    angle = math.atan2(dy, dx)
                    distance = math.sqrt(dx*dx + dy*dy)
                    points_with_angles.append({
                        'block': point['block'],
                        'angle': angle,
                        'distance': distance
                    })
            
            # 检查距离是否近似相等
            distances = [p['distance'] for p in points_with_angles]
            avg_distance = sum(distances) / len(distances) if distances else 0
            
            # 计算距离的标准差占平均值的比例
            distance_std = math.sqrt(sum((d - avg_distance)**2 for d in distances) / len(distances)) if distances else 0
            distance_variation = distance_std / avg_distance if avg_distance > 0 else float('inf')
            
            # 检查角度是否均匀分布
            angles = sorted([p['angle'] for p in points_with_angles])
            angle_diffs = [(angles[i+1] - angles[i]) % (2 * math.pi) for i in range(len(angles) - 1)]
            angle_diffs.append((angles[0] - angles[-1] + 2 * math.pi) % (2 * math.pi))
            
            avg_angle_diff = sum(angle_diffs) / len(angle_diffs) if angle_diffs else 0
            angle_std = math.sqrt(sum((d - avg_angle_diff)**2 for d in angle_diffs) / len(angle_diffs)) if angle_diffs else 0
            angle_variation = angle_std / avg_angle_diff if avg_angle_diff > 0 else float('inf')
            
            # 判断是否构成放射状模式
            if (distance_variation < 0.3 and  # 距离变异系数小于30%
                angle_variation < 0.3 and     # 角度变异系数小于30%
                len(points_with_angles) >= 3):  # 至少有3个辐射点
                
                blocks_in_pattern = [center_point['block']] + [p['block'] for p in points_with_angles]
                
                radial_patterns.append({
                    'type': 'radial',
                    'center_block': center_point['block'],
                    'blocks': blocks_in_pattern,
                    'avg_distance': avg_distance,
                    'avg_angle_diff': avg_angle_diff
                })
        
        return radial_patterns