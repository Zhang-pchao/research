#!/usr/bin/env python3
"""
计算氮气气泡质心的脚本
结合MDAnalysis读取LAMMPS轨迹和并查集聚类分析

更新说明 (PBC支持):
- 支持读取extended xyz格式的离子文件（包含PBC信息）
- 自动解析lattice="a 0.0 0.0 0.0 b 0.0 0.0 0.0 c"格式的盒子尺寸
- 在计算离子-气泡距离时优先使用从离子文件读取的盒子尺寸
- 确保所有距离计算都正确考虑周期性边界条件
"""

import os
import sys
import numpy as np
import argparse
import time
import re
from collections import defaultdict
import logging

# 导入MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

class UnionFind:
    """并查集数据结构用于聚类分析"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

class BubbleCentroidCalculator:
    """气泡质心计算器"""
    
    def __init__(self, cutoff_distance=5.5):
        self.cutoff_distance = cutoff_distance
        
        # LAMMPS原子类型到化学元素的映射
        self.type_to_element = {
            '1': 'H',   
            '2': 'O',     
            '3': 'N',  
            '4': 'Na',   
            '5': 'Cl',
            '6': 'Ti',
        }
        
        # 获取氮原子的类型编号
        self.nitrogen_type = None
        for atom_type, element in self.type_to_element.items():
            if element == 'N':
                self.nitrogen_type = int(atom_type)
                break
        
        # 初始化离子盒子数据存储
        self.ion_box_data = {}
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_progress(self, message, flush=True):
        """输出带时间戳的进度信息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if flush:
            sys.stdout.flush()
    
    def periodic_distance(self, coord1, coord2, box_dims):
        """计算考虑周期性边界条件的距离"""
        diff = coord1 - coord2
        for i in range(3):
            box_length = box_dims[i]
            diff[i] = diff[i] - box_length * round(diff[i] / box_length)
        return np.linalg.norm(diff)
    
    def cluster_nitrogen_atoms(self, n_coords, box_dims):
        """使用并查集进行氮原子聚类分析"""
        if len(n_coords) == 0:
            return [], []
        
        n_points = len(n_coords)
        uf = UnionFind(n_points)
        
        self.logger.info(f"开始聚类分析 {n_points} 个氮原子...")
        
        # 构建邻接关系
        for i in range(n_points):
            for j in range(i+1, n_points):
                if self.periodic_distance(n_coords[i], n_coords[j], box_dims) <= self.cutoff_distance:
                    uf.union(i, j)
        
        # 收集聚类结果
        clusters_dict = defaultdict(list)
        for atom in range(n_points):
            clusters_dict[uf.find(atom)].append(atom)
        
        # 转换为列表格式，按大小排序
        clusters = sorted(clusters_dict.values(), key=len, reverse=True)
        
        self.logger.info(f"发现 {len(clusters)} 个氮原子簇")
        if clusters:
            self.logger.info(f"最大簇包含 {len(clusters[0])} 个氮原子")
        
        return clusters
    
    def calculate_centroid_pbc(self, coords, box_dims):
        """考虑周期性边界条件计算质心"""
        centroid = np.zeros(3)
        
        for dim in range(3):
            box_length = box_dims[dim]
            
            # 转换为角度坐标
            angles = 2 * np.pi * coords[:, dim] / box_length
            
            # 计算平均角度
            cos_mean = np.mean(np.cos(angles))
            sin_mean = np.mean(np.sin(angles))
            mean_angle = np.arctan2(sin_mean, cos_mean)
            
            # 转换回笛卡尔坐标
            centroid[dim] = (mean_angle * box_length) / (2 * np.pi)
            if centroid[dim] < 0:
                centroid[dim] += box_length
        
        return centroid
    
    def read_lammps_with_mda(self, data_file, traj_file, atom_style=None):
        """使用MDAnalysis读取LAMMPS文件"""
        try:
            self.logger.info(f"使用MDAnalysis读取LAMMPS文件...")
            self.logger.info(f"数据文件: {data_file}")
            self.logger.info(f"轨迹文件: {traj_file}")
            
            # 如果用户指定了atom_style，使用指定的格式
            if atom_style:
                self.logger.info(f"使用用户指定的 atom_style='{atom_style}'")
                u = mda.Universe(data_file, traj_file, 
                               atom_style=atom_style,
                               format='LAMMPSDUMP')
            else:
                # 尝试不同的方法读取LAMMPS文件
                try:
                    self.logger.info("尝试使用 atom_style='id type x y z'")
                    u = mda.Universe(data_file, traj_file, 
                                   atom_style='id type x y z',
                                   format='LAMMPSDUMP')
                except Exception as e1:
                    self.logger.warning(f"方法1失败: {e1}")
                    try:
                        self.logger.info("尝试使用 atom_style='atomic'")
                        u = mda.Universe(data_file, traj_file, 
                                       atom_style='atomic',
                                       format='LAMMPSDUMP')
                    except Exception as e2:
                        self.logger.warning(f"方法2失败: {e2}")
                        self.logger.info("尝试使用 atom_style='full'")
                        u = mda.Universe(data_file, traj_file, 
                                       atom_style='full',
                                       format='LAMMPSDUMP')
            
            self.logger.info(f"成功读取 {len(u.atoms)} 个原子")
            self.logger.info(f"轨迹包含 {len(u.trajectory)} 帧")
            
            return u
            
        except Exception as e:
            self.logger.error(f"使用MDAnalysis读取LAMMPS文件时出错: {str(e)}")
            raise
    
    def process_trajectory(self, data_file, traj_file, atom_style=None, output_file="bubble_centroids.txt", 
                         step_interval=1, start_frame=0, end_frame=-1, ion_files=None, ions_analysis_output=None):
        """处理轨迹文件并计算气泡质心，可选离子分析"""
        
        # 检查氮原子类型是否找到
        if self.nitrogen_type is None:
            raise ValueError("未在 type_to_element 映射中找到氮原子类型")
        
        self.logger.info(f"从type_to_element映射获取氮原子类型: {self.nitrogen_type}")
        
        # 读取轨迹
        u = self.read_lammps_with_mda(data_file, traj_file, atom_style)
        
        # 计算要分析的帧
        total_frames = len(u.trajectory)
        actual_end_frame = total_frames if end_frame == -1 else min(end_frame, total_frames)
        frames_to_analyze = list(range(start_frame, actual_end_frame, step_interval))
        
        self.logger.info(f"轨迹总帧数: {total_frames}")
        self.logger.info(f"分析帧范围: {start_frame} - {actual_end_frame-1}")
        self.logger.info(f"帧间隔: {step_interval}")
        self.logger.info(f"将分析 {len(frames_to_analyze)} 帧")
        
        if not frames_to_analyze:
            raise ValueError("没有要分析的帧，请检查帧范围设置")
        
        # 读取离子数据（如果提供）
        ion_frames_data = {}
        if ion_files:
            self.logger.info("开始读取离子文件，支持extended xyz格式的PBC信息...")
            ion_configs = {
                'H3O': {'file': ion_files.get('h3o'), 'atoms_per_molecule': 4},
                'bulk_OH': {'file': ion_files.get('bulk_oh'), 'atoms_per_molecule': 2},
                'surface_OH': {'file': ion_files.get('surface_oh'), 'atoms_per_molecule': 2},
                'surface_H': {'file': ion_files.get('surface_h'), 'atoms_per_molecule': 1},
                'Na': {'file': ion_files.get('na'), 'atoms_per_molecule': 1},
                'Cl': {'file': ion_files.get('cl'), 'atoms_per_molecule': 1}
            }
            
            for ion_name, config in ion_configs.items():
                if config['file']:
                    ion_frames_data[ion_name] = self.read_xyz_file_with_frame_filter(
                        config['file'], set(frames_to_analyze), config['atoms_per_molecule'], ion_name)
            
            # 输出盒子信息统计
            if hasattr(self, 'ion_box_data') and self.ion_box_data:
                self.logger.info(f"从离子文件中成功解析了 {len(self.ion_box_data)} 帧的盒子尺寸信息")
                sample_frame = next(iter(self.ion_box_data))
                sample_box = self.ion_box_data[sample_frame]
                self.logger.info(f"示例盒子尺寸 (帧{sample_frame}): {sample_box}")
            else:
                self.logger.warning("未从离子文件中解析到盒子尺寸信息，将使用LAMMPS轨迹的盒子尺寸")
        
        # 准备输出数据
        centroids_data = []
        times = []
        bubble_sizes = []
        frame_numbers = []
        
        # 离子分析数据
        ions_distance_data = {ion_name: [] for ion_name in ion_frames_data.keys()}
        
        self.logger.info("开始处理轨迹帧...")
        
        # 遍历选定的帧
        for i, frame_idx in enumerate(frames_to_analyze):
            # 跳转到指定帧
            u.trajectory[frame_idx]
            ts = u.trajectory.ts
            
            self.logger.info(f"处理第 {i+1}/{len(frames_to_analyze)} 个选定帧 (帧索引: {frame_idx}, 时间: {ts.time})")
            
            # 获取盒子尺寸
            box_dims = u.dimensions[:3]  # 只取前三个维度 (x, y, z)
            
            # 选择氮原子
            nitrogen_atoms = u.select_atoms(f"type {self.nitrogen_type}")
            
            if len(nitrogen_atoms) == 0:
                self.logger.warning(f"帧 {frame_idx} 中没有找到氮原子")
                continue
            
            # 获取氮原子坐标
            n_coords = nitrogen_atoms.positions
            
            # 聚类分析
            clusters = self.cluster_nitrogen_atoms(n_coords, box_dims)
            
            if not clusters:
                self.logger.warning(f"帧 {frame_idx} 中没有找到氮原子簇")
                continue
            
            # 找到最大的氮气气泡
            largest_cluster_indices = clusters[0]  # 已经按大小排序
            largest_cluster_coords = n_coords[largest_cluster_indices]
            
            # 计算最大气泡的质心
            centroid = self.calculate_centroid_pbc(largest_cluster_coords, box_dims)
            
            # 记录数据
            times.append(ts.time)
            centroids_data.append(centroid)
            bubble_sizes.append(len(largest_cluster_indices))
            frame_numbers.append(frame_idx)
            
            self.logger.info(f"帧 {frame_idx}: 最大气泡包含 {len(largest_cluster_indices)} 个氮原子")
            self.logger.info(f"质心坐标: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
            
            # 离子分析（如果有离子数据）
            if ion_frames_data:
                # 找到气泡表面原子
                surface_coords = self.find_bubble_surface_n2_atoms(largest_cluster_coords, box_dims)
                
                # 分析每种离子
                for ion_name, ion_frames in ion_frames_data.items():
                    if frame_idx in ion_frames:
                        molecules = ion_frames[frame_idx]
                        if molecules:
                            # 计算离子距离，传递frame_idx以便使用正确的盒子尺寸
                            distances_data = self.calculate_ion_bubble_distances(
                                molecules, centroid, surface_coords, box_dims, ion_name, frame_idx)
                            
                            ions_distance_data[ion_name].extend(distances_data)
                            
                            self.logger.info(f"帧 {frame_idx}: 分析了 {len(molecules)} 个{ion_name}分子/离子")
        
        # 保存气泡质心结果
        self.save_results(times, centroids_data, bubble_sizes, frame_numbers, output_file, 
                         step_interval, start_frame, actual_end_frame)
        
        # 保存原始离子距离数据（如果有数据）
        if any(ions_distance_data.values()) and ions_analysis_output:
            self.save_raw_ion_distances(ions_distance_data, ions_analysis_output)
        
        # 分析并绘制离子分布（如果有数据）
        if any(ions_distance_data.values()) and ions_analysis_output:
            total_ions = sum(len(distances) for distances in ions_distance_data.values())
            self.logger.info(f"开始离子分布分析，共 {total_ions} 个数据点")
            self.plot_all_ions_distance_distributions(ions_distance_data, ions_analysis_output)
        elif ion_files and not any(ions_distance_data.values()):
            self.logger.warning("提供了离子文件但没有找到有效的离子数据")
        
        return times, centroids_data, bubble_sizes
    
    def save_results(self, times, centroids_data, bubble_sizes, frame_numbers, output_file, 
                   step_interval, start_frame, end_frame):
        """保存计算结果"""
        
        # 保存质心坐标
        centroids_file = output_file
        with open(centroids_file, 'w') as f:
            f.write("# FrameIndex Time(ps) X(Å) Y(Å) Z(Å) BubbleSize\n")
            for frame_idx, time, centroid, size in zip(frame_numbers, times, centroids_data, bubble_sizes):
                f.write(f"{frame_idx} {time:.1f} {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} {size}\n")
        
        self.logger.info(f"质心坐标已保存到: {centroids_file}")
        
        # 保存统计信息
        stats_file = output_file.replace('.txt', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("# 气泡质心计算统计信息\n")
            f.write(f"# 分析帧数: {len(times)}\n")
            f.write(f"# 帧筛选设置:\n")
            f.write(f"#   起始帧: {start_frame}\n")
            f.write(f"#   结束帧: {end_frame}\n")
            f.write(f"#   帧间隔: {step_interval}\n")
            f.write(f"# 截断距离: {self.cutoff_distance} Å\n")
            f.write(f"# 氮原子类型: {self.nitrogen_type} (来自type_to_element映射)\n")
            if times:
                f.write(f"# 时间范围: {min(times):.1f} - {max(times):.1f} ps\n")
                f.write(f"# 帧索引范围: {min(frame_numbers)} - {max(frame_numbers)}\n")
                f.write(f"# 平均气泡大小: {np.mean(bubble_sizes):.1f} 个氮原子\n")
                f.write(f"# 最大气泡大小: {max(bubble_sizes)} 个氮原子\n")
                f.write(f"# 最小气泡大小: {min(bubble_sizes)} 个氮原子\n")
        
        self.logger.info(f"统计信息已保存到: {stats_file}")
    
    def parse_lattice_from_extended_xyz(self, header_line):
        """从extended xyz文件的header行解析lattice信息"""
        try:
            # 寻找lattice信息
            lattice_match = re.search(r'lattice="([^"]+)"', header_line)
            if lattice_match:
                lattice_str = lattice_match.group(1)
                lattice_values = list(map(float, lattice_str.split()))
                
                # 对于正交盒子格式: "a 0.0 0.0 0.0 b 0.0 0.0 0.0 c"
                if len(lattice_values) == 9:
                    box_dims = np.array([lattice_values[0], lattice_values[4], lattice_values[8]])
                    self.logger.debug(f"从extended xyz解析的盒子尺寸: {box_dims}")
                    return box_dims
                else:
                    self.logger.warning(f"不支持的lattice格式: {lattice_str}")
                    return None
            else:
                self.logger.warning("未找到lattice信息")
                return None
                
        except Exception as e:
            self.logger.error(f"解析lattice信息失败: {e}")
            return None
    
    def read_xyz_file_with_frame_filter(self, xyz_file, frames_to_analyze, molecules_per_group, molecule_name):
        """读取xyz文件并根据帧筛选条件过滤，支持extended xyz格式的PBC信息"""
        frames_data = {}  # {frame: [molecules]}
        frames_box_data = {}  # {frame: box_dims}
        
        if not os.path.exists(xyz_file):
            self.logger.warning(f"{molecule_name}文件不存在: {xyz_file}")
            return frames_data
        
        self.logger.info(f"读取{molecule_name}文件: {xyz_file}")
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # 读取原子数
                if lines[i].strip().isdigit():
                    n_atoms = int(lines[i].strip())
                    i += 1
                    
                    # 读取帧信息和extended xyz header
                    frame_line = lines[i].strip()
                    frame_match = re.search(r'Frame[=\s](\d+)', frame_line)
                    if frame_match:
                        frame = int(frame_match.group(1))
                        
                        # 解析盒子尺寸信息
                        box_dims = self.parse_lattice_from_extended_xyz(frame_line)
                        
                        i += 1
                        
                        # 只处理需要分析的帧
                        if frame in frames_to_analyze:
                            # 读取原子坐标
                            atoms = []
                            for j in range(n_atoms):
                                if i + j < len(lines):
                                    parts = lines[i + j].strip().split()
                                    if len(parts) >= 4:
                                        element = parts[0]
                                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                        atoms.append((element, x, y, z))
                            
                            # 解析分子
                            molecules = self.parse_molecules(atoms, molecules_per_group, molecule_name)
                            frames_data[frame] = molecules
                            
                            # 存储盒子尺寸信息
                            if box_dims is not None:
                                frames_box_data[frame] = box_dims
                            
                        i += n_atoms
                    else:
                        i += 1
                else:
                    i += 1
            
            self.logger.info(f"成功读取{len(frames_data)}帧{molecule_name}数据")
            total_molecules = sum(len(molecules) for molecules in frames_data.values())
            self.logger.info(f"总{molecule_name}分子/离子数: {total_molecules}")
            
            # 将盒子信息存储到实例变量中供后续使用
            # 如果还没有盒子数据，使用当前文件的数据
            if not hasattr(self, 'ion_box_data') or not self.ion_box_data:
                self.ion_box_data = frames_box_data
            else:
                # 合并盒子数据，优先使用已有的数据（假设所有离子文件的盒子尺寸相同）
                for frame, box_dims in frames_box_data.items():
                    if frame not in self.ion_box_data:
                        self.ion_box_data[frame] = box_dims
            
        except Exception as e:
            self.logger.error(f"读取{molecule_name}文件失败: {e}")
        
        return frames_data
    
    def parse_molecules(self, atoms, molecules_per_group, molecule_name):
        """解析分子结构"""
        molecules = []
        
        if molecule_name == "H3O":
            # H3O: O H H H (4个原子一组)
            for atom_idx in range(0, len(atoms), 4):
                if atom_idx + 3 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h1_atom = atoms[atom_idx + 1]
                    h2_atom = atoms[atom_idx + 2] 
                    h3_atom = atoms[atom_idx + 3]
                    
                    if (o_atom[0] == 'O' and h1_atom[0] == 'H' and 
                        h2_atom[0] == 'H' and h3_atom[0] == 'H'):
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coords = [
                            np.array([h1_atom[1], h1_atom[2], h1_atom[3]]),
                            np.array([h2_atom[1], h2_atom[2], h2_atom[3]]),
                            np.array([h3_atom[1], h3_atom[2], h3_atom[3]])
                        ]
                        molecules.append(('O', o_coord, h_coords))
        
        elif molecule_name in ["bulk_OH", "surface_OH"]:
            # OH: O H (2个原子一组)
            for atom_idx in range(0, len(atoms), 2):
                if atom_idx + 1 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h_atom = atoms[atom_idx + 1]
                    
                    if o_atom[0] == 'O' and h_atom[0] == 'H':
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coord = np.array([h_atom[1], h_atom[2], h_atom[3]])
                        molecules.append(('O', o_coord, [h_coord]))
        
        elif molecule_name == "surface_H":
            # H: 单个H原子
            for atom in atoms:
                if atom[0] == 'H':
                    h_coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append(('H', h_coord, []))
        
        elif molecule_name in ["Na", "Cl"]:
            # 离子: 单个原子
            expected_element = 'Na' if molecule_name == 'Na' else 'Cl'
            for atom in atoms:
                if atom[0] == expected_element:
                    coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append((expected_element, coord, []))
        
        return molecules
    
    def find_bubble_surface_n2_atoms(self, largest_cluster_coords, box_dims):
        """找到气泡表面的N2原子（最外层）"""
        if len(largest_cluster_coords) < 2:
            return largest_cluster_coords
        
        # 计算气泡质心
        centroid = self.calculate_centroid_pbc(largest_cluster_coords, box_dims)
        
        # 计算每个N原子到质心的距离
        distances_to_center = []
        for coord in largest_cluster_coords:
            dist = self.periodic_distance(coord, centroid, box_dims)
            distances_to_center.append(dist)
        
        # 找到最大距离的80%作为表面阈值
        max_dist = max(distances_to_center)
        surface_threshold = max_dist * 0.8
        
        # 选择表面原子
        surface_indices = [i for i, dist in enumerate(distances_to_center) 
                          if dist >= surface_threshold]
        surface_coords = largest_cluster_coords[surface_indices]
        
        self.logger.debug(f"气泡表面N原子数: {len(surface_coords)}/{len(largest_cluster_coords)}")
        
        return surface_coords
    
    def calculate_ion_bubble_distances(self, molecules, centroid, surface_coords, box_dims, molecule_name, frame_idx=None):
        """计算离子相对于气泡的距离，使用适当的PBC条件"""
        distances_data = []
        
        # 优先使用从离子文件中读取的盒子尺寸
        ion_box_dims = box_dims  # 默认使用LAMMPS轨迹的盒子尺寸
        if hasattr(self, 'ion_box_data') and frame_idx is not None and frame_idx in self.ion_box_data:
            ion_box_dims = self.ion_box_data[frame_idx]
            self.logger.debug(f"帧{frame_idx}: 使用从{molecule_name}文件读取的盒子尺寸: {ion_box_dims}")
        else:
            self.logger.debug(f"帧{frame_idx}: 使用LAMMPS轨迹的盒子尺寸: {box_dims}")
        
        for molecule in molecules:
            element, center_coord, other_coords = molecule
            
            # 计算中心原子到气泡质心的距离 d_centroid
            d_centroid = self.periodic_distance(center_coord, centroid, ion_box_dims)
            
            # 计算中心原子到气泡表面最近N2的距离 d_interface
            min_surface_dist = float('inf')
            for surface_coord in surface_coords:
                dist = self.periodic_distance(center_coord, surface_coord, ion_box_dims)
                if dist < min_surface_dist:
                    min_surface_dist = dist
            
            d_interface = min_surface_dist
            distances_data.append((d_centroid, d_interface))
        
        return distances_data
    
    def save_raw_ion_distances(self, ions_distance_data, output_dir):
        """保存原始离子距离数据 - 按离子类型分别保存"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 按离子类型分别保存
        total_count = 0
        for ion_name, distances in ions_distance_data.items():
            if distances:
                ion_file = os.path.join(output_dir, f"raw_{ion_name}_distances.txt")
                with open(ion_file, 'w') as f:
                    f.write(f"# {ion_name}离子原始距离数据\n")
                    f.write("# 格式: d_centroid(Å) d_interface(Å)\n")
                    f.write("d_centroid\td_interface\n")
                    
                    for d_cent, d_int in distances:
                        f.write(f"{d_cent:.6f}\t{d_int:.6f}\n")
                
                total_count += len(distances)
                self.logger.info(f"{ion_name}离子距离数据已保存: {ion_file} (共{len(distances)}个数据点)")
        
        self.logger.info(f"所有离子原始距离数据保存完成，共{total_count}个数据点")
    
    def setup_nature_style(self):
        """设置Nature期刊风格的matplotlib参数"""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'text.usetex': False,
            'mathtext.default': 'regular'
        })
    
    def plot_all_ions_distance_distributions(self, ions_distance_data, output_dir):
        """绘制所有离子距离分布的数密度图"""
        if not ions_distance_data:
            self.logger.warning("没有离子距离数据可绘制")
            return
        
        # 设置绘图风格
        self.setup_nature_style()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义离子颜色和标记
        ion_styles = {
            'H3O': {'color': '#1f77b4', 'marker': 'o', 'label': r'$\mathrm{H_3O^+}$'},
            'bulk_OH': {'color': '#ff7f0e', 'marker': 's', 'label': r'$\mathrm{OH^-(bulk)}$'},
            'surface_OH': {'color': '#2ca02c', 'marker': '^', 'label': r'$\mathrm{OH^-(surf)}$'},
            'surface_H': {'color': '#d62728', 'marker': 'v', 'label': r'$\mathrm{H^+(surf)}$'},
            'Na': {'color': '#9467bd', 'marker': 'D', 'label': r'$\mathrm{Na^+}$'},
            'Cl': {'color': '#8c564b', 'marker': 'p', 'label': r'$\mathrm{Cl^-}$'}
        }
        
        bins = 50
        
        # 绘制两个子图：d_centroid 和 d_interface
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 处理每种离子
        for ion_name, distances in ions_distance_data.items():
            if not distances:
                continue
                
            all_d_centroid = [d[0] for d in distances]
            all_d_interface = [d[1] for d in distances]
            
            style = ion_styles.get(ion_name, {'color': 'black', 'marker': 'o', 'label': ion_name})
            
            # d_centroid分布
            hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            
            ax1.plot(bin_centers_centroid, hist_centroid, 
                    marker=style['marker'], color=style['color'], 
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # d_interface分布  
            hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            
            ax2.plot(bin_centers_interface, hist_interface,
                    marker=style['marker'], color=style['color'],
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
        
        # 设置d_centroid子图
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Ion Distance to Bubble Centroid', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=11)
        
        # 设置d_interface子图
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Ion Distance to Bubble Surface', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=11)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(output_dir, "all_ions_bubble_distance_distributions.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 保存数值数据
        data_file = os.path.join(output_dir, "all_ions_distance_distribution_data.txt")
        with open(data_file, 'w') as f:
            f.write("# All ions distance distribution data\n")
            f.write("# Format: ion_type d_centroid_center d_centroid_density d_interface_center d_interface_density\n")
            
            for ion_name, distances in ions_distance_data.items():
                if not distances:
                    continue
                    
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                
                hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
                bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
                
                hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
                bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
                
                f.write(f"\n# {ion_name} data\n")
                max_len = max(len(bin_centers_centroid), len(bin_centers_interface))
                for i in range(max_len):
                    centroid_center = bin_centers_centroid[i] if i < len(bin_centers_centroid) else ""
                    centroid_density = hist_centroid[i] if i < len(hist_centroid) else ""
                    interface_center = bin_centers_interface[i] if i < len(bin_centers_interface) else ""
                    interface_density = hist_interface[i] if i < len(hist_interface) else ""
                    f.write(f"{ion_name}\t{centroid_center}\t{centroid_density}\t{interface_center}\t{interface_density}\n")
        

        
        self.logger.info(f"所有离子距离分布图已保存: {plot_file}")
        self.logger.info(f"分布数据已保存: {data_file}")
        
        # 打印统计信息
        for ion_name, distances in ions_distance_data.items():
            if distances:
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                self.logger.info(f"{ion_name}距离统计:")
                self.logger.info(f"  d_centroid: 平均={np.mean(all_d_centroid):.3f}±{np.std(all_d_centroid):.3f} Å")
                self.logger.info(f"  d_interface: 平均={np.mean(all_d_interface):.3f}±{np.std(all_d_interface):.3f} Å")

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='计算LAMMPS轨迹中氮气气泡的质心')
    
    # 必需参数
    parser.add_argument('--traj_file', default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_nacl_tio2_water_layer/6/0-1.6ns/bubble_1k.lammpstrj',  help='LAMMPS轨迹文件路径')
    
    # 可选参数
    parser.add_argument('--data', default='../model_atomic.data', help='LAMMPS数据文件路径 (默认: ../model_atomic.data)')
    parser.add_argument('--output', default='bubble_centroids.txt', help='输出文件名')
    parser.add_argument('--cutoff', type=float, default=5.5, help='氮原子聚类截断距离 (Å)')
    parser.add_argument('--atom_style', default="id type x y z", help='LAMMPS atom_style (例如: "id type x y z", "atomic", "full")')
    
    # 轨迹帧筛选参数
    parser.add_argument('--step_interval', type=int, default=1, help='分析帧间隔，每隔n帧分析一次')
    parser.add_argument('--start_frame', type=int, default=0, help='开始分析的帧数')
    parser.add_argument('--end_frame', type=int, default=-1, help='结束分析的帧数，-1表示到最后一帧')
    
    # 离子分析参数
    ion_base_path = '../find_ion_4/ion_analysis_results'
    parser.add_argument('--h3o_file', default=f'{ion_base_path}/solution_bulk_h3o.xyz', help='H3O离子轨迹文件路径')
    parser.add_argument('--bulk_oh_file', default=f'{ion_base_path}/solution_bulk_oh.xyz', help='体相OH离子轨迹文件路径')
    parser.add_argument('--surface_oh_file', default=f'{ion_base_path}/solution_surface_oh.xyz', help='表面OH离子轨迹文件路径')
    parser.add_argument('--surface_h_file', default=f'{ion_base_path}/tio2_surface_h.xyz', help='表面H离子轨迹文件路径')
    parser.add_argument('--na_file', default=f'{ion_base_path}/na_ions.xyz', help='Na离子轨迹文件路径')
    parser.add_argument('--cl_file', default=f'{ion_base_path}/cl_ions.xyz', help='Cl离子轨迹文件路径')
    parser.add_argument('--ions_output', default='ions_analysis', help='离子分析结果输出目录')
    parser.add_argument('--disable_ions', action='store_true', help='禁用离子分析，仅计算气泡质心')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = get_args()
    
    # 检查输入文件
    if not os.path.exists(args.traj_file):
        print(f"错误: 轨迹文件 {args.traj_file} 不存在")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"错误: 数据文件 {args.data} 不存在")
        print(f"提示: 可以使用 --data 参数指定正确的数据文件路径")
        sys.exit(1)
    
    # 检查离子文件（可选）
    ion_files = {}
    if not args.disable_ions:
        ion_file_configs = {
            'h3o': args.h3o_file,
            'bulk_oh': args.bulk_oh_file,
            'surface_oh': args.surface_oh_file,
            'surface_h': args.surface_h_file,
            'na': args.na_file,
            'cl': args.cl_file
        }
        
        for ion_name, file_path in ion_file_configs.items():
            if file_path and os.path.exists(file_path):
                ion_files[ion_name] = file_path
                print(f"找到{ion_name}文件: {file_path}")
            elif file_path:
                # Na和Cl离子可能不存在于某些体系中，静默跳过
                if ion_name in ['na', 'cl']:
                    print(f"注意: {ion_name}文件不存在，跳过该离子分析 (某些体系不包含此离子)")
                else:
                    print(f"警告: {ion_name}文件 {file_path} 不存在，将跳过该离子分析")
        
        if not ion_files:
            print("警告: 没有找到任何有效的离子文件，将仅计算气泡质心")
    else:
        print("离子分析已禁用，仅计算气泡质心")
    
    # 创建计算器
    calculator = BubbleCentroidCalculator(
        cutoff_distance=args.cutoff
    )
    
    # 使用数据文件
    data_file = args.data
    
    try:
        # 处理轨迹
        times, centroids, sizes = calculator.process_trajectory(
            data_file=data_file,
            traj_file=args.traj_file,
            atom_style=args.atom_style,
            output_file=args.output,
            step_interval=args.step_interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            ion_files=ion_files if ion_files else None,
            ions_analysis_output=args.ions_output if ion_files else None
        )
        
        print("\n" + "="*50)
        print("计算完成!")
        print(f"处理了 {len(times)} 帧")
        if times:
            print(f"时间范围: {min(times):.1f} - {max(times):.1f} ps")
            print(f"平均气泡大小: {np.mean(sizes):.1f} 个氮原子")
        print(f"气泡质心结果保存到: {args.output}")
        
        if ion_files:
            print(f"离子分析结果保存到: {args.ions_output}/")
            print("  - raw_[ion_type]_distances.txt: 各离子类型的原始距离数据")
            print("  - all_ions_bubble_distance_distributions.png: 所有离子距离分布图")
            print("  - all_ions_distance_distribution_data.txt: 分布统计数据")
            print(f"  分析的离子类型: {', '.join(ion_files.keys())}")
        
        print("="*50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 