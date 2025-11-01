#!/usr/bin/env python3
"""
物质转移矩阵分析脚本

功能：
1. 读取多个时间段的xyz轨迹数据（solution_bulk_oh、solution_surface_oh、solution_surface_h2o）
2. 提取原子index信息进行物质转移追踪
3. 计算转移矩阵（基于O原子index）
4. 可视化转移矩阵（计数和概率）
5. 特别分析：solution_surface_oh → solution_surface_h2o时，H2O的OHH三个index是否连续
   - 连续：质子转移（来回跳动）
   - 不连续：bulk_h2o转移H给surface_oh
6. 保存所有数据和统计信息
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import glob
import re
from collections import defaultdict
import argparse

def setup_nature_style():
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

class TransitionAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.time_segments = []
        
        # 存储每个时间段的分子数据：{segment_key: {frame: [molecules]}}
        # molecule: {'coords': [(x,y,z), ...], 'symbols': ['O', 'H', ...], 'indices': [idx1, idx2, ...]}
        self.bulk_oh_data = {}
        self.surface_oh_data = {}
        self.surface_h2o_data = {}
        
        # 找到所有时间段
        self._find_time_segments()
        
    def _find_time_segments(self):
        """找到所有时间段目录"""
        pattern = os.path.join(self.base_path, "*ns")
        dirs = glob.glob(pattern)
        
        # 按时间顺序排序
        time_pattern = r'(\d+\.?\d*)-(\d+\.?\d*)ns'
        segments = []
        for d in dirs:
            dir_name = os.path.basename(d)
            match = re.search(time_pattern, dir_name)
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                # 检查是否有find_ion_4_matrix目录
                matrix_dir = os.path.join(d, "find_ion_4_matrix", "ion_analysis_results")
                if os.path.exists(matrix_dir):
                    segments.append((start_time, end_time, d))
        
        segments.sort(key=lambda x: x[0])
        self.time_segments = segments
        
        print(f"找到 {len(self.time_segments)} 个时间段:")
        for start, end, path in self.time_segments:
            print(f"  {start}-{end}ns: {os.path.basename(path)}")
    
    def parse_xyz_with_indices(self, xyz_file):
        """
        读取xyz文件并提取分子信息（包括atom indices）
        
        返回: {frame: [molecules]}
        molecule: {'coords': [(x,y,z), ...], 'symbols': ['O', 'H', ...], 'indices': [idx1, idx2, ...]}
        """
        frame_data = {}
        
        if not os.path.exists(xyz_file):
            print(f"  警告: 文件不存在: {xyz_file}")
            return frame_data
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # 读取原子数
                if lines[i].strip() and lines[i].strip().isdigit():
                    n_atoms = int(lines[i].strip())
                    i += 1
                    
                    if i >= len(lines):
                        break
                    
                    # 读取帧信息
                    frame_line = lines[i].strip()
                    frame_match = re.search(r'Frame[=\s]+(\d+)', frame_line)
                    if frame_match:
                        frame = int(frame_match.group(1))
                        i += 1
                        
                        # 检查是否有atom_index列
                        has_indices = 'atom_index' in frame_line.lower()
                        
                        # 读取原子数据
                        atoms = []
                        for j in range(n_atoms):
                            if i + j < len(lines):
                                parts = lines[i + j].strip().split()
                                if len(parts) >= 4:
                                    element = parts[0]
                                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                    atom_idx = int(parts[4]) if len(parts) >= 5 and has_indices else None
                                    atoms.append({'symbol': element, 'coord': (x, y, z), 'index': atom_idx})
                        
                        # 解析分子
                        molecules = self._group_atoms_into_molecules(atoms)
                        frame_data[frame] = molecules
                        
                        i += n_atoms
                    else:
                        i += 1
                else:
                    i += 1
        
        except Exception as e:
            print(f"  错误: 读取文件失败 {xyz_file}: {e}")
        
        return frame_data
    
    def _group_atoms_into_molecules(self, atoms):
        """将原子列表分组为分子"""
        molecules = []
        
        # 确定分子大小
        if not atoms:
            return molecules
        
        # 统计O和H的数量
        o_count = sum(1 for atom in atoms if atom['symbol'] == 'O')
        h_count = sum(1 for atom in atoms if atom['symbol'] == 'H')
        
        # 确定每个分子的原子数
        if o_count > 0:
            atoms_per_mol = len(atoms) // o_count
        else:
            return molecules
        
        # 按分子分组
        for i in range(0, len(atoms), atoms_per_mol):
            if i + atoms_per_mol <= len(atoms):
                mol_atoms = atoms[i:i+atoms_per_mol]
                
                # 验证分子
                symbols = [a['symbol'] for a in mol_atoms]
                coords = [a['coord'] for a in mol_atoms]
                indices = [a['index'] for a in mol_atoms if a['index'] is not None]
                
                # 验证分子完整性
                if atoms_per_mol == 2:  # OH
                    if symbols == ['O', 'H'] and len(indices) == 2:
                        molecules.append({
                            'symbols': symbols,
                            'coords': coords,
                            'indices': indices
                        })
                elif atoms_per_mol == 3:  # H2O
                    if symbols == ['O', 'H', 'H'] and len(indices) == 3:
                        molecules.append({
                            'symbols': symbols,
                            'coords': coords,
                            'indices': indices
                        })
        
        return molecules
    
    def load_all_data(self):
        """加载所有时间段的数据"""
        print("\n开始加载轨迹数据...")
        
        for start_time, end_time, segment_path in self.time_segments:
            segment_key = f"{start_time}-{end_time}ns"
            print(f"\n加载时间段: {segment_key}")
            
            results_dir = os.path.join(segment_path, "find_ion_4_matrix", "ion_analysis_results")
            
            # 加载bulk OH
            bulk_oh_file = os.path.join(results_dir, "solution_bulk_oh.xyz")
            print(f"  读取 solution_bulk_oh.xyz...")
            self.bulk_oh_data[segment_key] = self.parse_xyz_with_indices(bulk_oh_file)
            print(f"    加载了 {len(self.bulk_oh_data[segment_key])} 帧数据")
            
            # 加载surface OH
            surface_oh_file = os.path.join(results_dir, "solution_surface_oh.xyz")
            print(f"  读取 solution_surface_oh.xyz...")
            self.surface_oh_data[segment_key] = self.parse_xyz_with_indices(surface_oh_file)
            print(f"    加载了 {len(self.surface_oh_data[segment_key])} 帧数据")
            
            # 加载surface H2O
            surface_h2o_file = os.path.join(results_dir, "solution_surface_h2o.xyz")
            print(f"  读取 solution_surface_h2o.xyz...")
            self.surface_h2o_data[segment_key] = self.parse_xyz_with_indices(surface_h2o_file)
            print(f"    加载了 {len(self.surface_h2o_data[segment_key])} 帧数据")
        
        print("\n数据加载完成!")
    
    def build_unified_frame_sequence(self):
        """
        构建统一的帧序列，合并所有时间段
        
        返回: [(segment_key, frame), ...] 按时间顺序排列
        """
        frame_sequence = []
        
        for start_time, end_time, segment_path in self.time_segments:
            segment_key = f"{start_time}-{end_time}ns"
            
            # 获取该时间段的所有帧
            frames = set()
            if segment_key in self.bulk_oh_data:
                frames.update(self.bulk_oh_data[segment_key].keys())
            if segment_key in self.surface_oh_data:
                frames.update(self.surface_oh_data[segment_key].keys())
            if segment_key in self.surface_h2o_data:
                frames.update(self.surface_h2o_data[segment_key].keys())
            
            # 添加到序列
            for frame in sorted(frames):
                frame_sequence.append((segment_key, frame))
        
        return frame_sequence
    
    def build_transition_matrix(self):
        """
        构建物质转移矩阵
        
        返回:
        - transition_counts: 转移计数矩阵
        - species_names: 物质名称列表
        - transition_details: 详细转移信息
        - oh_to_h2o_analysis: OH→H2O转换的详细分析
        """
        species_names = ['solution_bulk_oh', 'solution_surface_oh', 'solution_surface_h2o']
        n_species = len(species_names)
        
        # 初始化转移计数矩阵
        transition_counts = np.zeros((n_species, n_species), dtype=int)
        
        # 存储详细转移信息
        transition_details = []
        
        # OH → H2O 转换分析
        oh_to_h2o_transitions = {
            'consecutive_indices': 0,  # index连续（质子转移）
            'non_consecutive_indices': 0,  # index不连续（bulk H2O转移H）
            'details': []
        }
        
        # 获取统一的帧序列
        frame_sequence = self.build_unified_frame_sequence()
        
        print(f"\n开始计算转移矩阵...")
        print(f"总帧数: {len(frame_sequence)}")
        
        # 遍历相邻帧
        for idx in range(len(frame_sequence) - 1):
            current_segment, current_frame = frame_sequence[idx]
            next_segment, next_frame = frame_sequence[idx + 1]
            
            # 构建当前帧的O原子index到物质类型的映射
            current_o_to_species = {}
            current_o_to_mol = {}  # O原子index到完整分子信息的映射
            
            # Bulk OH
            if current_segment in self.bulk_oh_data and current_frame in self.bulk_oh_data[current_segment]:
                for mol in self.bulk_oh_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]  # O原子总是第一个
                        current_o_to_species[o_idx] = 0
                        current_o_to_mol[o_idx] = mol
            
            # Surface OH
            if current_segment in self.surface_oh_data and current_frame in self.surface_oh_data[current_segment]:
                for mol in self.surface_oh_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        current_o_to_species[o_idx] = 1
                        current_o_to_mol[o_idx] = mol
            
            # Surface H2O
            if current_segment in self.surface_h2o_data and current_frame in self.surface_h2o_data[current_segment]:
                for mol in self.surface_h2o_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        current_o_to_species[o_idx] = 2
                        current_o_to_mol[o_idx] = mol
            
            # 构建下一帧的映射
            next_o_to_species = {}
            next_o_to_mol = {}
            
            # Bulk OH
            if next_segment in self.bulk_oh_data and next_frame in self.bulk_oh_data[next_segment]:
                for mol in self.bulk_oh_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 0
                        next_o_to_mol[o_idx] = mol
            
            # Surface OH
            if next_segment in self.surface_oh_data and next_frame in self.surface_oh_data[next_segment]:
                for mol in self.surface_oh_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 1
                        next_o_to_mol[o_idx] = mol
            
            # Surface H2O
            if next_segment in self.surface_h2o_data and next_frame in self.surface_h2o_data[next_segment]:
                for mol in self.surface_h2o_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 2
                        next_o_to_mol[o_idx] = mol
            
            # 统计转移
            for o_idx, from_species in current_o_to_species.items():
                if o_idx in next_o_to_species:
                    to_species = next_o_to_species[o_idx]
                    transition_counts[from_species, to_species] += 1
                    
                    # 记录详细转移信息
                    if from_species != to_species:
                        transition_info = {
                            'segment': current_segment,
                            'frame': current_frame,
                            'o_index': o_idx,
                            'from': species_names[from_species],
                            'to': species_names[to_species]
                        }
                        transition_details.append(transition_info)
                        
                        # 特别分析：surface OH → surface H2O
                        if from_species == 1 and to_species == 2:  # surface_oh → surface_h2o
                            # 检查H2O的三个index是否连续
                            h2o_mol = next_o_to_mol[o_idx]
                            h2o_indices = sorted(h2o_mol['indices'])
                            
                            # 判断是否连续
                            is_consecutive = (h2o_indices[1] == h2o_indices[0] + 1 and 
                                            h2o_indices[2] == h2o_indices[1] + 1)
                            
                            if is_consecutive:
                                oh_to_h2o_transitions['consecutive_indices'] += 1
                            else:
                                oh_to_h2o_transitions['non_consecutive_indices'] += 1
                            
                            oh_to_h2o_transitions['details'].append({
                                'segment': current_segment,
                                'frame': current_frame,
                                'o_index': o_idx,
                                'oh_indices': current_o_to_mol[o_idx]['indices'],
                                'h2o_indices': h2o_indices,
                                'is_consecutive': is_consecutive
                            })
        
        print(f"转移矩阵计算完成!")
        print(f"总转移事件: {len(transition_details)}")
        print(f"surface_oh → surface_h2o 转移事件: {oh_to_h2o_transitions['consecutive_indices'] + oh_to_h2o_transitions['non_consecutive_indices']}")
        
        return transition_counts, species_names, transition_details, oh_to_h2o_transitions
    
    def plot_transition_matrix(self, transition_counts, species_names, output_file):
        """绘制转移矩阵"""
        setup_nature_style()
        
        # 计算转移概率矩阵
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 简化物质名称用于显示
        display_names = [
            r'Bulk OH$^-$',
            r'Surface OH$^-$',
            r'Surface H$_2$O'
        ]
        
        # 绘制转移计数矩阵
        im1 = ax1.imshow(transition_counts, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(species_names)))
        ax1.set_yticks(range(len(species_names)))
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.set_yticklabels(display_names)
        ax1.set_xlabel('To Species')
        ax1.set_ylabel('From Species')
        ax1.set_title('Transition Counts')
        
        # 在每个格子中添加数值
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                text = ax1.text(j, i, int(transition_counts[i, j]),
                              ha="center", va="center", 
                              color="black" if transition_prob[i, j] < 0.5 else "white",
                              fontsize=14, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Counts', rotation=270, labelpad=20)
        
        # 绘制转移概率矩阵
        im2 = ax2.imshow(transition_prob, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(species_names)))
        ax2.set_yticks(range(len(species_names)))
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        ax2.set_yticklabels(display_names)
        ax2.set_xlabel('To Species')
        ax2.set_ylabel('From Species')
        ax2.set_title('Transition Probability')
        
        # 在每个格子中添加概率值
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                text = ax2.text(j, i, f'{transition_prob[i, j]:.3f}',
                              ha="center", va="center", 
                              color="black" if transition_prob[i, j] < 0.5 else "white",
                              fontsize=14, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Probability', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"转移矩阵图保存至: {output_file}")
    
    def plot_transition_probability_matrix_only(self, transition_counts, species_names, output_file):
        """单独绘制转移概率矩阵（优化版）"""
        setup_nature_style()
        
        # 计算转移概率矩阵
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums
        
        # 创建正方形图形
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # 改进的物质名称显示 - ion name (location)
        display_names = [
            r'OH$^-$ (bulk)',
            r'OH$^-$ (surf)',
            r'H$_2$O (surf)'
        ]
        
        # 使用白蓝渐变色
        im = ax.imshow(transition_prob, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(species_names)))
        ax.set_yticks(range(len(species_names)))
        ax.set_xticklabels(display_names, fontsize=14)
        #ax.set_yticklabels(display_names, fontsize=14)
        ax.set_yticklabels(display_names, fontsize=14, rotation=90,va='center', ha='center',x=-0.05)
        ax.set_xlabel('To Species', fontsize=15)
        ax.set_ylabel('From Species', fontsize=15, rotation=90)
        
        # 在每个格子中添加概率值
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                # 根据背景颜色自动调整文字颜色
                text_color = "white" if transition_prob[i, j] > 0.6 else "black"
                text = ax.text(j, i, f'{transition_prob[i, j]:.3f}',
                              ha="center", va="center", 
                              color=text_color,
                              fontsize=15)
        
        # Colorbar放在顶部，横向，向下移一点
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        #cbar.set_label('Transition Probability', fontsize=13, labelpad=12)
        cbar.ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"转移概率矩阵（单独）保存至: {output_file}")
    
    def plot_oh_to_h2o_analysis(self, oh_to_h2o_transitions, output_file):
        """绘制OH→H2O转换机制分析图"""
        setup_nature_style()
        
        total = (oh_to_h2o_transitions['consecutive_indices'] + 
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        if total == 0:
            print("警告: 没有surface_oh → surface_h2o的转移事件")
            return
        
        consecutive_ratio = oh_to_h2o_transitions['consecutive_indices'] / total
        non_consecutive_ratio = oh_to_h2o_transitions['non_consecutive_indices'] / total
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 饼图
        labels = [
            f'Proton Transfer\n(consecutive indices)\n{consecutive_ratio:.1%}',
            f'Bulk H$_2$O Transfer\n(non-consecutive indices)\n{non_consecutive_ratio:.1%}'
        ]
        sizes = [oh_to_h2o_transitions['consecutive_indices'],
                oh_to_h2o_transitions['non_consecutive_indices']]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax1.set_title(r'Surface OH$^-$ → Surface H$_2$O Mechanism', fontsize=14, fontweight='bold')
        
        # 柱状图
        categories = ['Proton\nTransfer', 'Bulk H$_2$O\nTransfer']
        counts = [oh_to_h2o_transitions['consecutive_indices'],
                 oh_to_h2o_transitions['non_consecutive_indices']]
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(r'Surface OH$^-$ → Surface H$_2$O Transition Counts', 
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"OH→H2O机制分析图保存至: {output_file}")
    
    def plot_oh_to_h2o_bar_only(self, oh_to_h2o_transitions, output_file):
        """单独绘制OH→H2O转换机制柱状图（优化版，百分比归一化）"""
        setup_nature_style()
        
        total = (oh_to_h2o_transitions['consecutive_indices'] + 
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        if total == 0:
            print("警告: 没有surface_oh → surface_h2o的转移事件")
            return
        
        consecutive_ratio = oh_to_h2o_transitions['consecutive_indices'] / total * 100
        non_consecutive_ratio = oh_to_h2o_transitions['non_consecutive_indices'] / total * 100
        
        # 创建窄一点的图形
        fig, ax = plt.subplots(figsize=(2, 4))
        
        # 改进的标签描述
        #categories = [
        #    'Surface H$^+$\nhopping',  # 表面质子跳跃
        #    'Proton exchange\nwith bulk H$_2$O'  # 与体相H2O质子交换
        #]
        categories = [
            'PT(surf)',  # 表面质子跳跃
            'PT(bulk)'  # 与体相H2O质子交换
        ]        
        
        percentages = [consecutive_ratio, non_consecutive_ratio]
        
        # Nature风格配色 - 使用更专业的颜色
        colors = ['#E64B35', '#4DBBD5']  # 红色和青色，对比鲜明
        
        bars = ax.bar(categories, percentages, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1, width=0.3)
        
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.set_ylim(0, 65)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        # 在柱子上添加百分比值
        #for bar, percentage in zip(bars, percentages):
        #    height = bar.get_height()
        #    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
        #            f'{percentage:.1f}%',
        #            ha='center', va='bottom', fontsize=15)
        
        # 调整x轴标签样式
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        
        # 移除顶部和右侧的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"OH→H2O机制柱状图（单独）保存至: {output_file}")
    
    def save_transition_data(self, transition_counts, species_names, transition_details, 
                            oh_to_h2o_transitions, output_dir):
        """保存转移矩阵数据和统计信息"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存转移计数矩阵
        counts_file = os.path.join(output_dir, "transition_counts.txt")
        with open(counts_file, 'w') as f:
            f.write("Transition Counts Matrix\n")
            f.write("From \\ To\t" + "\t".join(species_names) + "\n")
            for i, from_species in enumerate(species_names):
                f.write(f"{from_species}\t")
                f.write("\t".join(str(int(transition_counts[i, j])) for j in range(len(species_names))))
                f.write("\n")
        print(f"转移计数矩阵保存至: {counts_file}")
        
        # 2. 保存转移概率矩阵
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums
        
        prob_file = os.path.join(output_dir, "transition_probabilities.txt")
        with open(prob_file, 'w') as f:
            f.write("Transition Probability Matrix\n")
            f.write("From \\ To\t" + "\t".join(species_names) + "\n")
            for i, from_species in enumerate(species_names):
                f.write(f"{from_species}\t")
                f.write("\t".join(f"{transition_prob[i, j]:.6f}" for j in range(len(species_names))))
                f.write("\n")
        print(f"转移概率矩阵保存至: {prob_file}")
        
        # 3. 保存详细转移信息
        details_file = os.path.join(output_dir, "transition_details.txt")
        with open(details_file, 'w') as f:
            f.write("Segment\tFrame\tO_Index\tFrom_Species\tTo_Species\n")
            for detail in transition_details:
                f.write(f"{detail['segment']}\t{detail['frame']}\t{detail['o_index']}\t"
                       f"{detail['from']}\t{detail['to']}\n")
        print(f"详细转移信息保存至: {details_file}")
        
        # 4. 保存转移统计摘要
        summary_file = os.path.join(output_dir, "transition_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== Species Transition Summary ===\n\n")
            
            total_transitions = np.sum(transition_counts) - np.trace(transition_counts)
            f.write(f"Total transitions (excluding self-transitions): {int(total_transitions)}\n\n")
            
            for i, species in enumerate(species_names):
                f.write(f"\n{species}:\n")
                total_from = np.sum(transition_counts[i, :])
                total_to = np.sum(transition_counts[:, i])
                f.write(f"  Total transitions from this species: {int(total_from)}\n")
                f.write(f"  Total transitions to this species: {int(total_to)}\n")
                
                if total_from > 0:
                    f.write(f"  Main transitions from {species}:\n")
                    for j in range(len(species_names)):
                        if i != j and transition_counts[i, j] > 0:
                            prob = transition_prob[i, j]
                            f.write(f"    -> {species_names[j]}: {int(transition_counts[i, j])} ({prob:.2%})\n")
        print(f"转移统计摘要保存至: {summary_file}")
        
        # 5. 保存OH→H2O转换机制分析
        oh_h2o_file = os.path.join(output_dir, "oh_to_h2o_mechanism_analysis.txt")
        total = (oh_to_h2o_transitions['consecutive_indices'] + 
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        with open(oh_h2o_file, 'w') as f:
            f.write("=== Surface OH- → Surface H2O Mechanism Analysis ===\n\n")
            f.write(f"Total transitions: {total}\n\n")
            
            if total > 0:
                f.write("Mechanism 1: Proton Transfer (consecutive indices)\n")
                f.write(f"  Count: {oh_to_h2o_transitions['consecutive_indices']}\n")
                f.write(f"  Percentage: {oh_to_h2o_transitions['consecutive_indices']/total:.2%}\n")
                f.write("  Interpretation: OH- ⇌ H2O proton transfer (back-and-forth)\n\n")
                
                f.write("Mechanism 2: Bulk H2O H-transfer (non-consecutive indices)\n")
                f.write(f"  Count: {oh_to_h2o_transitions['non_consecutive_indices']}\n")
                f.write(f"  Percentage: {oh_to_h2o_transitions['non_consecutive_indices']/total:.2%}\n")
                f.write("  Interpretation: Bulk H2O transfers H to surface OH-\n\n")
                
                f.write("\n=== Detailed Events ===\n")
                f.write("Segment\tFrame\tO_Index\tOH_Indices\tH2O_Indices\tIs_Consecutive\tMechanism\n")
                for detail in oh_to_h2o_transitions['details']:
                    mechanism = "Proton Transfer" if detail['is_consecutive'] else "Bulk H2O Transfer"
                    f.write(f"{detail['segment']}\t{detail['frame']}\t{detail['o_index']}\t"
                           f"{detail['oh_indices']}\t{detail['h2o_indices']}\t"
                           f"{detail['is_consecutive']}\t{mechanism}\n")
        
        print(f"OH→H2O机制分析保存至: {oh_h2o_file}")

def main():
    parser = argparse.ArgumentParser(description='分析物质转移矩阵')
    parser.add_argument('--base_path', type=str, 
                       default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4',
                       help='基础路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4/analysis_ion/find_ion_4_matrix/results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("="*80)
    print("物质转移矩阵分析")
    print("="*80)
    
    # 创建分析器
    analyzer = TransitionAnalyzer(args.base_path)
    
    # 加载数据
    analyzer.load_all_data()
    
    # 计算转移矩阵
    transition_counts, species_names, transition_details, oh_to_h2o_transitions = \
        analyzer.build_transition_matrix()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制转移矩阵（完整版）
    matrix_plot = os.path.join(args.output_dir, "transition_matrix.png")
    analyzer.plot_transition_matrix(transition_counts, species_names, matrix_plot)
    
    # 单独绘制转移概率矩阵（优化版）
    prob_matrix_only = os.path.join(args.output_dir, "transition_probability_matrix.png")
    analyzer.plot_transition_probability_matrix_only(transition_counts, species_names, prob_matrix_only)
    
    # 绘制OH→H2O机制分析（完整版）
    oh_h2o_plot = os.path.join(args.output_dir, "oh_to_h2o_mechanism.png")
    analyzer.plot_oh_to_h2o_analysis(oh_to_h2o_transitions, oh_h2o_plot)
    
    # 单独绘制OH→H2O机制柱状图（优化版）
    oh_h2o_bar_only = os.path.join(args.output_dir, "oh_to_h2o_bar.png")
    analyzer.plot_oh_to_h2o_bar_only(oh_to_h2o_transitions, oh_h2o_bar_only)
    
    # 保存数据
    analyzer.save_transition_data(transition_counts, species_names, transition_details,
                                  oh_to_h2o_transitions, args.output_dir)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()

