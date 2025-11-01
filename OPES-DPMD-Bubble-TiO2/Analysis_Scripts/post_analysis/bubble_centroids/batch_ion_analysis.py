#!/usr/bin/env python3
"""
批量分析离子距离分布的脚本
读取多个时间段的离子距离数据，生成综合分布图、时间分组分布图和电荷分布图
支持命令行参数指定输入路径和输出目录
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from collections import defaultdict
import re
import glob
import argparse

class BatchIonAnalyzer:
    """批量离子距离分析器"""
    
    def __init__(self, base_dir="/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4", output_dir=None):
        self.base_dir = base_dir
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = os.path.join(base_dir, "analysis_ion/centroids_density")
        else:
            # 如果是相对路径，相对于base_dir；如果是绝对路径，直接使用
            if os.path.isabs(output_dir):
                self.output_dir = output_dir
            else:
                self.output_dir = os.path.join(base_dir, output_dir)
        
        # 定义离子样式
        self.ion_styles = {
            'H3O': {'color': '#1f77b4', 'marker': 'o', 'label': r'$\mathrm{H_3O^+}$'},
            'bulk_OH': {'color': '#ff7f0e', 'marker': 's', 'label': r'$\mathrm{OH^-(bulk)}$'},
            'surface_OH': {'color': '#2ca02c', 'marker': '^', 'label': r'$\mathrm{OH^-(surf)}$'},
            'surface_H': {'color': '#d62728', 'marker': 'v', 'label': r'$\mathrm{H^+(surf)}$'},
            'Na': {'color': '#9467bd', 'marker': 'D', 'label': r'$\mathrm{Na^+}$'},
            'Cl': {'color': '#8c564b', 'marker': 'p', 'label': r'$\mathrm{Cl^-}$'}
        }
        
        # 定义离子电荷
        self.ion_charges = {
            'H3O': 1,
            'bulk_OH': -1,
            'surface_OH': -1,
            'surface_H': 1,
            'Na': 1,
            'Cl': -1
        }
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
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
    
    def find_time_directories(self):
        """查找所有时间段目录并按时间顺序排序"""
        pattern = re.compile(r'(\d+\.?\d*)-(\d+\.?\d*)ns')
        time_dirs = []
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and pattern.match(item):
                match = pattern.match(item)
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                time_dirs.append((start_time, end_time, item, item_path))
        
        # 按开始时间排序
        time_dirs.sort(key=lambda x: x[0])
        
        print(f"找到 {len(time_dirs)} 个时间段目录:")
        total_time = 0
        for start, end, dirname, _ in time_dirs:
            duration = end - start
            total_time += duration
            print(f"  {dirname}: {start}-{end}ns (时长: {duration}ns)")
        print(f"总模拟时间: {total_time}ns")
        
        return time_dirs, total_time
    
    def read_ion_distances(self, time_dirs, centroids_subpath="centroids_density_2"):
        """读取所有时间段的离子距离数据"""
        all_ion_data = defaultdict(list)  # {ion_type: [(d_centroid, d_interface, time_period), ...]}
        time_grouped_data = defaultdict(lambda: defaultdict(list))  # {time_period: {ion_type: [(d_centroid, d_interface), ...]}}
        
        for i, (start_time, end_time, dirname, dir_path) in enumerate(time_dirs):
            ions_analysis_dir = os.path.join(dir_path, f"{centroids_subpath}/ions_analysis")
            
            if not os.path.exists(ions_analysis_dir):
                print(f"警告: {ions_analysis_dir} 不存在，跳过")
                continue
            
            print(f"处理时间段 {i+1}/{len(time_dirs)}: {dirname}")
            
            # 查找所有原始距离文件
            distance_files = glob.glob(os.path.join(ions_analysis_dir, "raw_*_distances.txt"))
            
            for file_path in distance_files:
                # 从文件名提取离子类型
                filename = os.path.basename(file_path)
                ion_match = re.search(r'raw_(.+)_distances\.txt', filename)
                if not ion_match:
                    continue
                
                ion_type = ion_match.group(1)
                
                # 读取数据
                try:
                    # 首先检查文件是否有有效数据
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # 跳过头部，检查是否有数据行
                    data_lines = [line.strip() for line in lines[3:] if line.strip() and not line.startswith('#')]
                    
                    if not data_lines:
                        print(f"    跳过 {ion_type}: 文件中无数据（可能该体系不包含此离子）")
                        continue
                    
                    # 读取数值数据
                    data = np.loadtxt(file_path, skiprows=3, dtype=float)
                    if data.size == 0:
                        print(f"    跳过 {ion_type}: 无有效数值数据")
                        continue
                    
                    if len(data.shape) == 1:
                        data = data.reshape(1, -1)
                    
                    print(f"    成功读取 {ion_type}: {len(data)} 个数据点")
                    
                    # 添加到总数据
                    for row in data:
                        # 根据新的定义，d_interface是离子到最近的N原子的距离。
                        # 本脚本假定上游数据生成过程已经更新，输入文件中的第二列即为这个新定义的值。
                        d_centroid, d_interface = row[0], row[1]
                        all_ion_data[ion_type].append((d_centroid, d_interface, i))
                        time_grouped_data[i][ion_type].append((d_centroid, d_interface))
                
                except Exception as e:
                    print(f"    读取 {ion_type} 时出错: {e}")
                    # 如果是Na或Cl离子，给出特殊提示
                    if ion_type in ['Na', 'Cl']:
                        print(f"      注意：某些体系可能不包含{ion_type}离子")
        
        # 输出实际找到的离子类型
        found_ions = list(all_ion_data.keys())
        expected_ions = ['H3O', 'bulk_OH', 'surface_OH', 'surface_H', 'Na', 'Cl']
        missing_ions = [ion for ion in expected_ions if ion not in found_ions]
        
        print(f"\n数据读取总结:")
        print(f"  成功读取的离子类型: {', '.join(found_ions) if found_ions else '无'}")
        if missing_ions:
            print(f"  未找到数据的离子类型: {', '.join(missing_ions)}")
            if 'Na' in missing_ions or 'Cl' in missing_ions:
                print(f"  注意：此体系可能不包含NaCl离子")
        
        return all_ion_data, time_grouped_data, len(time_dirs)
    
    def read_bubble_centroids(self, time_dirs, centroids_subpath="centroids_density_2"):
        """读取所有时间段的气泡质心数据，获取BubbleSize时间演化"""
        all_bubble_data = []  # [(frame_index, time_ns, bubble_size, time_period_idx), ...]
        
        for i, (start_time, end_time, dirname, dir_path) in enumerate(time_dirs):
            centroids_file = os.path.join(dir_path, f"{centroids_subpath}/bubble_centroids.txt")
            
            if not os.path.exists(centroids_file):
                print(f"警告: {centroids_file} 不存在，跳过")
                continue
            
            print(f"读取气泡数据 {i+1}/{len(time_dirs)}: {dirname}")
            
            try:
                # 读取bubble_centroids.txt文件
                data = np.loadtxt(centroids_file, skiprows=1)  # 跳过头部注释
                
                if data.size == 0:
                    print(f"    警告: {centroids_file} 无数据")
                    continue
                
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                
                # 提取帧索引、时间和气泡大小数据
                frame_indices = data[:, 0]  # FrameIndex列
                times_ps_column = data[:, 1]  # Time(ps)列（实际是帧数×1000）
                bubble_sizes = data[:, 5]  # BubbleSize列
                
                # 计算真实时间：每个时间段的起始时间 + 帧数×1ps
                # 每个子文件夹的帧数从1开始，1ps per frame
                times_ns = start_time + (frame_indices - 1) / 1000.0  # 转换为ns
                
                print(f"    成功读取: {len(frame_indices)} 个时间点")
                print(f"    帧数范围: {frame_indices[0]:.0f} - {frame_indices[-1]:.0f}")
                print(f"    时间范围: {times_ns[0]:.3f} - {times_ns[-1]:.3f} ns")
                print(f"    气泡大小范围: {bubble_sizes[0]/2:.0f} - {bubble_sizes[-1]/2:.0f} 个N2分子")
                
                # 添加到总数据，记录时间段索引和绝对帧数
                for frame_idx, time_ns, bubble_size in zip(frame_indices, times_ns, bubble_sizes):
                    # 计算绝对帧数（跨所有时间段的连续帧数）
                    absolute_frame = int(time_ns * 1000)  # 时间(ns) * 1000 = 绝对帧数
                    all_bubble_data.append((absolute_frame, time_ns, bubble_size, i))
            
            except Exception as e:
                print(f"    读取 {centroids_file} 时出错: {e}")
        
        if not all_bubble_data:
            print("错误: 没有读取到任何气泡数据")
            return None
        
        # 按时间排序
        all_bubble_data.sort(key=lambda x: x[1])  # 按时间ns排序
        
        print(f"\n气泡数据读取总结:")
        print(f"  总时间点数: {len(all_bubble_data)}")
        print(f"  时间范围: {all_bubble_data[0][1]:.3f} - {all_bubble_data[-1][1]:.3f} ns")
        print(f"  帧数范围: {all_bubble_data[0][0]} - {all_bubble_data[-1][0]}")
        print(f"  初始气泡大小: {all_bubble_data[0][2]/2:.0f} 个$\\mathrm{{N_2}}$分子")
        print(f"  最终气泡大小: {all_bubble_data[-1][2]/2:.0f} 个$\\mathrm{{N_2}}$分子")
        
        # 按时间段统计
        period_stats = defaultdict(int)
        for _, _, _, period_idx in all_bubble_data:
            period_stats[period_idx] += 1
        
        print(f"  各时间段数据点数:")
        for period_idx in sorted(period_stats.keys()):
            if period_idx < len(time_dirs):
                dirname = time_dirs[period_idx][2]
                print(f"    {dirname}: {period_stats[period_idx]} 个数据点")
        
        return all_bubble_data

    def save_plot_data(self, filename, headers, data_dict):
        """
        保存绘图数据到文本文件
        :param filename: 输出文件名
        :param headers: 文件头注释信息 (list of strings)
        :param data_dict: 包含绘图数据的字典, e.g., {'x_axis_label': x_data, 'y1_label': y1_data, ...}
        """
        try:
            with open(filename, 'w') as f:
                for header in headers:
                    f.write(f"# {header}\n")
                
                column_labels = list(data_dict.keys())
                f.write("# " + "\t".join(column_labels) + "\n")
                
                # 获取数据列并转置
                columns = list(data_dict.values())
                if not all(isinstance(c, np.ndarray) for c in columns):
                     print(f"警告: {filename} 中并非所有数据列都是numpy数组，跳过保存。")
                     return

                # 确保所有列长度相同，以最短的为准
                min_len = min(len(col) for col in columns if col is not None)
                
                for i in range(min_len):
                    line_parts = []
                    for col in columns:
                        if col is not None and i < len(col):
                            line_parts.append(f"{col[i]:.6e}")
                        else:
                            line_parts.append("NaN")
                    f.write("\t".join(line_parts) + "\n")
            print(f"绘图数据已保存: {filename}")
        except Exception as e:
            print(f"保存绘图数据到 {filename} 时出错: {e}")

    def plot_bubble_evolution(self, bubble_data, args):
        """绘制气泡大小随时间的演化曲线"""
        if not bubble_data:
            print("没有气泡数据可绘制演化曲线")
            return None
        
        self.setup_nature_style()
        
        # 提取数据
        frame_indices = [item[0] for item in bubble_data]
        times_ns = [item[1] for item in bubble_data]
        bubble_sizes = [item[2] / 2 for item in bubble_data]  # 除以2转换为分子数
        
        # 计算百分比（相对于初始值）
        initial_size = bubble_sizes[0]
        bubble_percentages = [(size / initial_size) * 100 for size in bubble_sizes]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：绝对大小
        ax1.plot(times_ns, bubble_sizes, 'b-', linewidth=2, label=r'$\mathrm{N_2}$ molecules')
        ax1.set_xlabel('Time (ns)', fontsize=14)
        ax1.set_ylabel(r'Bubble Size ($\mathrm{N_2}$ molecules)', fontsize=14)
        ax1.set_title('Bubble Size Evolution', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=12)
        
        # 下图：百分比
        ax2.plot(times_ns, bubble_percentages, 'r-', linewidth=2, label='Bubble Size (%)')
        ax2.set_xlabel('Time (ns)', fontsize=14)
        ax2.set_ylabel('Bubble Size (%)', fontsize=14)
        ax2.set_title('Bubble Size Evolution (Relative to Initial)', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.output_dir, "bubble_size_evolution.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"气泡演化曲线图已保存: {plot_file}")
        
        # 保存演化数据
        data_file = os.path.join(self.output_dir, "data_bubble_evolution.txt")
        with open(data_file, 'w') as f:
            f.write("# Bubble size evolution data\n")
            f.write("# Format: FrameIndex Time(ns) BubbleSize(N2_molecules) Percentage(%) TimePeriod\n")
            f.write("# Note: Time = FrameIndex / 1000.0 (1ps per frame converted to ns)\n")
            for item in bubble_data:
                frame_idx, time_ns, bubble_size_raw, period_idx = item
                bubble_size_molecules = bubble_size_raw / 2  # 除以2转换为分子数
                percentage = (bubble_size_molecules / initial_size) * 100
                f.write(f"{frame_idx}\t{time_ns:.3f}\t{bubble_size_molecules:.0f}\t{percentage:.2f}\t{period_idx}\n")
        
        print(f"气泡演化数据已保存: {data_file}")
        
        return bubble_data

    def plot_timerange_ion_distributions(self, all_ion_data, time_range_ns, args):
        """绘制指定时间范围内的离子距离分布图"""
        print(f"生成 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围的离子分布图...")
        
        filtered_ion_data = defaultdict(list)
        
        # 建立时间段索引到实际时间范围的映射
        if not hasattr(self, '_time_dirs'):
            print("错误: time_dirs 未初始化，无法按时间范围筛选")
            return
            
        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    # 检查时间段是否与指定范围有重叠
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))
        
        if not filtered_ion_data:
            print(f"警告: 在 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内没有找到离子数据")
            return
        
        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        total_selected = sum(len(data) for data in filtered_ion_data.values())
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内的离子数据点: {total_selected}")
        
        all_centroids_filtered = []
        for ion_data in filtered_ion_data.values():
            if ion_data:
                all_centroids_filtered.extend([item[0] for item in ion_data])
        avg_bubble_radius = np.mean(all_centroids_filtered) if all_centroids_filtered else 10.0

        # Data storage
        data_centroid_prob, data_interface_prob = {}, {}
        data_centroid_vol, data_interface_vol = {}, {}

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            print(f"  {ion_type}: {len(data_list)} 个数据点")
            
            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]
            
            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')
            
            # === 密度归一化分布图 ===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
            
            ax1.plot(bin_centers_centroid, hist_centroid_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
            
            ax2.plot(bin_centers_interface, hist_interface_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === 体积归一化分布图 ===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000  # %/nm³
            
            ax3.plot(bin_centers_centroid, hist_centroid_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000  # %/nm³
            
            ax4.plot(bin_centers_interface, hist_interface_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_prob:
                data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_prob:
                data_interface_prob['d_interface(A)'] = bin_centers_interface
                data_interface_vol['d_interface(A)'] = bin_centers_interface

            data_centroid_prob[f'Prob_{ion_label_safe}(%)'] = hist_centroid_norm
            data_interface_prob[f'Prob_{ion_label_safe}(%)'] = hist_interface_norm
            data_centroid_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_centroid_vol
            data_interface_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_interface_vol

        # 设置子图
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Probability (%)', fontsize=14)
        ax1.set_title(f'Ion Distance to Bubble Centroid\n(Density Normalized, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Probability (%)', fontsize=14)
        ax2.set_title(f'Ion Distance to Bubble Surface\n(Density Normalized, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Distance to Bubble Surface\n(Volume Normalized)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"ion_distributions_normalized_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 归一化离子分布图已保存: {plot_file}")

        # Save data to files
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_prob_{time_str}.txt"), [f"Ion dist to centroid (Density Norm) for {time_str}"], data_centroid_prob)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_prob_{time_str}.txt"), [f"Ion dist to interface (Density Norm) for {time_str}"], data_interface_prob)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_vol_{time_str}.txt"), [f"Ion dist to centroid (Volume Norm) for {time_str}"], data_centroid_vol)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_vol_{time_str}.txt"), [f"Ion dist to interface (Volume Norm) for {time_str}"], data_interface_vol)

    def plot_timerange_ion_distributions_absolute(self, all_ion_data, time_range_ns, args):
        """绘制指定时间范围内的离子距离分布图（绝对数量，非归一化版本）"""
        print(f"生成 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围的离子分布图（绝对数量）...")
        
        filtered_ion_data = defaultdict(list)
        
        # 建立时间段索引到实际时间范围的映射
        if not hasattr(self, '_time_dirs'):
            print("错误: time_dirs 未初始化，无法按时间范围筛选")
            return
            
        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    # 检查时间段是否与指定范围有重叠
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))
        
        if not filtered_ion_data:
            print(f"警告: 在 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内没有找到离子数据")
            return
        
        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        total_selected = sum(len(data) for data in filtered_ion_data.values())
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内的离子数据点: {total_selected}")
        
        # 计算时间范围内的总帧数（假设1ps per frame）
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # 检查时间段是否与指定范围有重叠
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # 计算重叠部分的时长
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1ps per frame
                total_frames_in_range += frames_in_overlap
        
        print(f"时间范围内估计帧数: {total_frames_in_range}")
        
        all_centroids_filtered = []
        for ion_data in filtered_ion_data.values():
            if ion_data:
                all_centroids_filtered.extend([item[0] for item in ion_data])
        avg_bubble_radius = np.mean(all_centroids_filtered) if all_centroids_filtered else 10.0

        # Data storage
        data_centroid_num, data_interface_num = {}, {}
        data_centroid_density, data_interface_density = {}, {}

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            print(f"  {ion_type}: {len(data_list)} 个数据点")
            
            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]
            
            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')
            
            # === 绝对数量分布图（除以帧数得到每帧平均数量）===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            # 除以帧数得到每帧平均数量
            hist_centroid_per_frame = hist_centroid / max(total_frames_in_range, 1)
            
            ax1.plot(bin_centers_centroid, hist_centroid_per_frame,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            # 除以帧数得到每帧平均数量
            hist_interface_per_frame = hist_interface / max(total_frames_in_range, 1)
            
            ax2.plot(bin_centers_interface, hist_interface_per_frame,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === 密度分布图（每帧平均数量/体积）===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_density = hist_centroid_per_frame / shell_volumes_centroid * 1000  # num per frame/nm³
            
            ax3.plot(bin_centers_centroid, hist_centroid_density,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_density = hist_interface_per_frame / shell_volumes_interface * 1000  # num per frame/nm³
            
            ax4.plot(bin_centers_interface, hist_interface_density,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_num:
                data_centroid_num['d_centroid(A)'] = bin_centers_centroid
                data_centroid_density['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_num:
                data_interface_num['d_interface(A)'] = bin_centers_interface
                data_interface_density['d_interface(A)'] = bin_centers_interface

            data_centroid_num[f'Count_{ion_label_safe}(num)'] = hist_centroid
            data_interface_num[f'Count_{ion_label_safe}(num)'] = hist_interface
            data_centroid_density[f'Density_{ion_label_safe}(num/nm^3)'] = hist_centroid_density
            data_interface_density[f'Density_{ion_label_safe}(num/nm^3)'] = hist_interface_density

        # 设置子图
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.set_title(f'Ion Count vs Distance to Bubble Centroid\n(Absolute Count, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2.set_title(f'Ion Count vs Distance to Bubble Surface\n(Absolute Count, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Ion Density (num / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Density vs Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Ion Density (num / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Density vs Distance to Bubble Surface\n(Volume Normalized)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"ion_distributions_absolute_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 绝对数量离子分布图已保存: {plot_file}")

        # Save data to files
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_absolute_{time_str}.txt"), [f"Ion dist to centroid (Absolute Count) for {time_str}"], data_centroid_num)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_absolute_{time_str}.txt"), [f"Ion dist to interface (Absolute Count) for {time_str}"], data_interface_num)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_density_{time_str}.txt"), [f"Ion dist to centroid (Density) for {time_str}"], data_centroid_density)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_density_{time_str}.txt"), [f"Ion dist to interface (Density) for {time_str}"], data_interface_density)

    def plot_comprehensive_distributions(self, all_ion_data, total_time, args):
        """绘制综合所有时间段的离子距离分布图"""
        if not all_ion_data:
            print("没有离子数据可绘制综合分布图")
            return
        
        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        all_centroids = []
        for ion_data in all_ion_data.values():
            if ion_data:
                all_centroids.extend([item[0] for item in ion_data])
        global_avg_bubble_radius = np.mean(all_centroids) if all_centroids else 10.0
        
        # Data storage
        data_centroid_prob, data_interface_prob = {}, {}
        data_centroid_vol, data_interface_vol = {}, {}

        for ion_type, data_list in all_ion_data.items():
            if not data_list:
                continue
            
            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]
            
            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')

            # === 密度归一化 ===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
            
            ax1.plot(bin_centers_centroid, hist_centroid_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
            
            ax2.plot(bin_centers_interface, hist_interface_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === 体积归一化 ===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000
            
            ax3.plot(bin_centers_centroid, hist_centroid_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = global_avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000
            
            ax4.plot(bin_centers_interface, hist_interface_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_prob:
                data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_prob:
                data_interface_prob['d_interface(A)'] = bin_centers_interface
                data_interface_vol['d_interface(A)'] = bin_centers_interface

            data_centroid_prob[f'Prob_{ion_label_safe}(%)'] = hist_centroid_norm
            data_interface_prob[f'Prob_{ion_label_safe}(%)'] = hist_interface_norm
            data_centroid_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_centroid_vol
            data_interface_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_interface_vol
        
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Probability (%)', fontsize=14)
        ax1.set_title(f'Ion Distance to Bubble Centroid\n(Density Normalized, Total time: {total_time:.1f}ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Probability (%)', fontsize=14)
        ax2.set_title(f'Ion Distance to Bubble Surface\n(Density Normalized, Total time: {total_time:.1f}ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Distance to Bubble Surface\n(Volume Normalized, Equiv. Sphere)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, "comprehensive_ion_distributions_normalized.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"归一化综合分布图已保存: {plot_file}")

        # Save data to files
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_centroid_prob.txt"), [f"Comprehensive ion dist to centroid (Density Norm)"], data_centroid_prob)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_interface_prob.txt"), [f"Comprehensive ion dist to interface (Density Norm)"], data_interface_prob)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_centroid_vol.txt"), [f"Comprehensive ion dist to centroid (Volume Norm)"], data_centroid_vol)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_interface_vol.txt"), [f"Comprehensive ion dist to interface (Volume Norm)"], data_interface_vol)

    def plot_time_grouped_distributions(self, time_grouped_data, num_time_periods, time_dirs, args):
        """绘制按时间段分组的离子距离分布图（4个时间段，颜色由浅到深）"""
        if not time_grouped_data:
            print("没有离子数据可绘制时间分组分布图")
            return
        
        self.setup_nature_style()
        
        group_size = max(1, num_time_periods // 4)
        groups = [list(range(i, min(i + group_size, num_time_periods))) for i in range(0, num_time_periods, group_size)]
        
        if len(groups) > 4 and len(groups[-1]) < group_size / 2:
            groups[-2].extend(groups.pop())
        
        print(f"将 {num_time_periods} 个时间段分成 {len(groups)} 组:")
        for i, group in enumerate(groups):
            time_ranges = [f"{time_dirs[j][2]}" for j in group]
            print(f"  组 {i+1}: {', '.join(time_ranges)}")
        
        all_ion_types = sorted({key for time_data in time_grouped_data.values() for key in time_data.keys()})
        
        if not all_ion_types:
            print("没有找到离子数据")
            return
        
        for ion_type in all_ion_types:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_evolution[0], args.figsize_evolution[1]*1.5))
            
            base_color = self.ion_styles.get(ion_type, {'color': '#1f77b4'})['color']
            import matplotlib.colors as mcolors
            color_alphas = np.linspace(0.4, 1.0, len(groups))
            
            bins = args.bins
            
            # Data storage
            ion_label_safe = self.ion_styles.get(ion_type, {'label': ion_type})['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')
            data_centroid_prob, data_interface_prob = {}, {}
            data_centroid_vol, data_interface_vol = {}, {}

            for group_idx, time_indices in enumerate(groups):
                group_d_centroids = []
                group_d_interfaces = []
                
                for time_idx in time_indices:
                    if time_idx in time_grouped_data and ion_type in time_grouped_data[time_idx]:
                        data_list = time_grouped_data[time_idx][ion_type]
                        group_d_centroids.extend([item[0] for item in data_list])
                        group_d_interfaces.extend([item[1] for item in data_list])
                
                if not group_d_centroids:
                    continue
                
                alpha = color_alphas[group_idx]
                color = mcolors.to_rgba(base_color, alpha)
                
                start_time = time_dirs[time_indices[0]][0]
                end_time = time_dirs[time_indices[-1]][1]
                label = f"Time {group_idx+1}: {start_time:.1f}-{end_time:.1f} ns"
                label_safe = label.replace(' ', '_').replace(':', '').replace('-', '_to_')

                # === 密度归一化 ===
                hist_centroid, bin_edges_centroid = np.histogram(group_d_centroids, bins=bins, density=False)
                bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
                hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
                ax1.plot(bin_centers_centroid, hist_centroid_norm, color=color, linewidth=2.5, label=label)
                
                hist_interface, bin_edges_interface = np.histogram(group_d_interfaces, bins=bins, density=False)
                bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
                hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
                ax2.plot(bin_centers_interface, hist_interface_norm, color=color, linewidth=2.5, label=label)
                
                # === 体积归一化 ===
                bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
                shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
                shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
                hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000
                ax3.plot(bin_centers_centroid, hist_centroid_vol, color=color, linewidth=2.5, label=label)
                
                bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
                avg_bubble_radius = np.mean(group_d_centroids)
                effective_radii = avg_bubble_radius + bin_centers_interface
                shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
                shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
                hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000
                ax4.plot(bin_centers_interface, hist_interface_vol, color=color, linewidth=2.5, label=label)

                # Store data
                if 'd_centroid(A)' not in data_centroid_prob:
                    data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                    data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
                if 'd_interface(A)' not in data_interface_prob:
                    data_interface_prob['d_interface(A)'] = bin_centers_interface
                    data_interface_vol['d_interface(A)'] = bin_centers_interface
                
                data_centroid_prob[f'Prob_{label_safe}(%)'] = hist_centroid_norm
                data_interface_prob[f'Prob_{label_safe}(%)'] = hist_interface_norm
                data_centroid_vol[f'ProbVol_{label_safe}(%/nm^3)'] = hist_centroid_vol
                data_interface_vol[f'ProbVol_{label_safe}(%/nm^3)'] = hist_interface_vol

            ion_label = self.ion_styles.get(ion_type, {'label': ion_type})['label']
            
            ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
            ax1.set_ylabel('Probability (%)', fontsize=14)
            ax1.set_title(f'{ion_label} Distance to Bubble Centroid\n(Density Normalized, Time Evolution)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(frameon=False, fontsize=9)
            
            ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
            ax2.set_ylabel('Probability (%)', fontsize=14)
            ax2.set_title(f'{ion_label} Distance to Bubble Surface\n(Density Normalized, Time Evolution)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=False, fontsize=9)
            
            ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
            ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
            ax3.set_title(f'{ion_label} Distance to Bubble Centroid\n(Volume Normalized, Time Evolution)', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend(frameon=False, fontsize=9)
            
            ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
            ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
            ax4.set_title(f'{ion_label} Distance to Bubble Surface\n(Volume Normalized, Time Evolution)', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(frameon=False, fontsize=9)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, f"time_evolution_normalized_{ion_type}_distributions.png")
            plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"{ion_type} 归一化时间演化分布图已保存: {plot_file}")

            # Save data
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_centroid_prob.txt"), [f"Time evolution for {ion_label_safe} dist to centroid (Density Norm)"], data_centroid_prob)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_interface_prob.txt"), [f"Time evolution for {ion_label_safe} dist to interface (Density Norm)"], data_interface_prob)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_centroid_vol.txt"), [f"Time evolution for {ion_label_safe} dist to centroid (Volume Norm)"], data_centroid_vol)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_interface_vol.txt"), [f"Time evolution for {ion_label_safe} dist to interface (Volume Norm)"], data_interface_vol)

    def plot_charge_distributions(self, all_ion_data, time_range_ns, args):
        """绘制指定时间范围内的累积电荷分布图"""
        print(f"生成 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围的电荷分布图...")

        filtered_ion_data = defaultdict(list)
        if not hasattr(self, '_time_dirs'):
            print("错误: time_dirs 未初始化，无法按时间范围筛选")
            return

        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"警告: 在 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内没有找到离子数据")
            return

        # 计算时间范围内的总帧数（假设1ps per frame）
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # 检查时间段是否与指定范围有重叠
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # 计算重叠部分的时长
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1ps per frame
                total_frames_in_range += frames_in_overlap
        
        print(f"时间范围内估计帧数: {total_frames_in_range}")

        self.setup_nature_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        bins = args.bins
        
        # 确定所有数据的范围以使用相同的binning
        all_d_centroids = [d[0] for data in filtered_ion_data.values() for d in data]
        all_d_interfaces = [d[1] for data in filtered_ion_data.values() for d in data]
        if not all_d_centroids:
            print("警告: 筛选后无数据点，无法生成电荷分布图")
            return
            
        range_centroid = (np.min(all_d_centroids), np.max(all_d_centroids))
        range_interface = (np.min(all_d_interfaces), np.max(all_d_interfaces))

        # 初始化总电荷直方图
        total_charge_centroid = np.zeros(bins)
        total_charge_interface = np.zeros(bins)
        bulk_only_charge_centroid = np.zeros(bins)
        bulk_only_charge_interface = np.zeros(bins)

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            charge = self.ion_charges.get(ion_type, 0)
            if charge == 0:
                continue

            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]

            # 计算离子数直方图并乘以电荷，然后除以总帧数得到每帧平均电荷
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, range=range_centroid)
            charge_per_frame_centroid = hist_centroid * charge / max(total_frames_in_range, 1)
            total_charge_centroid += charge_per_frame_centroid
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, range=range_interface)
            charge_per_frame_interface = hist_interface * charge / max(total_frames_in_range, 1)
            total_charge_interface += charge_per_frame_interface

            # 计算仅包含体相离子的电荷
            if ion_type not in ['surface_H', 'surface_OH']:
                bulk_only_charge_centroid += charge_per_frame_centroid
                bulk_only_charge_interface += charge_per_frame_interface

        # 绘制非体积归一化图
        bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
        ax1.plot(bin_centers_centroid, total_charge_centroid, color='k', linewidth=2.5, label='All Ions')
        ax1.plot(bin_centers_centroid, bulk_only_charge_centroid, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        
        bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
        ax2.plot(bin_centers_interface, total_charge_interface, color='k', linewidth=2.5, label='All Ions')
        ax2.plot(bin_centers_interface, bulk_only_charge_interface, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        # 绘制体积归一化图
        bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
        shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
        shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
        charge_density_centroid = total_charge_centroid / shell_volumes_centroid
        charge_density_centroid_bulk = bulk_only_charge_centroid / shell_volumes_centroid
        ax3.plot(bin_centers_centroid, charge_density_centroid, color='k', linewidth=2.5, label='All Ions')
        ax3.plot(bin_centers_centroid, charge_density_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        avg_bubble_radius = np.mean(all_d_centroids)
        bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
        effective_radii = avg_bubble_radius + bin_centers_interface
        shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
        shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
        charge_density_interface = total_charge_interface / shell_volumes_interface
        charge_density_interface_bulk = bulk_only_charge_interface / shell_volumes_interface
        ax4.plot(bin_centers_interface, charge_density_interface, color='k', linewidth=2.5, label='All Ions')
        ax4.plot(bin_centers_interface, charge_density_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        # 设置子图
        title_suffix = f"\n(Time: {time_range_ns[0]}-{time_range_ns[1]} ns)"
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Net Charge per Frame (e)', fontsize=14)
        ax1.set_title('Net Charge per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Net Charge per Frame (e)', fontsize=14)
        ax2.set_title('Net Charge per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax3.set_title('Charge Density per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax4.set_title('Charge Density per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"charge_distributions_normalized_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 电荷分布图已保存: {plot_file}")

        # Save data
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        data_to_save_centroid = {
            'd_centroid(A)': bin_centers_centroid,
            'NetCharge_All_per_frame(e)': total_charge_centroid,
            'NetCharge_Bulk_per_frame(e)': bulk_only_charge_centroid,
            'ChargeDensity_All_per_frame(e/A^3)': charge_density_centroid,
            'ChargeDensity_Bulk_per_frame(e/A^3)': charge_density_centroid_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_charge_dist_centroid_{time_str}.txt"), [f"Charge dist vs d_centroid for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}"], data_to_save_centroid)

        data_to_save_interface = {
            'd_interface(A)': bin_centers_interface,
            'NetCharge_All_per_frame(e)': total_charge_interface,
            'NetCharge_Bulk_per_frame(e)': bulk_only_charge_interface,
            'ChargeDensity_All_per_frame(e/A^3)': charge_density_interface,
            'ChargeDensity_Bulk_per_frame(e/A^3)': charge_density_interface_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_charge_dist_interface_{time_str}.txt"), [f"Charge dist vs d_interface for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}"], data_to_save_interface)

    def plot_cumulative_charge_distributions(self, all_ion_data, time_range_ns, args):
        """绘制指定时间范围内的累积电荷分布图（从最小值累加到当前位置）"""
        print(f"生成 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围的累积电荷分布图...")

        filtered_ion_data = defaultdict(list)
        if not hasattr(self, '_time_dirs'):
            print("错误: time_dirs 未初始化，无法按时间范围筛选")
            return

        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"警告: 在 {time_range_ns[0]}-{time_range_ns[1]} ns 时间范围内没有找到离子数据")
            return

        # 计算时间范围内的总帧数（假设1ps per frame）
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # 检查时间段是否与指定范围有重叠
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # 计算重叠部分的时长
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1ps per frame
                total_frames_in_range += frames_in_overlap
        
        print(f"时间范围内估计帧数: {total_frames_in_range}")

        self.setup_nature_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        bins = args.bins
        
        # 确定所有数据的范围以使用相同的binning
        all_d_centroids = [d[0] for data in filtered_ion_data.values() for d in data]
        all_d_interfaces = [d[1] for data in filtered_ion_data.values() for d in data]
        if not all_d_centroids:
            print("警告: 筛选后无数据点，无法生成累积电荷分布图")
            return
            
        range_centroid = (np.min(all_d_centroids), np.max(all_d_centroids))
        range_interface = (np.min(all_d_interfaces), np.max(all_d_interfaces))

        # 初始化总电荷直方图
        total_charge_centroid = np.zeros(bins)
        total_charge_interface = np.zeros(bins)
        bulk_only_charge_centroid = np.zeros(bins)
        bulk_only_charge_interface = np.zeros(bins)

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            charge = self.ion_charges.get(ion_type, 0)
            if charge == 0:
                continue

            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]

            # 计算离子数直方图并乘以电荷，然后除以总帧数得到每帧平均电荷
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, range=range_centroid)
            charge_per_frame_centroid = hist_centroid * charge / max(total_frames_in_range, 1)
            total_charge_centroid += charge_per_frame_centroid
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, range=range_interface)
            charge_per_frame_interface = hist_interface * charge / max(total_frames_in_range, 1)
            total_charge_interface += charge_per_frame_interface

            # 计算仅包含体相离子的电荷
            if ion_type not in ['surface_H', 'surface_OH']:
                bulk_only_charge_centroid += charge_per_frame_centroid
                bulk_only_charge_interface += charge_per_frame_interface

        # 计算累积电荷（从最小值累加到当前位置）
        cumulative_charge_centroid_all = np.cumsum(total_charge_centroid)
        cumulative_charge_centroid_bulk = np.cumsum(bulk_only_charge_centroid)
        cumulative_charge_interface_all = np.cumsum(total_charge_interface)
        cumulative_charge_interface_bulk = np.cumsum(bulk_only_charge_interface)

        # 绘制累积电荷图（非体积归一化）
        bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
        ax1.plot(bin_centers_centroid, cumulative_charge_centroid_all, color='k', linewidth=2.5, label='All Ions')
        ax1.plot(bin_centers_centroid, cumulative_charge_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
        
        bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
        ax2.plot(bin_centers_interface, cumulative_charge_interface_all, color='k', linewidth=2.5, label='All Ions')
        ax2.plot(bin_centers_interface, cumulative_charge_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # 计算累积球体体积（从中心到当前半径的体积）
        # 对于d_centroid: 球体体积 = 4/3 * π * r^3
        cumulative_volumes_centroid = (4.0/3.0) * np.pi * bin_centers_centroid**3
        cumulative_volumes_centroid = np.maximum(cumulative_volumes_centroid, 1e-10)
        cumulative_charge_density_centroid_all = cumulative_charge_centroid_all / cumulative_volumes_centroid
        cumulative_charge_density_centroid_bulk = cumulative_charge_centroid_bulk / cumulative_volumes_centroid
        
        ax3.plot(bin_centers_centroid, cumulative_charge_density_centroid_all, color='k', linewidth=2.5, label='All Ions')
        ax3.plot(bin_centers_centroid, cumulative_charge_density_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # 对于d_interface: 从气泡表面到当前距离的体积（等效球壳从R到R+d的体积）
        avg_bubble_radius = np.mean(all_d_centroids)
        # 累积体积 = 4/3*π*[(R+d)^3 - R^3]
        outer_radii = avg_bubble_radius + bin_centers_interface
        cumulative_volumes_interface = (4.0/3.0) * np.pi * (outer_radii**3 - avg_bubble_radius**3)
        cumulative_volumes_interface = np.maximum(cumulative_volumes_interface, 1e-10)
        cumulative_charge_density_interface_all = cumulative_charge_interface_all / cumulative_volumes_interface
        cumulative_charge_density_interface_bulk = cumulative_charge_interface_bulk / cumulative_volumes_interface
        
        ax4.plot(bin_centers_interface, cumulative_charge_density_interface_all, color='k', linewidth=2.5, label='All Ions')
        ax4.plot(bin_centers_interface, cumulative_charge_density_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # 设置子图
        title_suffix = f"\n(Time: {time_range_ns[0]}-{time_range_ns[1]} ns)"
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Cumulative Net Charge per Frame (e)', fontsize=14)
        ax1.set_title('Cumulative Net Charge per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Cumulative Net Charge per Frame (e)', fontsize=14)
        ax2.set_title('Cumulative Net Charge per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Cumulative Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax3.set_title('Cumulative Charge Density per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Cumulative Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax4.set_title('Cumulative Charge Density per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"cumulative_charge_distributions_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{time_range_ns[0]}-{time_range_ns[1]} ns 累积电荷分布图已保存: {plot_file}")

        # Save data
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        data_to_save_centroid = {
            'd_centroid(A)': bin_centers_centroid,
            'CumulativeCharge_All_per_frame(e)': cumulative_charge_centroid_all,
            'CumulativeCharge_Bulk_per_frame(e)': cumulative_charge_centroid_bulk,
            'CumulativeChargeDensity_All_per_frame(e/A^3)': cumulative_charge_density_centroid_all,
            'CumulativeChargeDensity_Bulk_per_frame(e/A^3)': cumulative_charge_density_centroid_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_cumulative_charge_dist_centroid_{time_str}.txt"), [f"Cumulative charge dist vs d_centroid for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}", f"Note: Cumulative from minimum distance to current distance"], data_to_save_centroid)

        data_to_save_interface = {
            'd_interface(A)': bin_centers_interface,
            'CumulativeCharge_All_per_frame(e)': cumulative_charge_interface_all,
            'CumulativeCharge_Bulk_per_frame(e)': cumulative_charge_interface_bulk,
            'CumulativeChargeDensity_All_per_frame(e/A^3)': cumulative_charge_density_interface_all,
            'CumulativeChargeDensity_Bulk_per_frame(e/A^3)': cumulative_charge_density_interface_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_cumulative_charge_dist_interface_{time_str}.txt"), [f"Cumulative charge dist vs d_interface for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}", f"Average bubble radius: {avg_bubble_radius:.2f} A", f"Note: Cumulative from bubble surface to current distance"], data_to_save_interface)

    def run_analysis(self, args=None):
        """运行完整的批量分析"""
        if args is None:
            args = type('DefaultArgs', (), {
                'bins': 50, 'dpi': 300,
                'figsize_comprehensive': [20, 8],
                'figsize_evolution': [20, 8],
                'figsize_combined': [24, 10]
            })()
        
        print("开始批量离子距离分布分析...")
        print(f"基础目录: {self.base_dir}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)
        
        time_dirs, total_time = self.find_time_directories()
        if not time_dirs:
            print("错误: 没有找到任何时间段目录")
            return
        
        self._time_dirs = time_dirs # 保存供后续使用
        
        print("="*60)
        
        print("读取气泡质心数据...")
        bubble_data = self.read_bubble_centroids(time_dirs, centroids_subpath=args.centroids_subpath)
        
        print("="*60)

        print("读取离子距离数据...")
        all_ion_data, time_grouped_data, num_time_periods = self.read_ion_distances(time_dirs, centroids_subpath=args.centroids_subpath)
        
        if not all_ion_data:
            print("错误: 没有读取到任何离子数据")
            # 即使没有离子数据，也可能需要处理气泡数据
            if bubble_data:
                print("="*60)
                print("分析气泡大小演化...")
                self.plot_bubble_evolution(bubble_data, args)
            return
        
        print("="*60)
        
        print("数据统计:")
        total_points = sum(len(data) for data in all_ion_data.values())
        for ion_type, data_list in all_ion_data.items():
            print(f"  {ion_type}: {len(data_list)} 个数据点")
        print(f"总数据点: {total_points}")
        print("="*60)
        
        print("生成综合分布图...")
        self.plot_comprehensive_distributions(all_ion_data, total_time, args)
        
        print("="*60)
        
        print("生成时间演化分布图...")
        self.plot_time_grouped_distributions(time_grouped_data, num_time_periods, time_dirs, args)
        
        print("="*60)

        if bubble_data:
            print("分析气泡大小演化...")
            self.plot_bubble_evolution(bubble_data, args)
        
        print("="*60)
        
        # 生成指定时间范围的离子和电荷分布图
        time_range_for_specific_plots = (args.time_range_start, args.time_range_end)
        self.plot_timerange_ion_distributions(all_ion_data, time_range_for_specific_plots, args)
        self.plot_timerange_ion_distributions_absolute(all_ion_data, time_range_for_specific_plots, args)
        self.plot_charge_distributions(all_ion_data, time_range_for_specific_plots, args)
        self.plot_cumulative_charge_distributions(all_ion_data, time_range_for_specific_plots, args)

        print("="*60)
        print("批量分析完成！")
        print(f"输出文件保存在: {self.output_dir}")
        print("\n生成的文件类型:")
        print("  图片文件 (.png):")
        print("  - comprehensive_ion_distributions_normalized.png: 综合所有时间段的离子分布图")
        print("  - time_evolution_normalized_[ion_type]_distributions.png: 各离子类型的时间演化分布图")
        print(f"  - ion_distributions_normalized_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: 特定时间段离子分布图（归一化）")
        print(f"  - ion_distributions_absolute_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: 特定时间段离子分布图（绝对数量）")
        print(f"  - charge_distributions_normalized_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: 每个bin内的电荷分布图")
        print(f"  - cumulative_charge_distributions_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: 累积电荷分布图（从最小距离累加）")
        if bubble_data:
            print("  - bubble_size_evolution.png: 气泡大小随时间演化图")
        
        print("\n  数据文件 (.txt):")
        print("  - data_*.txt: 每个图片文件对应的原始绘图数据")

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='批量分析离子距离分布的脚本')
    
    parser.add_argument('--base_path', 
                        default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4',
                        help='基础目录路径，包含所有时间段子目录的父目录')
    
    parser.add_argument('--output_dir', 
                        default=None,
                        help='输出目录路径。如果不指定，默认为base_path/analysis_ion/centroids_density')
    
    parser.add_argument('--bins', 
                        type=int, 
                        default=50,
                        help='直方图分箱数量 (默认: 50)')
    
    parser.add_argument('--dpi', 
                        type=int, 
                        default=300,
                        help='输出图片的DPI (默认: 300)')
    
    parser.add_argument('--figsize_comprehensive', 
                        nargs=2, 
                        type=float, 
                        default=[20, 8],
                        help='综合分布图的尺寸 (宽度 高度) (默认: 20 8)')
    
    parser.add_argument('--figsize_evolution', 
                        nargs=2, 
                        type=float, 
                        default=[20, 8],
                        help='时间演化图的尺寸 (宽度 高度) (默认: 20 8)')
    
    parser.add_argument('--figsize_combined', 
                        nargs=2, 
                        type=float, 
                        default=[24, 10],
                        help='合并时间演化图的尺寸 (宽度 高度) (默认: 24 10)')
    
    parser.add_argument('--centroids_subpath',
                        default='centroids_density_2',
                        help='气泡质心文件的子路径 (默认: centroids_density_2)')
    
    parser.add_argument('--time_range_start',
                        type=float,
                        default=2.0,
                        help='特定时间范围图的起始时间 (ns) (默认: 2.0)')
    
    parser.add_argument('--time_range_end',
                        type=float,
                        default=5.0,
                        help='特定时间范围图的结束时间 (ns) (默认: 5.0)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = get_args()
    
    if not os.path.isdir(args.base_path):
        print(f"错误: 基础目录 {args.base_path} 不是一个有效的目录")
        sys.exit(1)
    
    analyzer = BatchIonAnalyzer(base_dir=args.base_path, output_dir=args.output_dir)
    
    print("="*60)
    print("批量离子距离分布分析")
    print("="*60)
    print(f"输入目录: {args.base_path}")
    print(f"输出目录: {analyzer.output_dir}")
    print(f"图片参数: DPI={args.dpi}, 分箱数={args.bins}")
    print(f"图片尺寸: 综合图{args.figsize_comprehensive}, 演化图{args.figsize_evolution}, 合并图{args.figsize_combined}")
    print("="*60)
    
    analyzer.run_analysis(args)

if __name__ == "__main__":
    main()