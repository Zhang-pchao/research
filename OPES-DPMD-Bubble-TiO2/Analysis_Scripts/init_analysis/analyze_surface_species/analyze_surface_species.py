import numpy as np
from ase.io import read, write
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import logging
from datetime import datetime
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import argparse

# 设置日志
log_file = 'analyze_surface_species.log'
if os.path.exists(log_file):
    os.remove(log_file)  # 删除已存在的日志文件

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='分析TiO2表面吸附物种')
    parser.add_argument('--format', choices=['exyz', 'lammps'], default='exyz', 
                       help='输入文件格式 (exyz 或 lammps)')
    parser.add_argument('--input', default='dump.xyz', help='输入文件名 (exyz格式) 或 LAMMPS数据文件名')
    parser.add_argument('--traj', default=None, help='LAMMPS轨迹文件名 (仅lammps格式需要)')
    parser.add_argument('--atom_style', default=None, help='LAMMPS atom_style (例如: "id type x y z", "atomic", "full")')
    parser.add_argument('--output', default='dump_last_frame.xyz', help='提取的最后一帧文件名')
    
    # 添加cutoff参数
    parser.add_argument('--ti_o_cutoff', type=float, default=2.5, help='Ti-O距离阈值（Å），用于判断表面吸附')
    parser.add_argument('--oh_cutoff', type=float, default=1.4, help='O-H键长阈值（Å）')
    parser.add_argument('--h_ti_cutoff', type=float, default=2.1, help='H-Ti键长阈值（Å）')
    parser.add_argument('--h_h_cutoff', type=float, default=1.1, help='H-H键长阈值（H2分子）（Å）')
    parser.add_argument('--o_o_cutoff', type=float, default=1.5, help='O-O距离阈值（Å），用于检测O-O近邻')
    parser.add_argument('--n_n_cutoff', type=float, default=1.5, help='N-N距离阈值（Å），用于检测孤立N原子')
    
    return parser.parse_args()

def parse_lattice_from_xyz(filename):
    """从xyz文件的第二行提取lattice信息"""
    try:
        with open(filename, 'r') as f:
            f.readline()  # 跳过第一行（原子数）
            header = f.readline().strip()  # 读取第二行
            
        # 使用正则表达式提取lattice信息
        lattice_match = re.search(r'Lattice="([^"]+)"', header)
        if lattice_match:
            lattice_str = lattice_match.group(1)
            lattice_values = list(map(float, lattice_str.split()))
            
            # 重新组织为3x3矩阵
            lattice_matrix = np.array(lattice_values).reshape(3, 3)
            # 提取对角线元素作为盒子尺寸
            box = np.array([lattice_matrix[0,0], lattice_matrix[1,1], lattice_matrix[2,2]])
            logging.info(f"从文件提取的lattice尺寸: {box}")
            return box
        else:
            logging.warning("未找到lattice信息，使用默认值")
            return np.array([100.0, 100.0, 100.0])
            
    except Exception as e:
        logging.error(f"解析lattice信息失败: {e}")
        return np.array([100.0, 100.0, 100.0])

def apply_pbc(positions, box):
    """应用周期性边界条件，生成所有镜像"""
    # 创建镜像偏移量 (-1, 0, 1) 在每个方向
    offsets = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]])
    
    # 为每个原子创建所有镜像
    extended_positions = []
    original_indices = []
    
    for i, pos in enumerate(positions):
        for offset in offsets:
            extended_pos = pos + offset * box
            extended_positions.append(extended_pos)
            original_indices.append(i)
    
    return np.array(extended_positions), np.array(original_indices)

def find_nearest_oxygens_kdtree(h_positions, o_positions, box, oh_cutoff=1.35):
    """使用KDTree加速找到最近的O原子"""
    logging.info("使用KDTree计算H-O最近邻...")
    
    # 为O原子创建扩展的周期性镜像
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_o_positions)
    
    # 为每个H原子找到最近的O原子
    o_h_counts = defaultdict(int)
    h_bonded_to_o = set()  # 记录与O键合的H原子索引
    
    for h_idx, h_pos in enumerate(h_positions):
        # 查找截断距离内的所有O原子
        indices = tree.query_ball_point(h_pos, oh_cutoff)
        
        if indices:
            # 计算实际距离，找到最近的
            distances = []
            for idx in indices:
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(h_pos - extended_o_pos)
                distances.append((dist, o_original_indices[idx]))
            
            # 找到最近的O原子
            min_dist, nearest_o_idx = min(distances)
            if min_dist <= oh_cutoff:
                o_h_counts[nearest_o_idx] += 1
                h_bonded_to_o.add(h_idx)
    
    return o_h_counts, h_bonded_to_o

def find_h_ti_bonds_kdtree(h_positions, ti_positions, box, h_ti_cutoff=2.0):
    """使用KDTree检测H-Ti键"""
    logging.info("使用KDTree计算H-Ti键...")
    
    # 为Ti原子创建扩展的周期性镜像
    extended_ti_positions, ti_original_indices = apply_pbc(ti_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_ti_positions)
    
    h_bonded_to_ti = set()  # 记录与Ti键合的H原子索引
    
    for h_idx, h_pos in enumerate(h_positions):
        # 查找截断距离内的所有Ti原子
        indices = tree.query_ball_point(h_pos, h_ti_cutoff)
        
        if indices:
            # 检查是否有Ti在截断距离内
            for idx in indices:
                extended_ti_pos = extended_ti_positions[idx]
                dist = np.linalg.norm(h_pos - extended_ti_pos)
                if dist <= h_ti_cutoff:
                    h_bonded_to_ti.add(h_idx)
                    break
    
    return h_bonded_to_ti

def find_h2_molecules_kdtree(h_positions, box, h_h_cutoff=1.0):
    """使用KDTree检测H2分子"""
    logging.info("使用KDTree计算H-H键（H2分子）...")
    
    # 为H原子创建扩展的周期性镜像
    extended_h_positions, h_original_indices = apply_pbc(h_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_h_positions)
    
    h_bonded_to_h = set()  # 记录与H键合的H原子索引
    h2_pairs = []  # 记录H2分子对
    
    for h_idx, h_pos in enumerate(h_positions):
        if h_idx in h_bonded_to_h:
            continue  # 已经被标记为H2分子的一部分
            
        # 查找截断距离内的所有H原子
        indices = tree.query_ball_point(h_pos, h_h_cutoff)
        
        for idx in indices:
            partner_h_idx = h_original_indices[idx]
            # 排除自己，并确保只计算一次每个H2分子
            if partner_h_idx != h_idx and partner_h_idx not in h_bonded_to_h:
                extended_h_pos = extended_h_positions[idx]
                dist = np.linalg.norm(h_pos - extended_h_pos)
                if dist <= h_h_cutoff:
                    h_bonded_to_h.add(h_idx)
                    h_bonded_to_h.add(partner_h_idx)
                    h2_pairs.append((h_idx, partner_h_idx))
                    break
    
    return h_bonded_to_h, h2_pairs

def find_surface_oxygens_kdtree(o_positions, ti_positions, box, ti_o_cutoff=2.3):
    """使用KDTree加速找到表面吸附的O原子，区分上下表面"""
    logging.info("使用KDTree计算Ti-O表面吸附...")
    
    # 为Ti原子创建扩展的周期性镜像
    extended_ti_positions, ti_original_indices = apply_pbc(ti_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_ti_positions)
    
    # 找到最上层和最下层的Ti原子
    ti_z_coords = ti_positions[:, 2]
    max_ti_z = np.max(ti_z_coords)
    min_ti_z = np.min(ti_z_coords)
    
    # 考虑到可能有微小的数值差异，将z坐标相差小于2的Ti认为是在同一层
    top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
    bottom_ti_mask = np.abs(ti_z_coords - min_ti_z) < 2
    
    top_ti_positions = ti_positions[top_ti_mask]
    bottom_ti_positions = ti_positions[bottom_ti_mask]
    
    logging.info(f"找到{len(top_ti_positions)}个顶层Ti原子，z坐标: {max_ti_z:.3f}")
    logging.info(f"找到{len(bottom_ti_positions)}个底层Ti原子，z坐标: {min_ti_z:.3f}")
    
    # 标记表面O原子
    surface_o_mask = np.zeros(len(o_positions), dtype=bool)
    top_surface_mask = np.zeros(len(o_positions), dtype=bool)
    bottom_surface_mask = np.zeros(len(o_positions), dtype=bool)
    
    # 检查每个O原子
    for o_idx, o_pos in enumerate(o_positions):
        # 检查是否在顶层表面
        for ti_pos in top_ti_positions:
            dist = np.linalg.norm(o_pos - ti_pos)
            if dist <= ti_o_cutoff:
                surface_o_mask[o_idx] = True
                top_surface_mask[o_idx] = True
                break
                
        # 如果不在顶层，检查是否在底层表面
        if not surface_o_mask[o_idx]:
            for ti_pos in bottom_ti_positions:
                dist = np.linalg.norm(o_pos - ti_pos)
                if dist <= ti_o_cutoff:
                    surface_o_mask[o_idx] = True
                    bottom_surface_mask[o_idx] = True
                    break
    
    return surface_o_mask, top_surface_mask, bottom_surface_mask

def validate_atom_counts(total_species, surface_species, bulk_species, total_o, total_h, 
                        h_bonded_to_ti, h2_pairs, isolated_h):
    """验证各物种中的O和H原子总数是否与实际原子数一致"""
    logging.info("\n验证原子数量:")
    
    # 计算所有物种中的O原子总数
    calculated_o = 0
    for species in ["O", "OH", "H2O", "H3O"]:
        calculated_o += total_species[species]
    
    # 计算所有物种中的H原子总数
    calculated_h = 0
    calculated_h += total_species["OH"] * 1      # OH中有1个H
    calculated_h += total_species["H2O"] * 2     # H2O中有2个H
    calculated_h += total_species["H3O"] * 3     # H3O中有3个H
    calculated_h += len(h_bonded_to_ti)          # 与Ti键合的H原子
    calculated_h += len(h2_pairs) * 2            # H2分子中的H原子
    calculated_h += len(isolated_h)              # 孤立的H原子
    
    logging.info(f"  实际O原子数: {total_o}")
    logging.info(f"  计算O原子数: {calculated_o}")
    logging.info(f"  O原子差异: {abs(total_o - calculated_o)}")
    
    logging.info(f"  实际H原子数: {total_h}")
    logging.info(f"  计算H原子数: {calculated_h}")
    logging.info(f"    - 与O键合的H: {total_species['OH'] + total_species['H2O']*2 + total_species['H3O']*3}")
    logging.info(f"    - 与Ti键合的H: {len(h_bonded_to_ti)}")
    logging.info(f"    - H2分子中的H: {len(h2_pairs) * 2}")
    logging.info(f"    - 孤立的H: {len(isolated_h)}")
    logging.info(f"  H原子差异: {abs(total_h - calculated_h)}")
    
    if abs(total_o - calculated_o) > 0:
        logging.warning(f"O原子数不匹配！差异: {abs(total_o - calculated_o)}")
    else:
        logging.info("O原子数验证通过!")
        
    if abs(total_h - calculated_h) > 0:
        logging.warning(f"H原子数不匹配！差异: {abs(total_h - calculated_h)}")
    else:
        logging.info("H原子数验证通过!")
    
    return calculated_o, calculated_h

def plot_species_distribution(surface_species, bulk_species, output_file='species_distribution.png'):
    """绘制物种分布图"""
    logging.info("生成物种分布图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    surface_data = [surface_species['OH'], surface_species['H2O']]
    solution_data = [bulk_species['OH'], bulk_species['H3O']]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制TiO2表面吸附物种
    species_surface = ['OH', 'H2O']
    colors_surface = ['orange', 'lightblue']
    
    bars1 = ax1.bar(species_surface, surface_data, color=colors_surface, alpha=0.8)
    ax1.set_title('TiO2 Surface Adsorbed Species', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_xlabel('Species Type', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars1, surface_data):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(surface_data)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # 绘制溶液中的物种
    species_solution = ['OH-', 'H3O+']
    colors_solution = ['red', 'green']
    
    bars2 = ax2.bar(species_solution, solution_data, color=colors_solution, alpha=0.8)
    ax2.set_title('Solution Species', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Molecules', fontsize=12)
    ax2.set_xlabel('Species Type', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars2, solution_data):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(solution_data)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"物种分布图已保存为: {output_file}")

def extract_last_frame(dump_file, output_file):
    """提取dump.xyz的最后一帧并保存"""
    try:
        logging.info("尝试读取文件...")
        atoms = read(dump_file, index='-1')  # 只读取最后一帧
        if atoms is None:
            raise ValueError("无法读取文件内容")
        
        # 保存最后一帧
        write(output_file, atoms)
        logging.info(f"Successfully extracted last frame to {output_file}")
        
        return atoms
        
    except Exception as e:
        logging.error(f"Error extracting last frame: {str(e)}")
        raise

def read_lammps_files(data_file, traj_file, atom_style=None):
    """读取LAMMPS数据文件和轨迹文件"""
    try:
        logging.info(f"读取LAMMPS数据文件: {data_file}")
        logging.info(f"读取LAMMPS轨迹文件: {traj_file}")
        
        # 如果用户指定了atom_style，使用atom_style和LAMMPSDUMP格式
        if atom_style:
            logging.info(f"使用用户指定的 atom_style='{atom_style}' 和 format='LAMMPSDUMP'")
            u = mda.Universe(data_file, traj_file, 
                           atom_style=atom_style,
                           format='LAMMPSDUMP')
        else:
            # 尝试不同的方法读取LAMMPS文件
            try:
                # 方法1: 使用默认参数和LAMMPSDUMP格式
                logging.info("尝试方法1: 使用 atom_style='id type x y z' 和 format='LAMMPSDUMP'")
                u = mda.Universe(data_file, traj_file, 
                               atom_style='id type x y z',
                               format='LAMMPSDUMP')
            except Exception as e1:
                logging.warning(f"方法1失败: {e1}")
                try:
                    # 方法2: 尝试atomic style
                    logging.info("尝试方法2: 使用 atom_style='atomic' 和 format='LAMMPSDUMP'")
                    u = mda.Universe(data_file, traj_file, 
                                   atom_style='atomic',
                                   format='LAMMPSDUMP')
                except Exception as e2:
                    logging.warning(f"方法2失败: {e2}")
                    # 方法3: 尝试full style
                    logging.info("尝试方法3: 使用 atom_style='full' 和 format='LAMMPSDUMP'")
                    u = mda.Universe(data_file, traj_file, 
                                   atom_style='full',
                                   format='LAMMPSDUMP')
        
        # 获取最后一帧
        u.trajectory[-1]
        
        # 获取原子信息
        positions = u.atoms.positions
        atom_types = u.atoms.types
        
        # 获取盒子尺寸
        box_dims = u.dimensions[:3]  # 只取前三个维度 (x, y, z)
        
        logging.info(f"读取到 {len(u.atoms)} 个原子")
        logging.info(f"盒子尺寸: {box_dims}")
        logging.info(f"原子类型: {np.unique(atom_types)}")
        
        # 创建类似ASE的原子对象结构
        class LAMMPSAtoms:
            def __init__(self, positions, symbols, box):
                self.positions = positions
                self.symbols = symbols
                self.box = box
                
            def get_positions(self):
                return self.positions
                
            def get_chemical_symbols(self):
                return self.symbols
        
        # 将LAMMPS原子类型映射到化学元素
        # 这里需要根据你的实际情况调整映射关系
        type_to_element = {
            '1': 'H',   
            '2': 'O',     
            '3': 'N',  
            '4': 'Na',   
            '5': 'Cl',
            '6': 'Ti',
        }
        
        # 转换原子类型到化学符号
        symbols = [type_to_element.get(str(t), 'X') for t in atom_types]
        
        # 检查是否有未知的原子类型
        unknown_types = set(str(t) for t in atom_types if str(t) not in type_to_element)
        if unknown_types:
            logging.warning(f"发现未知的原子类型: {unknown_types}")
            logging.warning("请检查type_to_element映射关系")
        
        # 统计原子类型
        symbol_counts = Counter(symbols)
        logging.info(f"原子统计: {dict(symbol_counts)}")
        
        # 创建原子对象
        atoms = LAMMPSAtoms(positions, symbols, box_dims)
        
        return atoms, box_dims
        
    except Exception as e:
        logging.error(f"读取LAMMPS文件时出错: {str(e)}")
        raise

def extract_last_frame_lammps(data_file, traj_file, output_file, atom_style=None):
    """从LAMMPS文件中提取最后一帧并保存为xyz格式"""
    try:
        logging.info("读取LAMMPS文件...")
        atoms, box = read_lammps_files(data_file, traj_file, atom_style)
        
        # 保存为xyz格式
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        with open(output_file, 'w') as f:
            f.write(f"{len(positions)}\n")
            f.write(f'Lattice="{box[0]} 0.0 0.0 0.0 {box[1]} 0.0 0.0 0.0 {box[2]}" Properties=species:S:1:pos:R:3\n')
            for symbol, pos in zip(symbols, positions):
                f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        logging.info(f"成功提取最后一帧到 {output_file}")
        
        return atoms
        
    except Exception as e:
        logging.error(f"提取LAMMPS最后一帧时出错: {str(e)}")
        raise

def parse_lattice_from_lammps(box_dims):
    """从LAMMPS盒子尺寸获取lattice信息"""
    try:
        logging.info(f"LAMMPS盒子尺寸: {box_dims}")
        return box_dims
    except Exception as e:
        logging.error(f"解析LAMMPS lattice信息失败: {e}")
        return np.array([100.0, 100.0, 100.0])

def find_close_oxygen_pairs_kdtree(o_positions, box, o_o_cutoff=1.5):
    """使用KDTree检测O-O近邻对"""
    logging.info("使用KDTree检测O-O近邻...")
    
    # 为O原子创建扩展的周期性镜像
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_o_positions)
    
    # 存储找到的O-O近邻对
    close_o_pairs = set()
    
    # 检查每个O原子
    for o_idx, o_pos in enumerate(o_positions):
        # 查找截断距离内的所有O原子
        indices = tree.query_ball_point(o_pos, o_o_cutoff)
        
        for idx in indices:
            partner_o_idx = o_original_indices[idx]
            # 排除自己，并确保每对只计算一次
            if partner_o_idx > o_idx:  # 只记录一次每对原子
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(o_pos - extended_o_pos)
                if dist <= o_o_cutoff:
                    close_o_pairs.add((o_idx, partner_o_idx, dist))
    
    return close_o_pairs

def find_isolated_nitrogen_kdtree(n_positions, box, n_n_cutoff=1.5):
    """使用KDTree检测孤立N原子"""
    logging.info("使用KDTree检测孤立N原子...")
    
    if len(n_positions) == 0:
        return set()
    
    # 为N原子创建扩展的周期性镜像
    extended_n_positions, n_original_indices = apply_pbc(n_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_n_positions)
    
    # 存储孤立的N原子索引
    isolated_n = set()
    
    # 检查每个N原子
    for n_idx, n_pos in enumerate(n_positions):
        # 查找截断距离内的所有N原子
        indices = tree.query_ball_point(n_pos, n_n_cutoff)
        
        # 移除自身的索引
        indices = [idx for idx in indices if n_original_indices[idx] != n_idx]
        
        # 如果没有近邻N原子，则认为是孤立的
        if not indices:
            isolated_n.add(n_idx)
    
    return isolated_n

def analyze_surface_species(atoms, box_dims=None, output_file=None, ti_o_cutoff=2.3, oh_cutoff=1.35, 
                          h_ti_cutoff=2.0, h_h_cutoff=1.0, o_o_cutoff=1.5, n_n_cutoff=1.5):
    """分析表面吸附物种"""
    try:
        # 检查atoms是否有效
        if atoms is None:
            raise ValueError("Invalid atoms object")
        
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # 统计原子总数
        atom_counts = Counter(symbols)
        total_o = atom_counts.get('O', 0)
        total_h = atom_counts.get('H', 0)
        
        # 获取盒子尺寸
        if box_dims is not None:
            box = box_dims
        elif output_file is not None:
            box = parse_lattice_from_xyz(output_file)
        else:
            box = np.array([100.0, 100.0, 100.0])
            logging.warning("未提供盒子尺寸信息，使用默认值")
        
        # 获取各种原子的索引
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]
        ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
        n_indices = [i for i, s in enumerate(symbols) if s == "N"]  # 添加N原子索引
        
        logging.info(f"找到原子数量: O={len(o_indices)}, H={len(h_indices)}, Ti={len(ti_indices)}, N={len(n_indices)}")
        
        if len(o_indices) == 0 or len(ti_indices) == 0:
            raise ValueError("No O or Ti atoms found in the structure")
        
        # 找到最小的Ti原子索引，用于区分TiO2中的O和溶液中的O
        min_ti_index = min(ti_indices)
        max_ti_index = max(ti_indices)
        
        # 区分TiO2中的O和溶液中的O
        tio2_o_indices = [i for i in o_indices if i < min_ti_index]
        solution_o_indices = [i for i in o_indices if i > max_ti_index]
        
        logging.info(f"TiO2中的O原子数量: {len(tio2_o_indices)}")
        logging.info(f"溶液中的O原子数量: {len(solution_o_indices)}")
        
        # 检查是否有位于Ti原子之间的O原子（这种情况不应该存在）
        middle_o_indices = [i for i in o_indices if min_ti_index <= i <= max_ti_index]
        if middle_o_indices:
            logging.warning(f"发现{len(middle_o_indices)}个O原子的索引位于Ti原子之间，这可能表明原子排序有问题")
        
        o_positions = positions[o_indices]
        h_positions = positions[h_indices] if h_indices else np.array([])
        ti_positions = positions[ti_indices]
        n_positions = positions[n_indices] if n_indices else np.array([])  # 添加N原子位置
        
        # 找到最上层的Ti原子
        ti_z_coords = ti_positions[:, 2]
        max_ti_z = np.max(ti_z_coords)
        # 考虑到可能有微小的数值差异，将z坐标相差小于2的Ti认为是在同一层
        top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
        top_ti_positions = ti_positions[top_ti_mask]
        
        if len(top_ti_positions) == 0:
            raise ValueError("No top layer Ti atoms found")
        
        logging.info(f"找到{len(top_ti_positions)}个顶层Ti原子，z坐标: {max_ti_z:.3f}")
        
        # 使用KDTree快速计算表面Ti和O之间的距离，区分上下表面
        surface_o_mask, top_surface_mask, bottom_surface_mask = find_surface_oxygens_kdtree(
            o_positions, ti_positions, box, ti_o_cutoff)
        
        # 初始化表面物种计数器（分别统计来自TiO2和溶液的，以及上下表面）
        surface_species_tio2_top = defaultdict(int)     # TiO2顶表面物种计数
        surface_species_solution_top = defaultdict(int)  # 溶液来源的顶表面物种计数
        surface_species_tio2_bottom = defaultdict(int)   # TiO2底表面物种计数
        surface_species_solution_bottom = defaultdict(int)  # 溶液来源的底表面物种计数
        bulk_species = defaultdict(int)  # 体相物种计数
        
        # 使用KDTree快速计算H-O最近邻
        o_h_counts = defaultdict(int)
        h_bonded_to_o = set()
        if len(h_indices) > 0 and len(o_indices) > 0:
            o_h_counts, h_bonded_to_o = find_nearest_oxygens_kdtree(h_positions, o_positions, box, oh_cutoff)
        
        # 检测H-Ti键
        h_bonded_to_ti = set()
        if len(h_indices) > 0 and len(ti_indices) > 0:
            h_bonded_to_ti = find_h_ti_bonds_kdtree(h_positions, ti_positions, box, h_ti_cutoff)
        
        # 检测H2分子
        h_bonded_to_h = set()
        h2_pairs = []
        if len(h_indices) > 1:
            h_bonded_to_h, h2_pairs = find_h2_molecules_kdtree(h_positions, box, h_h_cutoff)
        
        # 找到孤立的H原子（既不与O、Ti、H键合的H原子）
        all_h_indices = set(range(len(h_indices)))
        isolated_h = all_h_indices - h_bonded_to_o - h_bonded_to_ti - h_bonded_to_h
        
        # 检测O-O近邻
        close_o_pairs = find_close_oxygen_pairs_kdtree(o_positions, box, o_o_cutoff)
        if close_o_pairs:
            logging.info("\nO-O近邻分析:")
            logging.info(f"发现{len(close_o_pairs)}对O-O近邻（距离 < {o_o_cutoff}Å）:")
            for o1_idx, o2_idx, dist in close_o_pairs:
                logging.info(f"  O原子对: {o_indices[o1_idx]}-{o_indices[o2_idx]}, 距离: {dist:.3f}Å")
        else:
            logging.info(f"\n未发现O-O近邻（距离 < {o_o_cutoff}Å）")
        
        # 检测孤立N原子
        if len(n_positions) > 0:
            isolated_n = find_isolated_nitrogen_kdtree(n_positions, box, n_n_cutoff)
            if isolated_n:
                logging.info("\nN原子解离分析:")
                logging.info(f"发现{len(isolated_n)}个孤立N原子（周围{n_n_cutoff}Å内无其他N原子）:")
                for n_idx in isolated_n:
                    logging.info(f"  孤立N原子索引: {n_indices[n_idx]}")
            else:
                logging.info(f"\n未发现孤立N原子（所有N原子在{n_n_cutoff}Å范围内都有N原子近邻）")
        
        # 输出额外的H原子信息
        logging.info(f"\nH原子键合分析:")
        logging.info(f"  与O键合的H原子: {len(h_bonded_to_o)}")
        logging.info(f"  与Ti键合的H原子: {len(h_bonded_to_ti)}")
        logging.info(f"  H2分子数量: {len(h2_pairs)} (包含{len(h2_pairs)*2}个H原子)")
        logging.info(f"  孤立H原子: {len(isolated_h)}")
        
        # 分析每个O原子
        tio2_o_count = 0  # TiO2中的O原子计数
        tio2_bulk_o_count = 0  # TiO2体相中的O原子计数
        solution_o_count = 0  # 溶液中的O原子计数
        unaccounted_solution_o = []  # 未被计入任何物种的溶液中的O原子
        
        for i, o_idx in enumerate(o_indices):
            h_count = o_h_counts[i]
            is_surface = surface_o_mask[i]
            is_top = top_surface_mask[i]
            is_bottom = bottom_surface_mask[i]
            
            # 如果是TiO2中的O原子
            if o_idx < min_ti_index:
                tio2_o_count += 1
                # 根据H数量确定物种类型
                species_type = "O"
                if h_count == 1:
                    species_type = "OH"
                elif h_count == 2:
                    species_type = "H2O"
                elif h_count == 3:
                    species_type = "H3O"
                
                # 统计表面物种
                if is_top:
                    surface_species_tio2_top[species_type] += 1
                elif is_bottom:
                    surface_species_tio2_bottom[species_type] += 1
                else:
                    tio2_bulk_o_count += 1
            
            # 如果是溶液中的O原子
            elif o_idx > max_ti_index:
                solution_o_count += 1
                # 溶液中的物种
                if h_count == 1:
                    species_type = "OH"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 2:
                    species_type = "H2O"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 3:
                    species_type = "H3O"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 0:
                    unaccounted_solution_o.append((o_idx, h_count))
                    logging.warning(f"发现溶液中的孤立O原子，索引为{o_idx}")
        
        # 合并表面物种统计（用于总体统计）
        surface_species = defaultdict(int)
        for species in ["O", "OH", "H2O", "H3O"]:
            surface_species[species] = (
                surface_species_tio2_top[species] + surface_species_solution_top[species] +
                surface_species_tio2_bottom[species] + surface_species_solution_bottom[species]
            )
        
        # 输出分析结果
        logging.info("\n整个体系中的物种分布:")
        total_species = defaultdict(int)
        for species in ["O", "OH", "H2O", "H3O"]:
            total = surface_species[species] + bulk_species[species]
            total_species[species] = total
            logging.info(f"  {species}: {total} (表面: {surface_species[species]}, 体相: {bulk_species[species]})")
        
        logging.info("\nTiO2顶表面吸附物种分析:")
        logging.info("来自TiO2的表面物种:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species_tio2_top[species]
            if count > 0:
                logging.info(f"  吸附的{species}: {count}")
        
        logging.info("\nTiO2顶表面吸附来自溶液的表面物种:")
        for species in ["OH", "H2O", "H3O"]:
            count = surface_species_solution_top[species]
            if count > 0:
                logging.info(f"  吸附的{species}: {count}")
        
        logging.info("\nTiO2底表面吸附物种分析:")
        logging.info("来自TiO2的表面物种:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species_tio2_bottom[species]
            if count > 0:
                logging.info(f"  吸附的{species}: {count}")
        
        logging.info("\nTiO2底表面吸附来自溶液的表面物种:")
        for species in ["OH", "H2O", "H3O"]:
            count = surface_species_solution_bottom[species]
            if count > 0:
                logging.info(f"  吸附的{species}: {count}")
        
        logging.info("\n表面物种总计:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species[species]
            if count > 0:
                total_top = surface_species_tio2_top[species] + surface_species_solution_top[species]
                total_bottom = surface_species_tio2_bottom[species] + surface_species_solution_bottom[species]
                logging.info(f"  吸附的{species}: {count} (顶表面: {total_top}, 底表面: {total_bottom})")
        
        logging.info("\n溶液中的物种分布:")
        for species in ["OH", "H2O", "H3O"]:
            count = bulk_species[species]
            if count > 0:
                logging.info(f"  溶液中的{species}: {count}")
        
        # 输出氧原子统计信息
        logging.info("\n氧原子详细统计:")
        logging.info(f"  TiO2中的O原子总数: {tio2_o_count}")
        top_surface_o = sum(surface_species_tio2_top.values())
        bottom_surface_o = sum(surface_species_tio2_bottom.values())
        logging.info(f"    - 顶表面O原子数: {top_surface_o}")
        logging.info(f"    - 底表面O原子数: {bottom_surface_o}")
        logging.info(f"    - 体相O原子数: {tio2_bulk_o_count}")
        logging.info(f"  溶液中的O原子总数: {solution_o_count}")
        logging.info(f"    - 参与成键的O原子数: {solution_o_count - len(unaccounted_solution_o)}")
        if unaccounted_solution_o:
            logging.info(f"    - 未成键的O原子数: {len(unaccounted_solution_o)}")
            logging.info("    - 未成键O原子详细信息:")
            for o_idx, h_count in unaccounted_solution_o:
                logging.info(f"      索引: {o_idx}, 连接H数: {h_count}")
        
        # 验证原子数量
        validate_atom_counts(total_species, surface_species, bulk_species, total_o, total_h,
                           h_bonded_to_ti, h2_pairs, isolated_h)
        
        # 生成物种分布图
        plot_species_distribution(surface_species, bulk_species)
        
        return total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n if len(n_positions) > 0 else set()
        
    except Exception as e:
        logging.error(f"分析过程中出错: {str(e)}")
        logging.error("错误详细信息:", exc_info=True)
        raise

if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    
    try:
        # 记录开始时间
        start_time = datetime.now()
        
        if args.format == 'exyz':
            logging.info(f"开始分析 {args.input} (EXYZ格式)")
            # 提取最后一帧
            atoms = extract_last_frame(args.input, args.output)
            
            # 分析表面物种
            total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n = analyze_surface_species(
                atoms,
                output_file=args.output,
                ti_o_cutoff=args.ti_o_cutoff,
                oh_cutoff=args.oh_cutoff,
                h_ti_cutoff=args.h_ti_cutoff,
                h_h_cutoff=args.h_h_cutoff,
                o_o_cutoff=args.o_o_cutoff,
                n_n_cutoff=args.n_n_cutoff
            )
            
        elif args.format == 'lammps':
            # 设置默认的LAMMPS文件路径
            if args.traj is None:
                args.traj = '../bubble_1k.lammpstrj'
            if args.input == 'dump.xyz':  # 如果使用默认值，改为LAMMPS数据文件
                args.input = '../model_atomic.data'
                
            logging.info(f"开始分析 LAMMPS文件: {args.input}, {args.traj}")
            if args.atom_style:
                logging.info(f"使用指定的 atom_style: {args.atom_style}")
            
            # 提取最后一帧
            atoms = extract_last_frame_lammps(args.input, args.traj, args.output, args.atom_style)
            
            # 获取盒子尺寸
            _, box_dims = read_lammps_files(args.input, args.traj, args.atom_style)
            
            # 分析表面物种
            total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n = analyze_surface_species(
                atoms,
                box_dims=box_dims,
                ti_o_cutoff=args.ti_o_cutoff,
                oh_cutoff=args.oh_cutoff,
                h_ti_cutoff=args.h_ti_cutoff,
                h_h_cutoff=args.h_h_cutoff,
                o_o_cutoff=args.o_o_cutoff,
                n_n_cutoff=args.n_n_cutoff
            )
        
        # 记录结束时间和总用时
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\n分析完成！总用时: {duration}")
        
    except Exception as e:
        logging.error(f"分析过程中出错: {e}")
        raise 