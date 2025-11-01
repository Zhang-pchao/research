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
def setup_logging(enable_log_file=False):
    """设置日志配置"""
    log_handlers = [logging.StreamHandler()]
    
    if enable_log_file:
        log_file = 'analyze_ion_species.log'
        if os.path.exists(log_file):
            os.remove(log_file)  # 删除已存在的日志文件
        log_handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='分析TiO2表面离子物种')
    parser.add_argument('--format', choices=['exyz', 'lammps'], default='lammps', 
                       help='输入文件格式 (exyz 或 lammps)')
    parser.add_argument('--input', default='../model_atomic.data', help='输入文件名 (exyz格式) 或 LAMMPS数据文件名')
    parser.add_argument('--traj', default='../bubble_1k.lammpstrj', help='LAMMPS轨迹文件名 (仅lammps格式需要)')
    parser.add_argument('--atom_style', default=None, help='LAMMPS atom_style (例如: "id type x y z", "atomic", "full")')
    parser.add_argument('--step_interval', type=int, default=100, help='分析帧间隔，每隔n帧分析一次')
    parser.add_argument('--start_frame', type=int, default=0, help='开始分析的帧数')
    parser.add_argument('--end_frame', type=int, default=-1, help='结束分析的帧数，-1表示到最后一帧')
    
    # 添加cutoff参数
    parser.add_argument('--ti_o_cutoff', type=float, default=3.5, help='Ti-O距离阈值（Å），用于判断表面吸附')
    parser.add_argument('--oh_cutoff', type=float, default=1.4, help='O-H键长阈值（Å）')
    parser.add_argument('--h_ti_cutoff', type=float, default=2.1, help='H-Ti键长阈值（Å）')
    parser.add_argument('--enable_log_file', action='store_true', help='启用log文件输出')
    
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
    """使用KDTree加速找到最近的O原子，建立准确的O-H键合关系"""
    
    # 为O原子创建扩展的周期性镜像
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)
    
    # 构建KDTree
    tree = cKDTree(extended_o_positions)
    
    # 建立O-H键合关系映射
    o_h_bonds = defaultdict(list)  # 每个O原子键合的H原子索引列表
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
                o_h_bonds[nearest_o_idx].append(h_idx)
                h_bonded_to_o.add(h_idx)
    
    # 计算每个O原子的H计数
    o_h_counts = {o_idx: len(h_list) for o_idx, h_list in o_h_bonds.items()}
    
    return o_h_counts, h_bonded_to_o, o_h_bonds

def find_surface_oxygens_kdtree(o_positions, ti_positions, box, ti_o_cutoff=3.5):
    """使用KDTree加速找到表面吸附的O原子，只考虑顶表面"""
    
    # 找到最上层的Ti原子
    ti_z_coords = ti_positions[:, 2]
    max_ti_z = np.max(ti_z_coords)
    
    # 考虑到可能有微小的数值差异，将z坐标相差小于2的Ti认为是在同一层
    top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
    top_ti_positions = ti_positions[top_ti_mask]
    
    logging.debug(f"找到{len(top_ti_positions)}个顶层Ti原子，z坐标: {max_ti_z:.3f}")
    
    # 标记顶表面O原子
    top_surface_mask = np.zeros(len(o_positions), dtype=bool)
    
    # 检查每个O原子是否在顶层表面
    for o_idx, o_pos in enumerate(o_positions):
        for ti_pos in top_ti_positions:
            dist = np.linalg.norm(o_pos - ti_pos)
            if dist <= ti_o_cutoff:
                top_surface_mask[o_idx] = True
                break
    
    return top_surface_mask

def read_lammps_frame(u, frame_idx):
    """读取LAMMPS轨迹的指定帧"""
    try:
        u.trajectory[frame_idx]
        
        # 获取原子信息
        positions = u.atoms.positions
        atom_types = u.atoms.types
        
        # 获取完整的盒子信息 [a, b, c, alpha, beta, gamma]
        box_info = u.dimensions
        box_dims = box_info[:3]  # 只取前三个维度 (x, y, z) 用于距离计算
        
        # 将LAMMPS原子类型映射到化学元素
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
        
        # 创建类似ASE的原子对象结构
        class LAMMPSAtoms:
            def __init__(self, positions, symbols, box_dims, box_info):
                self.positions = positions
                self.symbols = symbols
                self.box = box_dims
                self.box_info = box_info  # 完整的盒子信息
                
            def get_positions(self):
                return self.positions
                
            def get_chemical_symbols(self):
                return self.symbols
        
        atoms = LAMMPSAtoms(positions, symbols, box_dims, box_info)
        
        return atoms, box_dims, box_info
        
    except Exception as e:
        logging.error(f"读取第{frame_idx}帧时出错: {str(e)}")
        return None, None, None

def write_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info=None, append_mode=False, atom_indices=None):
    """将坐标写入extended xyz文件，可选地包含原子index信息"""
    try:
        mode = 'a' if append_mode else 'w'
        with open(filename, mode) as f:
            f.write(f"{len(positions)}\n")
            
            # 构建extended xyz格式的第二行
            header_parts = []
            
            # 添加Frame信息
            header_parts.append(f"Frame={frame_idx}")
            
            # 添加PBC信息（默认三个方向都是周期性的）
            header_parts.append('pbc="T T T"')
            
            # 添加晶格信息（如果有的话）
            if box_info is not None and len(box_info) >= 3:
                # 构建3x3晶格矩阵
                a, b, c = box_info[:3]
                if len(box_info) >= 6:
                    alpha, beta, gamma = box_info[3:6]
                    # 对于非正交盒子，需要转换为晶格向量
                    # 简化处理：假设是正交盒子（大多数LAMMPS模拟）
                    lattice_str = f"{a} 0.0 0.0 0.0 {b} 0.0 0.0 0.0 {c}"
                else:
                    # 正交盒子
                    lattice_str = f"{a} 0.0 0.0 0.0 {b} 0.0 0.0 0.0 {c}"
                header_parts.append(f'lattice="{lattice_str}"')
            
            # 添加properties信息 - 如果有atom_indices，添加额外的列
            if atom_indices is not None:
                header_parts.append('properties=species:S:1:pos:R:3:atom_index:I:1')
            else:
                header_parts.append('properties=species:S:1:pos:R:3')
            
            # 写入header行
            header = " ".join(header_parts)
            f.write(f"{header}\n")
            
            # 写入原子坐标
            if atom_indices is not None:
                for symbol, pos, idx in zip(symbols, positions, atom_indices):
                    f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {idx}\n")
            else:
                for symbol, pos in zip(symbols, positions):
                    f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    except Exception as e:
        logging.error(f"写入坐标文件 {filename} 时出错: {e}")

def append_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info=None, atom_indices=None):
    """将坐标追加到extended xyz文件"""
    write_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info, append_mode=True, atom_indices=atom_indices)

def verify_bonding_distances(coords, symbols, oh_cutoff=1.35, max_oh_distance=1.8):
    """验证分子内原子的键合距离是否合理"""
    if len(coords) == 0:
        return False, "空分子"
    
    # 找到O原子和H原子的位置
    o_positions = []
    h_positions = []
    
    for i, (coord, symbol) in enumerate(zip(coords, symbols)):
        if symbol == 'O':
            o_positions.append((i, coord))
        elif symbol == 'H':
            h_positions.append((i, coord))
    
    # 验证应该只有一个O原子
    if len(o_positions) != 1:
        return False, f"分子中应该只有1个O原子，实际有{len(o_positions)}个"
    
    # 验证O-H键合距离
    o_coord = o_positions[0][1]
    for h_idx, h_coord in h_positions:
        distance = np.linalg.norm(np.array(o_coord) - np.array(h_coord))
        if distance > max_oh_distance:
            return False, f"O-H距离过大: {distance:.3f}Å > {max_oh_distance}Å"
        if distance < 0.5:  # 过小的距离也不合理
            return False, f"O-H距离过小: {distance:.3f}Å < 0.5Å"
    
    return True, "键合距离验证通过"

def create_valid_molecule(o_coord, h_coords, h_indices, expected_h_count, oh_cutoff=1.35):
    """创建经过验证的分子，确保所有H原子都与O原子真正键合"""
    if len(h_coords) != expected_h_count:
        logging.warning(f"H原子数量不匹配: 期望{expected_h_count}个，实际{len(h_coords)}个")
        return None
    
    # 验证每个H原子与O原子的距离
    valid_h_coords = []
    valid_h_indices = []
    
    for h_coord, h_idx in zip(h_coords, h_indices):
        distance = np.linalg.norm(np.array(o_coord) - np.array(h_coord))
        if distance <= oh_cutoff:
            valid_h_coords.append(h_coord)
            valid_h_indices.append(h_idx)
        else:
            logging.warning(f"H原子 {h_idx} 与O原子距离过大 ({distance:.3f}Å)，跳过")
    
    # 检查有效的H原子数量是否符合期望
    if len(valid_h_coords) != expected_h_count:
        logging.warning(f"有效H原子数量不匹配: 期望{expected_h_count}个，有效{len(valid_h_coords)}个")
        return None
    
    # 创建分子
    molecule = {
        'coords': [o_coord] + valid_h_coords,
        'symbols': ['O'] + ['H'] * len(valid_h_coords)
    }
    
    # 验证键合距离
    is_valid, message = verify_bonding_distances(molecule['coords'], molecule['symbols'], oh_cutoff)
    if not is_valid:
        logging.warning(f"分子键合验证失败: {message}")
        return None
    
    return molecule

def validate_molecular_integrity(results, frame_idx):
    """验证分子完整性并生成报告，包括符号序列和键合距离验证"""
    integrity_report = {
        'frame': frame_idx,
        'tio2_surface_h': {'valid': 0, 'invalid': 0},
        'solution_surface_oh': {'valid': 0, 'invalid': 0},
        'solution_surface_h2o': {'valid': 0, 'invalid': 0},
        'solution_bulk_oh': {'valid': 0, 'invalid': 0},
        'solution_bulk_h3o': {'valid': 0, 'invalid': 0}
    }
    
    # 验证TiO2表面H (期望1个原子: H)
    for molecule in results['tio2_surface_h']['molecules']:
        if (len(molecule['coords']) == 1 and len(molecule['symbols']) == 1 and 
            molecule['symbols'] == ['H']):
            integrity_report['tio2_surface_h']['valid'] += 1
        else:
            integrity_report['tio2_surface_h']['invalid'] += 1
            logging.warning(f"第{frame_idx}帧: TiO2表面H分子不完整: {len(molecule['coords'])}个坐标, "
                          f"符号: {molecule['symbols']}")
    
    # 验证表面OH (期望2个原子: O, H)
    for molecule in results['solution_surface_oh']['molecules']:
        is_valid = (len(molecule['coords']) == 2 and len(molecule['symbols']) == 2 and 
                   molecule['symbols'] == ['O', 'H'])
        if is_valid:
            # 额外验证键合距离
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_surface_oh']['valid'] += 1
            else:
                integrity_report['solution_surface_oh']['invalid'] += 1
                logging.warning(f"第{frame_idx}帧: 表面OH分子键合验证失败: {message}")
        else:
            integrity_report['solution_surface_oh']['invalid'] += 1
            logging.warning(f"第{frame_idx}帧: 表面OH分子格式错误: {len(molecule['coords'])}个坐标, "
                          f"符号: {molecule['symbols']}")
    
    # 验证表面H2O (期望3个原子: O, H, H)
    for molecule in results['solution_surface_h2o']['molecules']:
        is_valid = (len(molecule['coords']) == 3 and len(molecule['symbols']) == 3 and 
                   molecule['symbols'] == ['O', 'H', 'H'])
        if is_valid:
            # 额外验证键合距离
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_surface_h2o']['valid'] += 1
            else:
                integrity_report['solution_surface_h2o']['invalid'] += 1
                logging.warning(f"第{frame_idx}帧: 表面H2O分子键合验证失败: {message}")
        else:
            integrity_report['solution_surface_h2o']['invalid'] += 1
            logging.warning(f"第{frame_idx}帧: 表面H2O分子格式错误: {len(molecule['coords'])}个坐标, "
                          f"符号: {molecule['symbols']}")
    
    # 验证体相OH (期望2个原子: O, H)
    for molecule in results['solution_bulk_oh']['molecules']:
        is_valid = (len(molecule['coords']) == 2 and len(molecule['symbols']) == 2 and 
                   molecule['symbols'] == ['O', 'H'])
        if is_valid:
            # 额外验证键合距离
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_bulk_oh']['valid'] += 1
            else:
                integrity_report['solution_bulk_oh']['invalid'] += 1
                logging.warning(f"第{frame_idx}帧: 体相OH分子键合验证失败: {message}")
        else:
            integrity_report['solution_bulk_oh']['invalid'] += 1
            logging.warning(f"第{frame_idx}帧: 体相OH分子格式错误: {len(molecule['coords'])}个坐标, "
                          f"符号: {molecule['symbols']}")
    
    # 验证H3O (期望4个原子: O, H, H, H)
    for molecule in results['solution_bulk_h3o']['molecules']:
        is_valid = (len(molecule['coords']) == 4 and len(molecule['symbols']) == 4 and 
                   molecule['symbols'] == ['O', 'H', 'H', 'H'])
        if is_valid:
            # 额外验证键合距离
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_bulk_h3o']['valid'] += 1
            else:
                integrity_report['solution_bulk_h3o']['invalid'] += 1
                logging.warning(f"第{frame_idx}帧: H3O分子键合验证失败: {message}")
        else:
            integrity_report['solution_bulk_h3o']['invalid'] += 1
            logging.warning(f"第{frame_idx}帧: H3O分子格式错误: {len(molecule['coords'])}个坐标, "
                          f"符号: {molecule['symbols']}")
    
    return integrity_report

def analyze_frame_ion_species(atoms, box_dims, frame_idx, ti_o_cutoff=3.5, oh_cutoff=1.35):
    """分析单帧的离子物种"""
    try:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # 获取各种原子的索引
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]
        ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
        na_indices = [i for i, s in enumerate(symbols) if s == "Na"]
        cl_indices = [i for i, s in enumerate(symbols) if s == "Cl"]
        
        if len(o_indices) == 0 or len(ti_indices) == 0 or len(h_indices) == 0:
            logging.warning(f"第{frame_idx}帧：缺少必要的原子类型")
            return None
        
        # 找到最小的Ti原子索引，用于区分TiO2中的O和溶液中的O
        min_ti_index = min(ti_indices)
        max_ti_index = max(ti_indices)
        
        # 区分TiO2中的O和溶液中的O
        tio2_o_indices = [i for i in o_indices if i < min_ti_index]
        solution_o_indices = [i for i in o_indices if i > max_ti_index]
        
        o_positions = positions[o_indices]
        h_positions = positions[h_indices]
        ti_positions = positions[ti_indices]
        
        # 找到顶表面的O原子
        top_surface_mask = find_surface_oxygens_kdtree(o_positions, ti_positions, box_dims, ti_o_cutoff)
        
        # 计算H-O键合
        o_h_counts, h_bonded_to_o, o_h_bonds = find_nearest_oxygens_kdtree(h_positions, o_positions, box_dims, oh_cutoff)
        
        # 初始化结果 - 使用分子为单位的数据结构，现在包含atom indices
        results = {
            'frame': frame_idx,
            'tio2_surface_h': {'count': 0, 'molecules': []},      # A: TiO2表面吸附的H，每个元素是单个H原子
            'solution_surface_oh': {'count': 0, 'molecules': []}, # B: 溶液来源的表面OH，每个元素是[O, H]
            'solution_surface_h2o': {'count': 0, 'molecules': []},  # C: 溶液来源的表面H2O，每个元素是[O, H, H]
            'solution_bulk_oh': {'count': 0, 'molecules': []},    # D: 溶液中的OH，每个元素是[O, H]
            'solution_bulk_h3o': {'count': 0, 'molecules': []},   # E: 溶液中的H3O，每个元素是[O, H, H, H]
            'na_ions': {'count': 0, 'coords': [], 'symbols': [], 'indices': []},             # Na离子
            'cl_ions': {'count': 0, 'coords': [], 'symbols': [], 'indices': []},             # Cl离子
            'tio2_surface_o': {'count': 0, 'z_coords': [], 'avg_z': 0.0}      # TiO2表面吸附的O原子
        }
        
        # 分析每个O原子
        for i, o_idx in enumerate(o_indices):
            h_count = o_h_counts.get(i, 0)  # 使用get方法，默认值为0
            is_top_surface = top_surface_mask[i]
            
            # 获取O原子坐标
            o_coord = positions[o_idx]
            
            # 如果是TiO2中的O原子
            if o_idx < min_ti_index:
                # TiO2表面吸附的O原子 (h_count=0)
                if is_top_surface and h_count == 0:
                    results['tio2_surface_o']['count'] += 1
                    results['tio2_surface_o']['z_coords'].append(o_coord[2])
                
                # A: TiO2表面吸附的H (原来的OH，现在标记为H)
                elif is_top_surface and h_count == 1:
                    results['tio2_surface_h']['count'] += 1
                    # 找到与这个O键合的H原子
                    if i in o_h_bonds:
                        for h_idx in o_h_bonds[i]:
                            h_pos = h_positions[h_idx]
                            # 存储单个H原子的信息，包含原子index
                            h_molecule = {
                                'coords': [h_pos],
                                'symbols': ['H'],
                                'indices': [h_indices[h_idx]]  # 记录原始H原子index
                            }
                            results['tio2_surface_h']['molecules'].append(h_molecule)
                            break
                    
            # 如果是溶液中的O原子
            elif o_idx > max_ti_index:
                if is_top_surface:
                    if h_count == 1:
                        # B: 溶液来源的表面OH
                        if i in o_h_bonds and len(o_h_bonds[i]) == 1:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # 使用验证函数创建OH分子
                            oh_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 1, oh_cutoff)
                            if oh_molecule is not None:
                                # 添加原子indices信息
                                oh_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_surface_oh']['count'] += 1
                                results['solution_surface_oh']['molecules'].append(oh_molecule)
                            else:
                                logging.warning(f"第{frame_idx}帧: 表面OH分子创建失败（O索引：{o_idx}）")
                    elif h_count == 2:
                        # C: 溶液来源的表面H2O
                        if i in o_h_bonds and len(o_h_bonds[i]) == 2:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # 创建H2O分子
                            h2o_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 2, oh_cutoff)
                            if h2o_molecule is not None:
                                # 添加原子indices信息
                                h2o_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_surface_h2o']['count'] += 1
                                results['solution_surface_h2o']['molecules'].append(h2o_molecule)
                            else:
                                logging.warning(f"第{frame_idx}帧: 表面H2O分子创建失败（O索引：{o_idx}）")
                else:
                    # 体相物种
                    if h_count == 1:
                        # D: 溶液中的OH
                        if i in o_h_bonds and len(o_h_bonds[i]) == 1:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # 使用验证函数创建OH分子
                            oh_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 1, oh_cutoff)
                            if oh_molecule is not None:
                                # 添加原子indices信息
                                oh_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_bulk_oh']['count'] += 1
                                results['solution_bulk_oh']['molecules'].append(oh_molecule)
                            else:
                                logging.warning(f"第{frame_idx}帧: 体相OH分子创建失败（O索引：{o_idx}）")
                    elif h_count == 3:
                        # E: 溶液中的H3O
                        if i in o_h_bonds and len(o_h_bonds[i]) == 3:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # 使用验证函数创建H3O分子
                            h3o_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 3, oh_cutoff)
                            if h3o_molecule is not None:
                                # 添加原子indices信息
                                h3o_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_bulk_h3o']['count'] += 1
                                results['solution_bulk_h3o']['molecules'].append(h3o_molecule)
                            else:
                                logging.warning(f"第{frame_idx}帧: H3O分子创建失败（O索引：{o_idx}）")
        
        # 记录Na离子坐标
        if na_indices:
            for na_idx in na_indices:
                na_coord = positions[na_idx]
                results['na_ions']['count'] += 1
                results['na_ions']['coords'].append(na_coord)
                results['na_ions']['symbols'].append('Na')
                results['na_ions']['indices'].append(na_idx)
        
        # 记录Cl离子坐标
        if cl_indices:
            for cl_idx in cl_indices:
                cl_coord = positions[cl_idx]
                results['cl_ions']['count'] += 1
                results['cl_ions']['coords'].append(cl_coord)
                results['cl_ions']['symbols'].append('Cl')
                results['cl_ions']['indices'].append(cl_idx)
        
        # 计算TiO2表面吸附O原子的平均z坐标
        if results['tio2_surface_o']['z_coords']:
            results['tio2_surface_o']['avg_z'] = np.mean(results['tio2_surface_o']['z_coords'])
        
        return results
        
    except Exception as e:
        logging.error(f"分析第{frame_idx}帧时出错: {str(e)}")
        return None

def save_frame_coordinates(results, frame_idx, output_dir, box_info=None, is_first_frame=False):
    """保存每一帧的坐标文件到统一的xyz文件中，确保分子完整性，包含原子indices"""
    
    def extract_molecular_coords_and_symbols(molecules, expected_atoms_per_molecule):
        """从分子列表中提取坐标、符号和indices，并验证完整性和键合距离"""
        all_coords = []
        all_symbols = []
        all_indices = []
        valid_molecules = 0
        invalid_molecules = 0
        
        for molecule in molecules:
            mol_coords = molecule['coords']
            mol_symbols = molecule['symbols']
            mol_indices = molecule.get('indices', None)
            
            # 验证分子原子数
            if len(mol_coords) != expected_atoms_per_molecule or len(mol_symbols) != expected_atoms_per_molecule:
                logging.warning(f"分子原子数不匹配: 期望{expected_atoms_per_molecule}个原子，实际{len(mol_coords)}个坐标，{len(mol_symbols)}个符号")
                invalid_molecules += 1
                continue
            
            # 验证indices数量
            if mol_indices is not None and len(mol_indices) != expected_atoms_per_molecule:
                logging.warning(f"分子indices数量不匹配: 期望{expected_atoms_per_molecule}个，实际{len(mol_indices)}个")
                invalid_molecules += 1
                continue
            
            # 验证符号序列的正确性
            if expected_atoms_per_molecule == 1:  # H原子
                if mol_symbols != ['H']:
                    logging.warning(f"H分子符号序列错误: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 2:  # OH分子
                if mol_symbols != ['O', 'H']:
                    logging.warning(f"OH分子符号序列错误: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 3:  # H2O分子
                if mol_symbols != ['O', 'H', 'H']:
                    logging.warning(f"H2O分子符号序列错误: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 4:  # H3O分子
                if mol_symbols != ['O', 'H', 'H', 'H']:
                    logging.warning(f"H3O分子符号序列错误: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            
            # 对于多原子分子，验证键合距离
            if expected_atoms_per_molecule > 1:
                is_valid, message = verify_bonding_distances(mol_coords, mol_symbols)
                if not is_valid:
                    logging.warning(f"分子键合验证失败: {message}")
                    invalid_molecules += 1
                    continue
            
            # 通过所有验证的分子
            all_coords.extend(mol_coords)
            all_symbols.extend(mol_symbols)
            if mol_indices is not None:
                all_indices.extend(mol_indices)
            valid_molecules += 1
        
        # 验证总原子数是否为期望值的倍数
        if len(all_coords) % expected_atoms_per_molecule != 0:
            logging.warning(f"总原子数{len(all_coords)}不是{expected_atoms_per_molecule}的倍数")
        
        if invalid_molecules > 0:
            logging.info(f"分子验证结果: 有效{valid_molecules}个，无效{invalid_molecules}个")
        
        return all_coords, all_symbols, all_indices
    
    # A: TiO2表面吸附的H
    if results['tio2_surface_h']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['tio2_surface_h']['molecules'], 1)
        if coords:
            filename = os.path.join(output_dir, "tio2_surface_h.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
    
    # B: 溶液来源的表面OH
    if results['solution_surface_oh']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_surface_oh']['molecules'], 2)
        if coords and len(coords) % 2 == 0:  # 确保原子数是2的倍数
            filename = os.path.join(output_dir, "solution_surface_oh.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"第{frame_idx}帧: 表面OH总原子数({len(coords)})不是2的倍数，跳过写入")
    
    # C: 溶液来源的表面H2O
    if results['solution_surface_h2o']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_surface_h2o']['molecules'], 3)
        if coords and len(coords) % 3 == 0:  # 确保原子数是3的倍数
            filename = os.path.join(output_dir, "solution_surface_h2o.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"第{frame_idx}帧: 表面H2O总原子数({len(coords)})不是3的倍数，跳过写入")
    
    # D: 溶液中的OH
    if results['solution_bulk_oh']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_bulk_oh']['molecules'], 2)
        if coords and len(coords) % 2 == 0:  # 确保原子数是2的倍数
            filename = os.path.join(output_dir, "solution_bulk_oh.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"第{frame_idx}帧: 体相OH总原子数({len(coords)})不是2的倍数，跳过写入")
    
    # E: 溶液中的H3O
    if results['solution_bulk_h3o']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_bulk_h3o']['molecules'], 4)
        if coords and len(coords) % 4 == 0:  # 确保原子数是4的倍数
            filename = os.path.join(output_dir, "solution_bulk_h3o.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"第{frame_idx}帧: H3O总原子数({len(coords)})不是4的倍数，跳过写入")
    
    # Na离子
    if results['na_ions']['count'] > 0:
        filename = os.path.join(output_dir, "na_ions.xyz")
        if is_first_frame:
            write_coordinates_to_xyz(
                results['na_ions']['coords'],
                results['na_ions']['symbols'],
                filename, frame_idx, box_info, append_mode=False,
                atom_indices=results['na_ions']['indices']
            )
        else:
            append_coordinates_to_xyz(
                results['na_ions']['coords'],
                results['na_ions']['symbols'],
                filename, frame_idx, box_info,
                atom_indices=results['na_ions']['indices']
            )
    
    # Cl离子
    if results['cl_ions']['count'] > 0:
        filename = os.path.join(output_dir, "cl_ions.xyz")
        if is_first_frame:
            write_coordinates_to_xyz(
                results['cl_ions']['coords'],
                results['cl_ions']['symbols'],
                filename, frame_idx, box_info, append_mode=False,
                atom_indices=results['cl_ions']['indices']
            )
        else:
            append_coordinates_to_xyz(
                results['cl_ions']['coords'],
                results['cl_ions']['symbols'],
                filename, frame_idx, box_info,
                atom_indices=results['cl_ions']['indices']
            )

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
        'text.usetex': False,  # 使用matplotlib的内置数学渲染而非LaTeX
        'mathtext.default': 'regular'  # 确保数学文本使用常规字体
    })

def plot_species_evolution(all_results, output_file='ion_species_evolution.png'):
    """绘制物种数量随时间的演化图 - Nature风格"""
    
    # 设置Nature风格
    setup_nature_style()
    
    frames = [r['frame'] for r in all_results]
    
    tio2_surface_h = [r['tio2_surface_h']['count'] for r in all_results]
    solution_surface_oh = [r['solution_surface_oh']['count'] for r in all_results]
    solution_bulk_oh = [r['solution_bulk_oh']['count'] for r in all_results]
    solution_bulk_h3o = [r['solution_bulk_h3o']['count'] for r in all_results]
    na_ions = [r['na_ions']['count'] for r in all_results]
    cl_ions = [r['cl_ions']['count'] for r in all_results]
    
    # Nature风格配色方案
    colors = {
        'tio2_h': '#1f77b4',      # 蓝色
        'surface_oh': '#ff7f0e',   # 橙色  
        'bulk_oh': '#2ca02c',      # 绿色
        'bulk_h3o': '#d62728',     # 红色
        'na': '#9467bd',           # 紫色
        'cl': '#8c564b'            # 棕色
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 第一个子图：主要离子物种
    ax1 = axes[0]
    ax1.plot(frames, tio2_surface_h, 'o-', label=r'TiO$_2$ Surface Adsorbed H$^+$', 
             color=colors['tio2_h'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_surface_oh, 's-', label=r'Surface Adsorbed OH$^-$', 
             color=colors['surface_oh'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_bulk_oh, '^-', label=r'Bulk OH$^-$', 
             color=colors['bulk_oh'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_bulk_h3o, 'v-', label=r'Bulk H$_3$O$^+$', 
             color=colors['bulk_h3o'], linewidth=2, markersize=5)
    
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Species Count')
    ax1.set_title('Ion Species Evolution Over Time')
    ax1.legend(frameon=False, loc='best')
    
    # 第二个子图：Na和Cl离子（如果存在）
    ax2 = axes[1]
    has_ions = False
    
    if max(na_ions) > 0:
        ax2.plot(frames, na_ions, 'o-', label=r'Na$^+$ ions', 
                color=colors['na'], linewidth=2, markersize=5)
        has_ions = True
    
    if max(cl_ions) > 0:
        ax2.plot(frames, cl_ions, 's-', label=r'Cl$^-$ ions', 
                color=colors['cl'], linewidth=2, markersize=5)
        has_ions = True
    
    if has_ions:
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Ion Count')
        ax2.set_title('Salt Ions Distribution')
        ax2.legend(frameon=False, loc='best')
    else:
        ax2.text(0.5, 0.5, r'No Na$^+$ or Cl$^-$ ions detected', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=14, style='italic')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Ion Count')
        ax2.set_title('Salt Ions Distribution')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Species evolution plot saved as: {output_file}")

def build_species_transition_matrix(all_results):
    """
    构建物质转移矩阵，追踪五种物质之间的转换
    主要基于O原子的index追踪（因为H可能通过Grotthuss机理转移）
    
    五种物质：
    0: solution_bulk_h3o (体相H3O+)
    1: solution_bulk_oh (体相OH-)
    2: solution_surface_oh (表面吸附OH-)
    3: solution_surface_h2o (表面吸附H2O)
    4: tio2_surface_h (TiO2表面吸附H+, 虽然没有O，但可以通过H的index追踪)
    """
    
    species_names = ['solution_bulk_h3o', 'solution_bulk_oh', 'solution_surface_oh', 
                     'solution_surface_h2o', 'tio2_surface_h']
    n_species = len(species_names)
    
    # 初始化转移计数矩阵
    transition_counts = np.zeros((n_species, n_species), dtype=int)
    
    # 用于存储详细的转移信息
    transition_details = []
    
    for frame_idx in range(len(all_results) - 1):
        current_frame = all_results[frame_idx]
        next_frame = all_results[frame_idx + 1]
        
        # 为当前帧构建O原子index到物质类型的映射
        current_o_to_species = {}
        # 为当前帧构建H原子index到物质类型的映射（用于tio2_surface_h）
        current_h_to_species = {}
        
        # 处理当前帧的每种物质
        for species_idx, species_name in enumerate(species_names):
            molecules = current_frame[species_name].get('molecules', [])
            for mol in molecules:
                indices = mol.get('indices', [])
                if species_name == 'tio2_surface_h':
                    # tio2_surface_h只有H原子
                    if indices:
                        current_h_to_species[indices[0]] = species_idx
                else:
                    # 其他物质有O原子，O原子总是第一个
                    if indices:
                        o_idx = indices[0]
                        current_o_to_species[o_idx] = species_idx
        
        # 为下一帧构建O原子index到物质类型的映射
        next_o_to_species = {}
        next_h_to_species = {}
        
        for species_idx, species_name in enumerate(species_names):
            molecules = next_frame[species_name].get('molecules', [])
            for mol in molecules:
                indices = mol.get('indices', [])
                if species_name == 'tio2_surface_h':
                    if indices:
                        next_h_to_species[indices[0]] = species_idx
                else:
                    if indices:
                        o_idx = indices[0]
                        next_o_to_species[o_idx] = species_idx
        
        # 统计O原子的转移（基于O原子的index）
        for o_idx, from_species in current_o_to_species.items():
            if o_idx in next_o_to_species:
                to_species = next_o_to_species[o_idx]
                transition_counts[from_species, to_species] += 1
                
                # 记录详细转移信息
                if from_species != to_species:
                    transition_details.append({
                        'frame': current_frame['frame'],
                        'o_index': o_idx,
                        'from': species_names[from_species],
                        'to': species_names[to_species]
                    })
        
        # 统计H原子的转移（仅用于tio2_surface_h）
        for h_idx, from_species in current_h_to_species.items():
            if h_idx in next_h_to_species:
                to_species = next_h_to_species[h_idx]
                transition_counts[from_species, to_species] += 1
                
                if from_species != to_species:
                    transition_details.append({
                        'frame': current_frame['frame'],
                        'h_index': h_idx,
                        'from': species_names[from_species],
                        'to': species_names[to_species]
                    })
    
    return transition_counts, species_names, transition_details

def plot_transition_matrix(transition_counts, species_names, output_file='species_transition_matrix.png'):
    """绘制物质转移矩阵热图"""
    
    setup_nature_style()
    
    # 计算转移概率矩阵
    # 每一行的和代表该物质的总转移次数
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # 避免除以0
    row_sums[row_sums == 0] = 1
    transition_prob = transition_counts / row_sums
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 简化物质名称用于显示
    display_names = [
        r'Bulk H$_3$O$^+$',
        r'Bulk OH$^-$',
        r'Surface OH$^-$',
        r'Surface H$_2$O',
        r'Surface H$^+$'
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
                          ha="center", va="center", color="black" if transition_prob[i, j] < 0.5 else "white",
                          fontsize=10)
    
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
            text = ax2.text(j, i, f'{transition_prob[i, j]:.2f}',
                          ha="center", va="center", color="black" if transition_prob[i, j] < 0.5 else "white",
                          fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Probability', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Transition matrix plot saved as: {output_file}")

def save_transition_data(transition_counts, species_names, transition_details, output_dir):
    """保存转移矩阵数据和详细转移信息"""
    
    # 保存转移计数矩阵
    counts_file = os.path.join(output_dir, "transition_counts.txt")
    with open(counts_file, 'w') as f:
        f.write("Transition Counts Matrix\n")
        f.write("From \\ To\t" + "\t".join(species_names) + "\n")
        for i, from_species in enumerate(species_names):
            f.write(f"{from_species}\t")
            f.write("\t".join(str(int(transition_counts[i, j])) for j in range(len(species_names))))
            f.write("\n")
    
    logging.info(f"Transition counts saved to: {counts_file}")
    
    # 计算并保存转移概率矩阵
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
    
    logging.info(f"Transition probabilities saved to: {prob_file}")
    
    # 保存详细转移信息
    details_file = os.path.join(output_dir, "transition_details.txt")
    with open(details_file, 'w') as f:
        f.write("Frame\tAtom_Index\tFrom_Species\tTo_Species\n")
        for detail in transition_details:
            atom_idx = detail.get('o_index', detail.get('h_index', 'N/A'))
            f.write(f"{detail['frame']}\t{atom_idx}\t{detail['from']}\t{detail['to']}\n")
    
    logging.info(f"Transition details saved to: {details_file}")
    
    # 保存转移统计摘要
    summary_file = os.path.join(output_dir, "transition_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Species Transition Summary ===\n\n")
        
        # 总转移次数
        total_transitions = np.sum(transition_counts) - np.trace(transition_counts)
        f.write(f"Total transitions (excluding self-transitions): {int(total_transitions)}\n\n")
        
        # 每种物质的转移统计
        for i, species in enumerate(species_names):
            f.write(f"\n{species}:\n")
            total_from = np.sum(transition_counts[i, :])
            total_to = np.sum(transition_counts[:, i])
            f.write(f"  Total transitions from this species: {int(total_from)}\n")
            f.write(f"  Total transitions to this species: {int(total_to)}\n")
            
            # 主要转移目标
            if total_from > 0:
                f.write(f"  Main transitions from {species}:\n")
                for j in range(len(species_names)):
                    if i != j and transition_counts[i, j] > 0:
                        prob = transition_prob[i, j]
                        f.write(f"    -> {species_names[j]}: {int(transition_counts[i, j])} ({prob:.2%})\n")
    
    logging.info(f"Transition summary saved to: {summary_file}")

def main():
    args = get_args()
    
    # 设置日志
    setup_logging(enable_log_file=args.enable_log_file)
    
    start_time = datetime.now()
    logging.info(f"Starting ion species analysis, frame interval: {args.step_interval}")
    
    try:
        # 创建输出目录
        output_dir = "ion_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if args.format == 'lammps':
            # 读取LAMMPS文件
            logging.info(f"Reading LAMMPS files: {args.input}, {args.traj}")
            
            # 尝试读取MDAnalysis Universe
            if args.atom_style:
                u = mda.Universe(args.input, args.traj, 
                               atom_style=args.atom_style,
                               format='LAMMPSDUMP')
            else:
                try:
                    u = mda.Universe(args.input, args.traj, 
                                   atom_style='id type x y z',
                                   format='LAMMPSDUMP')
                except:
                    try:
                        u = mda.Universe(args.input, args.traj, 
                                       atom_style='atomic',
                                       format='LAMMPSDUMP')
                    except:
                        u = mda.Universe(args.input, args.traj, 
                                       atom_style='full',
                                       format='LAMMPSDUMP')
            
            total_frames = len(u.trajectory)
            logging.info(f"Total frames in trajectory: {total_frames}")
            
            # 确定分析的帧范围
            start_frame = args.start_frame
            end_frame = total_frames if args.end_frame == -1 else min(args.end_frame, total_frames)
            
            frames_to_analyze = list(range(start_frame, end_frame, args.step_interval))
            logging.info(f"Will analyze {len(frames_to_analyze)} frames")
            
            all_results = []
            
            # 分析每一帧
            for i, frame_idx in enumerate(frames_to_analyze):
                logging.info(f"Analyzing frame {frame_idx}...")
                
                atoms, box_dims, box_info = read_lammps_frame(u, frame_idx)
                if atoms is None:
                    continue
                
                results = analyze_frame_ion_species(
                    atoms, box_dims, frame_idx,
                    ti_o_cutoff=args.ti_o_cutoff,
                    oh_cutoff=args.oh_cutoff
                )
                
                if results is not None:
                    # 验证分子完整性
                    integrity_report = validate_molecular_integrity(results, frame_idx)
                    
                    all_results.append(results)
                    
                    # 保存坐标文件（第一帧创建新文件，后续帧追加）
                    is_first_frame = (i == 0)
                    save_frame_coordinates(results, frame_idx, output_dir, box_info, is_first_frame)
                    
                    # 输出当前帧的统计，包括分子完整性信息
                    logging.info(f"  Frame {frame_idx} results:")
                    logging.info(f"    TiO2 Surface Adsorbed H: {results['tio2_surface_h']['count']} (有效分子: {integrity_report['tio2_surface_h']['valid']})")
                    logging.info(f"    Surface Adsorbed OH (from solution): {results['solution_surface_oh']['count']} (有效分子: {integrity_report['solution_surface_oh']['valid']})")
                    logging.info(f"    Surface Adsorbed H2O (from solution): {results['solution_surface_h2o']['count']} (有效分子: {integrity_report['solution_surface_h2o']['valid']})")
                    logging.info(f"    Bulk OH in solution: {results['solution_bulk_oh']['count']} (有效分子: {integrity_report['solution_bulk_oh']['valid']})")
                    logging.info(f"    Bulk H3O in solution: {results['solution_bulk_h3o']['count']} (有效分子: {integrity_report['solution_bulk_h3o']['valid']})")
                    logging.info(f"    Na+ ions: {results['na_ions']['count']}")
                    logging.info(f"    Cl- ions: {results['cl_ions']['count']}")
                    logging.info(f"    TiO2 Surface Adsorbed O: {results['tio2_surface_o']['count']} (avg z: {results['tio2_surface_o']['avg_z']:.3f})")
                    
                    # 如果有无效分子，给出警告
                    total_invalid = (integrity_report['tio2_surface_h']['invalid'] + 
                                   integrity_report['solution_surface_oh']['invalid'] + 
                                   integrity_report['solution_surface_h2o']['invalid'] +
                                   integrity_report['solution_bulk_oh']['invalid'] + 
                                   integrity_report['solution_bulk_h3o']['invalid'])
                    if total_invalid > 0:
                        logging.warning(f"    第{frame_idx}帧发现{total_invalid}个不完整的分子")
            
            # 保存统计结果
            stats_file = os.path.join(output_dir, "species_statistics.txt")
            with open(stats_file, 'w') as f:
                f.write("Frame\tTiO2_Surface_H\tSolution_Surface_OH\tSolution_Surface_H2O\tSolution_Bulk_OH\tSolution_Bulk_H3O\tNa_Ions\tCl_Ions\tTiO2_Surface_O\tTiO2_Surface_O_Avg_Z\n")
                for result in all_results:
                    f.write(f"{result['frame']}\t{result['tio2_surface_h']['count']}\t"
                           f"{result['solution_surface_oh']['count']}\t{result['solution_surface_h2o']['count']}\t"
                           f"{result['solution_bulk_oh']['count']}\t{result['solution_bulk_h3o']['count']}\t"
                           f"{result['na_ions']['count']}\t{result['cl_ions']['count']}\t"
                           f"{result['tio2_surface_o']['count']}\t{result['tio2_surface_o']['avg_z']:.6f}\n")
            
            # 保存分子完整性统计
            integrity_file = os.path.join(output_dir, "molecular_integrity_statistics.txt")
            with open(integrity_file, 'w') as f:
                f.write("Frame\tTiO2_H_Valid\tTiO2_H_Invalid\tSurface_OH_Valid\tSurface_OH_Invalid\t"
                       f"Surface_H2O_Valid\tSurface_H2O_Invalid\t"
                       f"Bulk_OH_Valid\tBulk_OH_Invalid\tBulk_H3O_Valid\tBulk_H3O_Invalid\n")
                for i, result in enumerate(all_results):
                    integrity_report = validate_molecular_integrity(result, result['frame'])
                    f.write(f"{result['frame']}\t"
                           f"{integrity_report['tio2_surface_h']['valid']}\t{integrity_report['tio2_surface_h']['invalid']}\t"
                           f"{integrity_report['solution_surface_oh']['valid']}\t{integrity_report['solution_surface_oh']['invalid']}\t"
                           f"{integrity_report['solution_surface_h2o']['valid']}\t{integrity_report['solution_surface_h2o']['invalid']}\t"
                           f"{integrity_report['solution_bulk_oh']['valid']}\t{integrity_report['solution_bulk_oh']['invalid']}\t"
                           f"{integrity_report['solution_bulk_h3o']['valid']}\t{integrity_report['solution_bulk_h3o']['invalid']}\n")
            
            logging.info(f"Statistics saved to: {stats_file}")
            logging.info(f"Molecular integrity statistics saved to: {integrity_file}")
            
            # 保存TiO2表面吸附O原子的z坐标统计
            tio2_o_file = os.path.join(output_dir, "tio2_surface_o_z_coordinates.txt")
            with open(tio2_o_file, 'w') as f:
                f.write("Frame\tTiO2_Surface_O_Count\tAverage_Z_Coordinate\n")
                for result in all_results:
                    f.write(f"{result['frame']}\t{result['tio2_surface_o']['count']}\t"
                           f"{result['tio2_surface_o']['avg_z']:.6f}\n")
            
            logging.info(f"TiO2 surface O z-coordinates saved to: {tio2_o_file}")
            
            # 绘制演化图
            if len(all_results) > 1:
                evolution_plot = os.path.join(output_dir, "ion_species_evolution.png")
                plot_species_evolution(all_results, evolution_plot)
                
                # 计算并可视化转移矩阵
                logging.info("\n=== Computing Species Transition Matrix ===")
                transition_counts, species_names, transition_details = build_species_transition_matrix(all_results)
                
                # 绘制转移矩阵
                transition_plot = os.path.join(output_dir, "species_transition_matrix.png")
                plot_transition_matrix(transition_counts, species_names, transition_plot)
                
                # 保存转移矩阵数据
                save_transition_data(transition_counts, species_names, transition_details, output_dir)
            
            # 计算总体分子完整性统计
            total_integrity_stats = {
                'tio2_h': {'valid': 0, 'invalid': 0},
                'surface_oh': {'valid': 0, 'invalid': 0},
                'surface_h2o': {'valid': 0, 'invalid': 0},
                'bulk_oh': {'valid': 0, 'invalid': 0},
                'bulk_h3o': {'valid': 0, 'invalid': 0}
            }
            
            for result in all_results:
                integrity_report = validate_molecular_integrity(result, result['frame'])
                total_integrity_stats['tio2_h']['valid'] += integrity_report['tio2_surface_h']['valid']
                total_integrity_stats['tio2_h']['invalid'] += integrity_report['tio2_surface_h']['invalid']
                total_integrity_stats['surface_oh']['valid'] += integrity_report['solution_surface_oh']['valid']
                total_integrity_stats['surface_oh']['invalid'] += integrity_report['solution_surface_oh']['invalid']
                total_integrity_stats['surface_h2o']['valid'] += integrity_report['solution_surface_h2o']['valid']
                total_integrity_stats['surface_h2o']['invalid'] += integrity_report['solution_surface_h2o']['invalid']
                total_integrity_stats['bulk_oh']['valid'] += integrity_report['solution_bulk_oh']['valid']
                total_integrity_stats['bulk_oh']['invalid'] += integrity_report['solution_bulk_oh']['invalid']
                total_integrity_stats['bulk_h3o']['valid'] += integrity_report['solution_bulk_h3o']['valid']
                total_integrity_stats['bulk_h3o']['invalid'] += integrity_report['solution_bulk_h3o']['invalid']
            
            # 输出总体统计
            logging.info("\n=== Overall Statistics ===")
            avg_tio2_h = np.mean([r['tio2_surface_h']['count'] for r in all_results])
            avg_surface_oh = np.mean([r['solution_surface_oh']['count'] for r in all_results])
            avg_surface_h2o = np.mean([r['solution_surface_h2o']['count'] for r in all_results])
            avg_bulk_oh = np.mean([r['solution_bulk_oh']['count'] for r in all_results])
            avg_bulk_h3o = np.mean([r['solution_bulk_h3o']['count'] for r in all_results])
            avg_na_ions = np.mean([r['na_ions']['count'] for r in all_results])
            avg_cl_ions = np.mean([r['cl_ions']['count'] for r in all_results])
            avg_tio2_surface_o = np.mean([r['tio2_surface_o']['count'] for r in all_results])
            avg_tio2_surface_o_z = np.mean([r['tio2_surface_o']['avg_z'] for r in all_results if r['tio2_surface_o']['avg_z'] > 0])
            
            logging.info(f"TiO2 Surface Adsorbed H - Average count: {avg_tio2_h:.2f} (有效分子总数: {total_integrity_stats['tio2_h']['valid']}, 无效: {total_integrity_stats['tio2_h']['invalid']})")
            logging.info(f"Surface Adsorbed OH (from solution) - Average count: {avg_surface_oh:.2f} (有效分子总数: {total_integrity_stats['surface_oh']['valid']}, 无效: {total_integrity_stats['surface_oh']['invalid']})")
            logging.info(f"Surface Adsorbed H2O (from solution) - Average count: {avg_surface_h2o:.2f} (有效分子总数: {total_integrity_stats['surface_h2o']['valid']}, 无效: {total_integrity_stats['surface_h2o']['invalid']})")
            logging.info(f"Bulk OH in solution - Average count: {avg_bulk_oh:.2f} (有效分子总数: {total_integrity_stats['bulk_oh']['valid']}, 无效: {total_integrity_stats['bulk_oh']['invalid']})")
            logging.info(f"Bulk H3O in solution - Average count: {avg_bulk_h3o:.2f} (有效分子总数: {total_integrity_stats['bulk_h3o']['valid']}, 无效: {total_integrity_stats['bulk_h3o']['invalid']})")
            logging.info(f"Na+ ions - Average count: {avg_na_ions:.2f}")
            logging.info(f"Cl- ions - Average count: {avg_cl_ions:.2f}")
            logging.info(f"TiO2 Surface Adsorbed O - Average count: {avg_tio2_surface_o:.2f}")
            if not np.isnan(avg_tio2_surface_o_z):
                logging.info(f"TiO2 Surface Adsorbed O - Average Z coordinate: {avg_tio2_surface_o_z:.3f} Å")
            
            # 计算并报告总体数据质量
            total_valid = sum(stats['valid'] for stats in total_integrity_stats.values())
            total_invalid = sum(stats['invalid'] for stats in total_integrity_stats.values())
            if total_valid + total_invalid > 0:
                quality_percentage = total_valid / (total_valid + total_invalid) * 100
                logging.info(f"\n=== 数据质量报告 ===")
                logging.info(f"总分子数: {total_valid + total_invalid}, 有效分子: {total_valid}, 无效分子: {total_invalid}")
                logging.info(f"数据质量: {quality_percentage:.1f}%")
            
        else:
            logging.error("Currently only supports LAMMPS format for multi-frame analysis")
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nAnalysis completed! Total time: {duration}")
        logging.info(f"Results saved in directory: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 