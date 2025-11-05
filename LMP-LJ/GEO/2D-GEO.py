#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D网格系统生成器 - 用于LAMMPS分子动力学模拟
====================================================
本程序用于生成油-气-水系统的初始配置文件
生成两种格式的输出文件：
1. LAMMPS data 格式 (.dat)
2. XYZ 格式 (.xyz)

原子类型定义（共6种）：
-----------------------
类型1 (sub):      底层固定层 - 用于底部边界条件
类型2 (water):    水相 - 主体水相原子
类型3 (gas):      气相 - 气泡区域原子
类型4 (oil-head): 油相头部 - 油分子的亲水端
类型5 (oil-tail): 油相尾部 - 油分子的疏水端
类型6 (top):      顶层固定层 - 用于顶部边界条件

作者：改编自 C++ 版本
日期：2025-11-04
"""

from typing import Tuple, List


class GridSystemGenerator:
    """3D网格系统生成器类"""
    
    def __init__(self, x_size: int = 101, y_size: int = 5, z_size: int = 81, 
                 top_layer_offset: int = 10,
                 gas_x_start: int = 21, gas_x_width: int = 10,
                 oil_x_start: int = 61, oil_x_width: int = 20,
                 mixed_phase_z_start: int = 3, mixed_phase_z_thickness: int = 6):
        """
        初始化网格系统
        
        参数说明：
        ----------
        x_size : int, default=101
            X方向的网格数（建议范围：10-200）
        y_size : int, default=5
            Y方向的网格数（建议范围：3-50）
        z_size : int, default=81
            Z方向的网格数（建议范围：10-200）
        top_layer_offset : int, default=10
            顶层固定层相对于Z_SIZE的向下偏移量
            例如：Z_SIZE=81, offset=10 时，顶层在 z=69-70
                 Z_SIZE=81, offset=5 时，顶层在 z=74-75
            建议范围：5-20
        gas_x_start : int, default=21
            气相(gas)在X方向的起始位置
        gas_x_width : int, default=10
            气相(gas)在X方向的宽度
        oil_x_start : int, default=61
            油相(oil)在X方向的起始位置
        oil_x_width : int, default=20
            油相(oil)在X方向的宽度
        mixed_phase_z_start : int, default=3
            混合相区域(gas+oil+water)在Z方向的起始位置
        mixed_phase_z_thickness : int, default=6
            混合相区域(gas+oil+water)在Z方向的厚度
        """
        self.X = x_size
        self.Y = y_size
        self.Z = z_size
        self.top_offset = top_layer_offset
        
        # Gas和Oil的位置参数
        self.gas_x_start = gas_x_start
        self.gas_x_width = gas_x_width
        self.oil_x_start = oil_x_start
        self.oil_x_width = oil_x_width
        self.mixed_z_start = mixed_phase_z_start
        self.mixed_z_thickness = mixed_phase_z_thickness
        
        # 初始化3D数组：存储原子类型（使用嵌套列表）
        self.ntype = [[[0 for _ in range(x_size)] for _ in range(y_size)] for _ in range(z_size)]
        
        # 初始化3D数组：存储原子ID（使用嵌套列表）
        self.id = [[[0 for _ in range(x_size)] for _ in range(y_size)] for _ in range(z_size)]
        
        # 存储键连接信息 [键ID, [原子1_ID, 原子2_ID]]
        self.bonds = []
        
        # 原子总数
        self.total_atoms = 0
        
    def define_regions(self, regions: List[dict] = None):
        """
        定义不同区域的原子类型
        
        参数说明：
        ----------
        regions : List[dict], optional
            区域定义列表，每个字典包含：
            - 'z_range': (z_min, z_max)
            - 'x_range': (x_min, x_max)
            - 'y_range': (y_min, y_max)
            - 'atom_type': int (原子类型编号)
            
            如果不提供，将使用默认配置（原C++程序的配置）
        """
        if regions is None:
            # 使用默认区域配置（参数可调整）
            # 系统从下到上分为：底层固定层 -> 混合相区域 -> 主体水相 -> 顶层固定层
            
            # 计算混合相区域的Z范围
            mixed_z_end = self.mixed_z_start + self.mixed_z_thickness - 1
            
            # 计算gas和oil的X范围
            gas_x_end = self.gas_x_start + self.gas_x_width - 1
            oil_x_end = self.oil_x_start + self.oil_x_width - 1
            
            # Gas和Oil之间的间距
            gap_between_gas_oil = self.oil_x_start - gas_x_end - 1
            
            regions = [
                # 区域1：底层固定层 (z=1-2) - 类型1 (sub)
                {'z_range': (1, 2), 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 1},
                
                # 区域2：混合相区域 - 左侧水相 (gas之前) - 类型2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (1, self.gas_x_start - 1), 
                 'y_range': (1, self.Y-1), 'atom_type': 2},
                
                # 区域3：混合相区域 - 气泡区 (gas) - 类型3 (gas)
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (self.gas_x_start, gas_x_end), 
                 'y_range': (1, self.Y-1), 'atom_type': 3},
                
                # 区域4：混合相区域 - 中部水相 (gas和oil之间) - 类型2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (gas_x_end + 1, self.oil_x_start - 1), 
                 'y_range': (1, self.Y-1), 'atom_type': 2},
                
                # 区域5：混合相区域 - 油相头部层 (oil-head, y=1) - 类型4 (oil-head)
                # 油分子的亲水端，与水相接触
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (self.oil_x_start, oil_x_end), 
                 'y_range': (1, 1), 'atom_type': 4},
                
                # 区域6：混合相区域 - 油相尾部层 (oil-tail, y=2-4) - 类型5 (oil-tail)
                # 油分子的疏水端，形成油滴内部
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (self.oil_x_start, oil_x_end), 
                 'y_range': (2, self.Y-1), 'atom_type': 5},
                
                # 区域7：混合相区域 - 右侧水相 (oil之后) - 类型2 (water)
                {'z_range': (self.mixed_z_start, mixed_z_end), 
                 'x_range': (oil_x_end + 1, self.X-1), 
                 'y_range': (1, self.Y-1), 'atom_type': 2},
                
                # 区域8：主体水相 (z=9-50) - 类型2 (water)
                {'z_range': (9, 50), 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 2},
                
                # 注意：z=51到顶层之间为空白区域（无原子）
                
                # 区域9：顶层固定层 - 类型6 (top)
                # 位置由 top_layer_offset 参数控制
                {'z_range': (self.Z - self.top_offset - 2, self.Z - self.top_offset - 1), 
                 'x_range': (1, self.X-1), 'y_range': (1, self.Y-1), 'atom_type': 6},
            ]
        
        # 遍历所有网格点，分配原子类型
        for region in regions:
            z_min, z_max = region['z_range']
            x_min, x_max = region['x_range']
            y_min, y_max = region['y_range']
            atom_type = region['atom_type']
            
            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if 0 <= z < self.Z and 0 <= y < self.Y and 0 <= x < self.X:
                            # 只在该位置尚未分配原子时才计数和分配ID
                            if self.ntype[z][y][x] == 0:
                                self.total_atoms += 1
                                self.id[z][y][x] = self.total_atoms
                            # 更新原子类型（允许覆盖）
                            self.ntype[z][y][x] = atom_type
    
    def create_bonds(self, bond_regions: List[dict] = None):
        """
        创建原子间的键连接
        
        参数说明：
        ----------
        bond_regions : List[dict], optional
            键连接区域定义，每个字典包含：
            - 'z_range': (z_min, z_max)
            - 'x_range': (x_min, x_max)
            - 'y_range': (y_min, y_max)
            - 'direction': 'x', 'y', 或 'z' (沿哪个方向连接)
            
            如果不提供，将使用默认配置（原C++程序的配置）
        """
        if bond_regions is None:
            # 默认配置：在油相区域沿Y方向创建键
            # 根据参数动态确定油相位置
            mixed_z_end = self.mixed_z_start + self.mixed_z_thickness - 1
            oil_x_end = self.oil_x_start + self.oil_x_width - 1
            
            for z in range(self.mixed_z_start, mixed_z_end + 1):
                for x in range(self.oil_x_start, oil_x_end + 1):
                    for y in range(1, min(4, self.Y)):  # y=1 to 3 (或Y-1)
                        if y + 1 < self.Y:
                            atom1_id = self.id[z][y][x]
                            atom2_id = self.id[z][y+1][x]
                            if atom1_id > 0 and atom2_id > 0:
                                self.bonds.append([atom1_id, atom2_id])
        else:
            # 使用自定义键连接配置
            for bond_region in bond_regions:
                z_min, z_max = bond_region['z_range']
                x_min, x_max = bond_region['x_range']
                y_min, y_max = bond_region['y_range']
                direction = bond_region.get('direction', 'y')
                
                for z in range(z_min, z_max + 1):
                    for x in range(x_min, x_max + 1):
                        for y in range(y_min, y_max + 1):
                            atom1_id = self.id[z][y][x]
                            if atom1_id > 0:
                                atom2_id = 0
                                if direction == 'x' and x + 1 < self.X:
                                    atom2_id = self.id[z][y][x+1]
                                elif direction == 'y' and y + 1 < self.Y:
                                    atom2_id = self.id[z][y+1][x]
                                elif direction == 'z' and z + 1 < self.Z:
                                    atom2_id = self.id[z+1][y][x]
                                
                                if atom2_id > 0:
                                    self.bonds.append([atom1_id, atom2_id])
    
    def write_lammps_data(self, filename: str = "oil_bubble_2.dat", num_atom_types: int = 7):
        """
        写入LAMMPS data格式文件
        
        参数说明：
        ----------
        filename : str, default="oil_bubble_2.dat"
            输出文件名
        num_atom_types : int, default=7
            原子类型总数
        """
        with open(filename, 'w') as f:
            # 文件头
            f.write("liquid data\n")
            f.write(f"\n{self.total_atoms} atoms\n")
            f.write(f"{len(self.bonds)} bonds\n")
            f.write(f"\n{num_atom_types} atom types\n")
            f.write("\n1 bond types\n")
            
            # 盒子尺寸
            f.write(f"0 {self.X - 1} xlo xhi\n")
            f.write(f"0 {self.Y - 1} ylo yhi\n")
            f.write(f"0 {self.Z - 1} zlo zhi\n")
            
            # 原子信息
            f.write("\n\n\nAtoms\n")
            for z in range(1, self.Z):
                for y in range(1, self.Y):
                    for x in range(1, self.X):
                        if self.ntype[z][y][x] > 0 and self.id[z][y][x] >= 1:
                            atom_id = self.id[z][y][x]
                            atom_type = self.ntype[z][y][x]
                            # 坐标（转换为浮点数）
                            coord_x = float(x)
                            coord_y = float(y)
                            coord_z = float(z)
                            # 格式：原子ID  分子ID  原子类型  x  y  z
                            f.write(f"\n{atom_id:4d}       {1:4d}     {atom_type:4d}     "
                                   f"{coord_x:8.5f}     {coord_y:8.5f}   {coord_z:8.5f}")
            
            # 键信息
            if len(self.bonds) > 0:
                f.write("\n\nBonds\n")
                for bond_id, (atom1, atom2) in enumerate(self.bonds, start=1):
                    # 格式：键ID  键类型  原子1  原子2
                    f.write(f"\n{bond_id:4d}   {1:4d}       {atom1:4d}     {atom2:4d}")
        
        print(f"LAMMPS data文件已保存至: {filename}")
        print(f"  - 原子总数: {self.total_atoms}")
        print(f"  - 键总数: {len(self.bonds)}")
    
    def write_xyz(self, filename: str = "oil_bubble_2.xyz"):
        """
        写入XYZ格式文件（用于可视化）
        
        参数说明：
        ----------
        filename : str, default="oil_bubble_2.xyz"
            输出文件名
        """
        with open(filename, 'w') as f:
            # XYZ文件头
            f.write(f"{self.total_atoms}           \n")
            f.write("Atoms.   Timestep:0")
            
            # 原子坐标
            for z in range(1, self.Z):
                for y in range(1, self.Y):
                    for x in range(1, self.X):
                        if self.ntype[z][y][x] > 0 and self.id[z][y][x] >= 1:
                            atom_type = self.ntype[z][y][x]
                            coord_x = float(x)
                            coord_y = float(y)
                            coord_z = float(z)
                            # 格式：原子类型  x  y  z
                            f.write(f"\n{atom_type:4d}     {coord_x:8.5f}     "
                                   f"{coord_y:8.5f}   {coord_z:8.5f}")
        
        print(f"XYZ文件已保存至: {filename}")
    
    def print_statistics(self):
        """打印系统统计信息"""
        # 原子类型名称映射
        atom_type_names = {
            1: 'sub (底层固定层)',
            2: 'water (水相)',
            3: 'gas (气相)',
            4: 'oil-head (油相头部)',
            5: 'oil-tail (油相尾部)',
            6: 'top (顶层固定层)'
        }
        
        print("\n" + "="*70)
        print("系统配置统计")
        print("="*70)
        print(f"网格尺寸: X={self.X}, Y={self.Y}, Z={self.Z}")
        print(f"总网格数: {self.X * self.Y * self.Z}")
        print(f"原子总数: {self.total_atoms}")
        print(f"键总数: {len(self.bonds)}")
        
        # 统计各类型原子数量
        atom_type_count = {}
        for z in range(self.Z):
            for y in range(self.Y):
                for x in range(self.X):
                    if self.ntype[z][y][x] > 0:
                        atom_type = self.ntype[z][y][x]
                        atom_type_count[atom_type] = atom_type_count.get(atom_type, 0) + 1
        
        print("\n各类型原子数量:")
        for atom_type in sorted(atom_type_count.keys()):
            count = atom_type_count[atom_type]
            percentage = (count / self.total_atoms) * 100
            type_name = atom_type_names.get(atom_type, '未知类型')
            print(f"  类型 {atom_type} - {type_name:25s}: {count:6d} 个 ({percentage:5.2f}%)")
        print("="*70 + "\n")


def main():
    """
    主函数：演示如何使用 GridSystemGenerator
    
    用户可修改参数的位置：
    ----------------------
    1. 网格尺寸: X_SIZE, Y_SIZE, Z_SIZE
    2. 顶层位置: TOP_LAYER_OFFSET (控制顶层固定层的Z坐标位置)
    3. Gas位置和尺寸: GAS_X_START, GAS_X_WIDTH
    4. Oil位置和尺寸: OIL_X_START, OIL_X_WIDTH
    5. 混合相区域Z方向: MIXED_PHASE_Z_START, MIXED_PHASE_Z_THICKNESS
    6. 输出文件名: LAMMPS_FILENAME, XYZ_FILENAME
    7. 原子类型数: NUM_ATOM_TYPES
    8. 区域定义: 可在 define_regions() 中传入自定义regions
    9. 键连接: 可在 create_bonds() 中传入自定义bond_regions
    """
    
    # ========== 用户输入参数区域 ==========
    
    # 参数1: 网格尺寸
    X_SIZE = 101  # X方向网格数
    Y_SIZE = 5    # Y方向网格数
    Z_SIZE = 81   # Z方向网格数
    
    # 参数2: 顶层位置控制
    TOP_LAYER_OFFSET = 7  # 顶层相对于Z_SIZE的向下偏移量
                           # offset=10 时，顶层在 z=69-70 (Z=81时)
                           # offset=5 时，顶层在 z=74-75 (Z=81时)
                           # 建议范围：5-20
    
    # 参数3: Gas（气相）位置和尺寸控制
    GAS_X_START = 21      # Gas在X方向的起始位置
    GAS_X_WIDTH = 20      # Gas在X方向的宽度 (默认10: x=21-30)
    
    # 参数4: Oil（油相）位置和尺寸控制
    OIL_X_START = 56      # Oil在X方向的起始位置
    OIL_X_WIDTH = 25      # Oil在X方向的宽度 (默认20: x=61-80)
    
    # 说明：Gas和Oil之间的间距 = OIL_X_START - (GAS_X_START + GAS_X_WIDTH)
    #      例如：61 - (21 + 10) = 30（即x=31-60为水相）
    
    # 参数5: 混合相区域的Z方向控制
    MIXED_PHASE_Z_START = 3       # 混合相区域在Z方向的起始位置
    MIXED_PHASE_Z_THICKNESS = 9   # 混合相区域在Z方向的厚度 (默认6: z=3-8)
    
    # 参数6: 输出文件名
    LAMMPS_FILENAME = "geo.dat"  # LAMMPS数据文件名
    XYZ_FILENAME = "geo.xyz"     # XYZ文件名
    
    # 参数7: 原子类型数
    NUM_ATOM_TYPES = 6
    
    # ====================================
    
    # 计算并显示系统配置信息
    gas_x_end = GAS_X_START + GAS_X_WIDTH - 1
    oil_x_end = OIL_X_START + OIL_X_WIDTH - 1
    gas_oil_gap = OIL_X_START - gas_x_end - 1
    mixed_z_end = MIXED_PHASE_Z_START + MIXED_PHASE_Z_THICKNESS - 1
    
    print("开始生成3D网格系统...")
    print(f"网格尺寸: X={X_SIZE}, Y={Y_SIZE}, Z={Z_SIZE}")
    print(f"\n位置配置:")
    print(f"  - 顶层固定层位置: z={Z_SIZE - TOP_LAYER_OFFSET - 2}-{Z_SIZE - TOP_LAYER_OFFSET - 1}")
    print(f"  - 混合相区域厚度: z={MIXED_PHASE_Z_START}-{mixed_z_end} (厚度={MIXED_PHASE_Z_THICKNESS})")
    print(f"  - Gas区域位置: x={GAS_X_START}-{gas_x_end} (宽度={GAS_X_WIDTH})")
    print(f"  - Oil区域位置: x={OIL_X_START}-{oil_x_end} (宽度={OIL_X_WIDTH})")
    print(f"  - Gas与Oil间距: {gas_oil_gap} (水相区域: x={gas_x_end+1}-{OIL_X_START-1})")
    
    # 创建生成器对象
    generator = GridSystemGenerator(
        x_size=X_SIZE, y_size=Y_SIZE, z_size=Z_SIZE,
        top_layer_offset=TOP_LAYER_OFFSET,
        gas_x_start=GAS_X_START, gas_x_width=GAS_X_WIDTH,
        oil_x_start=OIL_X_START, oil_x_width=OIL_X_WIDTH,
        mixed_phase_z_start=MIXED_PHASE_Z_START, 
        mixed_phase_z_thickness=MIXED_PHASE_Z_THICKNESS
    )
    
    # 定义区域（使用默认配置，也可以传入自定义regions）
    print("\n正在定义原子区域...")
    generator.define_regions()
    
    # 创建键连接（使用默认配置）
    print("正在创建原子键连接...")
    generator.create_bonds()
    
    # 打印统计信息
    generator.print_statistics()
    
    # 写入文件
    print("正在写入输出文件...")
    generator.write_lammps_data(filename=LAMMPS_FILENAME, num_atom_types=NUM_ATOM_TYPES)
    generator.write_xyz(filename=XYZ_FILENAME)
    
    print("\n✓ 所有文件生成完成！")


if __name__ == "__main__":
    main()

