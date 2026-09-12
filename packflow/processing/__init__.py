"""
Crystal Structure Processing Module

This module contains utilities for processing crystallographic structures
from mmCIF files, including parsing, conformer generation, and transformation.
"""

from .final_crystal_processor import (
    parse_mmcif_to_crystal_dict,
    process_single_crystal,
    process_single_mmcif_file,
    process_directory,
    # apply_transformations_and_visualize
)

__all__ = [
    'parse_mmcif_to_crystal_dict',
    'process_single_crystal',
    'process_single_mmcif_file', 
    'process_directory',
    'apply_transformations_and_visualize'
] 