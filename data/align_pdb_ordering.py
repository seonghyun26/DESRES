#!/usr/bin/env python3
"""
Compare atom ordering between two PDB files and create a mapping.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict

molecule = "1FME"
# molecule = "2F4K"
# molecule = "GTT"
# molecule = "NTL9"


def parse_pdb_atoms(pdb_file: Path) -> List[Tuple[int, str, str, str, int, float, float, float]]:
    """
    Parse PDB file and extract atom information.
    Returns list of tuples: (atom_number, atom_name, residue_name, chain, residue_number, x, y, z)
    """
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB ATOM/HETATM record
                atom_number = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain = line[21:22].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                atoms.append((atom_number, atom_name, residue_name, chain, residue_number, x, y, z))
    
    return atoms


def parse_pdb_atoms_with_lines(pdb_file: Path) -> Tuple[List[Tuple[int, str, str, str, int, float, float, float]], List[str]]:
    """
    Parse PDB file and extract atom information along with original ATOM/HETATM lines.
    Returns tuple of (atoms, lines) where atoms is the same structure as parse_pdb_atoms
    and lines contains the original PDB lines for ATOM/HETATM records in the same order.
    """
    atoms: List[Tuple[int, str, str, str, int, float, float, float]] = []
    lines: List[str] = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_number = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain = line[21:22].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                atoms.append((atom_number, atom_name, residue_name, chain, residue_number, x, y, z))
                lines.append(line.rstrip('\n'))

    return atoms, lines


def _rewrite_serial_number(pdb_line: str, new_serial: int) -> str:
    """Rewrite the atom serial number (columns 7-11) while preserving the rest of the line."""
    # PDB is 1-based fixed width, columns 7-11 are indices 6:11 (0-based slicing)
    prefix = pdb_line[:6]
    suffix = pdb_line[11:]
    return f"{prefix}{new_serial:>5}{suffix}"


def write_reordered_pdb_from_mapping(
    maestro_lines: List[str],
    mapping: List[int],
    pymol_atoms: List[Tuple[int, str, str, str, int, float, float, float]],
    output_path: Path,
) -> None:
    """
    Create a new PDB by reordering Maestro ATOM/HETATM lines to match PyMOL atom ordering.
    - mapping[i] gives PyMOL index (1-based) for Maestro atom i (0-based index i).
    - For each PyMOL index from 1..len(pymol_atoms), write the corresponding Maestro line
      if present in mapping; skip indices without a match (with a warning).
    - Renumber atom serials sequentially starting from 1 in the output.
    """
    # Invert mapping: pymol_index (1-based) -> maestro_index (0-based)
    inverse: Dict[int, int] = {}
    for maestro_idx, pymol_idx in enumerate(mapping):
        if pymol_idx != -1:
            # Only keep first occurrence if duplicates occur
            if pymol_idx not in inverse:
                inverse[pymol_idx] = maestro_idx

    output_lines: List[str] = []
    new_serial = 1
    total = len(pymol_atoms)
    for pymol_index in range(1, total + 1):
        maestro_idx = inverse.get(pymol_index, None)
        if maestro_idx is None:
            print(f"Warning: No Maestro atom corresponds to PyMOL index {pymol_index}; skipping.")
            continue
        original_line = maestro_lines[maestro_idx]
        output_lines.append(_rewrite_serial_number(original_line, new_serial))
        new_serial += 1

    # Write to file, append END record
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")
        f.write("END\n")


def create_atom_key(atom_info: Tuple) -> str:
    """Create a unique key for each atom based on residue and atom type."""
    atom_number, atom_name, residue_name, chain, residue_number, x, y, z = atom_info
    return f"{chain}_{residue_number}_{residue_name}_{atom_name}"


def find_mapping(maestro_atoms: List, pymol_atoms: List) -> List[int]:
    """
    Find mapping from maestro file to pymol file.
    Returns list where mapping[i] gives the pymol index for maestro atom i.
    """
    # Create dictionaries for fast lookup
    pymol_dict = {}
    for i, atom in enumerate(pymol_atoms):
        key = create_atom_key(atom)
        pymol_dict[key] = i
    
    mapping = []
    for i, maestro_atom in enumerate(maestro_atoms):
        maestro_key = create_atom_key(maestro_atom)
        
        if maestro_key in pymol_dict:
            # Found exact match
            pymol_index = pymol_dict[maestro_key]
            mapping.append(pymol_index + 1)  # +1 for 1-based indexing
        else:
            # Try to find by coordinates (in case of slight naming differences)
            maestro_coords = maestro_atom[5:8]  # x, y, z
            best_match = None
            min_distance = float('inf')
            
            for j, pymol_atom in enumerate(pymol_atoms):
                pymol_coords = pymol_atom[5:8]  # x, y, z
                distance = sum((a - b)**2 for a, b in zip(maestro_coords, pymol_coords))**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = j + 1  # +1 for 1-based indexing
            
            if min_distance < 0.01:  # Very small tolerance for coordinate matching
                mapping.append(best_match)
            else:
                print(f"Warning: No match found for maestro atom {i+1}: {maestro_key}")
                mapping.append(-1)  # Mark as not found
    
    return mapping


def analyze_differences(maestro_atoms: List, pymol_atoms: List, mapping: List[int]) -> None:
    """Analyze and report differences in atom ordering."""
    
    print(f"Total atoms in Maestro file: {len(maestro_atoms)}")
    print(f"Total atoms in PyMOL file: {len(pymol_atoms)}")
    print()
    
    # Count differences by residue type
    residue_diffs = {}
    atom_type_diffs = {}
    
    for i, (maestro_atom, pymol_idx) in enumerate(zip(maestro_atoms, mapping)):
        if pymol_idx == -1:
            continue
            
        maestro_key = create_atom_key(maestro_atom)
        pymol_atom = pymol_atoms[pymol_idx - 1]  # Convert back to 0-based
        pymol_key = create_atom_key(pymol_atom)
        
        # Check if ordering is different
        if i + 1 != pymol_idx:  # +1 for 1-based comparison
            residue_name = maestro_atom[2]
            atom_name = maestro_atom[1]
            
            if residue_name not in residue_diffs:
                residue_diffs[residue_name] = 0
            residue_diffs[residue_name] += 1
            
            if atom_name not in atom_type_diffs:
                atom_type_diffs[atom_name] = 0
            atom_type_diffs[atom_name] += 1
    
    print("Residue types with ordering differences:")
    for residue, count in sorted(residue_diffs.items()):
        print(f"  {residue}: {count} atoms")
    print()
    
    print("Atom types with ordering differences:")
    for atom_type, count in sorted(atom_type_diffs.items()):
        print(f"  {atom_type}: {count} atoms")
    print()


def main():
    # File paths
    maestro_file = Path(f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_from_mae.pdb")
    pymol_file = Path(f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_from_pymol.pdb")
    
    print("Parsing PDB files...")
    maestro_atoms, maestro_lines = parse_pdb_atoms_with_lines(maestro_file)
    pymol_atoms = parse_pdb_atoms(pymol_file)
    
    print("Creating mapping...")
    mapping = find_mapping(maestro_atoms, pymol_atoms)
    
    print("Analyzing differences...")
    analyze_differences(maestro_atoms, pymol_atoms, mapping)
    
    print("Mapping (Maestro -> PyMOL atom indices):")
    print("mapping =", mapping)
    print()

    # Write reordered PDB following PyMOL atom ordering
    # out_file = Path(f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_from_mae_reordered.pdb")
    # print(f"Writing reordered PDB to: {out_file}")
    # write_reordered_pdb_from_mapping(maestro_lines, mapping, pymol_atoms, out_file)
    
    # Show first 20 mappings as example
    # print("First 20 atom mappings:")
    # print("Maestro Index -> PyMOL Index (Residue, Atom)")
    # for i in range(min(20, len(mapping))):
    #     maestro_atom = maestro_atoms[i]
    #     pymol_idx = mapping[i]
    #     if pymol_idx != -1:
    #         pymol_atom = pymol_atoms[pymol_idx - 1]
    #         print(f"{i+1:3d} -> {pymol_idx:3d}  ({maestro_atom[2]} {maestro_atom[1]} -> {pymol_atom[2]} {pymol_atom[1]})")
    #     else:
    #         print(f"{i+1:3d} -> ???  ({maestro_atom[2]} {maestro_atom[1]} -> NOT FOUND)")
    
    # Verify mapping correctness
    print("\nVerification - checking if coordinates match:")
    mismatches = 0
    for i, pymol_idx in enumerate(mapping[:10]):  # Check first 10
        if pymol_idx == -1:
            continue
        maestro_coords = maestro_atoms[i][5:8]
        pymol_coords = pymol_atoms[pymol_idx - 1][5:8]
        distance = sum((a - b)**2 for a, b in zip(maestro_coords, pymol_coords))**0.5
        if distance > 0.01:
            mismatches += 1
            print(f"  Mismatch at index {i+1}: distance = {distance:.6f}")
    
    if mismatches == 0:
        print("  All checked coordinates match perfectly!")
    else:
        print(f"  Found {mismatches} coordinate mismatches")


if __name__ == "__main__":
    main()
