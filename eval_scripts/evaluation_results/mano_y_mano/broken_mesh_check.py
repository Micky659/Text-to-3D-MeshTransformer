"""
Mesh Validation Tool

Analyzes a folder of mesh files, identifies valid vs broken meshes,
and generates detailed reports.

Note: This tool uses basic geometric validation and doesn't require
advanced trimesh features. Works with trimesh 3.x and 4.x.
"""

import numpy as np
from pathlib import Path
import trimesh
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_mesh_safely(mesh_path: Path):
    """
    Safely load mesh file with error handling.

    Returns:
        (vertices, faces) or (None, None) if failed
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if mesh is None:
            return None, None

        # Handle different mesh types
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            return vertices, faces
        else:
            return None, None
    except Exception:
        return None, None


class MeshValidator:
    """
    Validates mesh files and provides detailed diagnostics.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, applies stricter validation criteria
        """
        self.strict_mode = strict_mode
        self.validation_results = {
            'valid': [],
            'broken': [],
            'broken_details': {}
        }

    def validate_mesh_file(self, mesh_path: Path) -> Tuple[bool, str]:
        """
        Validate a single mesh file using robust geometric checks.

        Args:
            mesh_path: Path to mesh file (.obj, .ply, .stl, etc.)

        Returns:
            (is_valid, failure_reason)
        """
        try:
            # Try to load mesh safely
            vertices, faces = load_mesh_safely(mesh_path)

            # Check if loading failed
            if vertices is None or faces is None:
                return False, "failed_to_load"

            # Basic checks
            if len(vertices) == 0:
                return False, "no_vertices"

            if len(faces) == 0:
                return False, "no_faces"

            # Check for NaN or Inf
            if np.any(np.isnan(vertices)):
                return False, "nan_vertices"

            if np.any(np.isinf(vertices)):
                return False, "inf_vertices"

            # Check if all vertices are at the same point (degenerate)
            vertex_range = np.ptp(vertices, axis=0)  # peak-to-peak (max - min)
            if np.max(vertex_range) < 1e-8:
                return False, "degenerate_all_same_point"

            # Check for unreasonably large coordinates
            max_coord = np.max(np.abs(vertices))
            if max_coord > 1e6:
                return False, f"unreasonably_large_coords_{max_coord:.2e}"

            # Check for unreasonably small mesh (might be numerical noise)
            if np.max(vertex_range) < 1e-6:
                return False, "unreasonably_small_mesh"

            # Check face indices are valid
            max_index = np.max(faces)
            min_index = np.min(faces)

            if min_index < 0:
                return False, "negative_face_indices"

            if max_index >= len(vertices):
                return False, f"invalid_face_indices_max_{max_index}_vs_{len(vertices)}"

            # Check for degenerate faces (vertices of a face are not all unique)
            for i, face in enumerate(faces):
                unique_verts = len(np.unique(face))
                expected_verts = len(face)  # Should be 3 for triangles, 4 for quads

                if unique_verts < expected_verts:
                    return False, f"degenerate_face_{i}_only_{unique_verts}_unique"

                # Early exit after checking first 100 faces in non-strict mode
                if not self.strict_mode and i >= 100:
                    break

            # Additional geometric validation
            try:
                # Check if we can compute face areas (basic mesh validity)
                face_vertices = vertices[faces]  # Shape: (num_faces, 3, 3)

                # Compute edge vectors for each triangle
                v0 = face_vertices[:, 0, :]
                v1 = face_vertices[:, 1, :]
                v2 = face_vertices[:, 2, :]

                edge1 = v1 - v0
                edge2 = v2 - v0

                # Cross product to get face normals
                normals = np.cross(edge1, edge2)
                areas = 0.5 * np.linalg.norm(normals, axis=1)

                # Check for zero-area faces (degenerate)
                zero_area_faces = np.sum(areas < 1e-10)
                if zero_area_faces > 0:
                    return False, f"zero_area_faces_{zero_area_faces}"

            except Exception as e:
                return False, f"geometric_validation_failed_{str(e)[:30].replace(' ', '_')}"

            # Strict mode: additional checks
            if self.strict_mode:
                # Check for duplicate vertices
                # unique_vertices, inverse = np.unique(
                #     vertices.round(decimals=6),  # Round to avoid floating point issues
                #     axis=0,
                #     return_inverse=True
                # )
                # if len(unique_vertices) < len(vertices):
                #     duplicate_count = len(vertices) - len(unique_vertices)
                #     return False, f"duplicate_vertices_{duplicate_count}"

                # Check for unreferenced vertices
                referenced_indices = np.unique(faces.flatten())
                unreferenced_count = len(vertices) - len(referenced_indices)
                if unreferenced_count > 0:
                    return False, f"unreferenced_vertices_{unreferenced_count}"

                # Check mesh is manifold (each edge is shared by at most 2 faces)
                try:
                    edges = []
                    for face in faces:
                        for i in range(len(face)):
                            edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                            edges.append(edge)

                    from collections import Counter
                    edge_counts = Counter(edges)
                    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)

                    if non_manifold_edges > 0:
                        return False, f"non_manifold_{non_manifold_edges}_edges"

                except Exception:
                    pass  # Skip manifold check if it fails

            # All checks passed
            return True, "valid"

        except Exception as e:
            error_msg = str(e)[:50].replace(' ', '_').replace('/', '_')
            return False, f"exception_{error_msg}"

    def validate_folder(
        self,
        folder_path: str,
        extensions: List[str] = ['.obj', '.ply', '.stl', '.off'],
        save_report: bool = True,
        report_path: str = None
    ) -> Dict:
        """
        Validate all mesh files in a folder.

        Args:
            folder_path: Path to folder containing mesh files
            extensions: List of file extensions to check
            save_report: Whether to save detailed report
            report_path: Where to save report (auto-generated if None)

        Returns:
            Dictionary with validation results
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        print(f"\n{'='*80}")
        print(f"MESH VALIDATION: {folder_path}")
        print(f"{'='*80}\n")

        # Find all mesh files
        mesh_files = []
        for ext in extensions:
            mesh_files.extend(folder_path.glob(f"*{ext}"))
            mesh_files.extend(folder_path.glob(f"**/*{ext}"))  # Recursive

        mesh_files = sorted(set(mesh_files))  # Remove duplicates

        print(f"Found {len(mesh_files)} mesh files")
        print(f"Extensions: {extensions}")
        print(f"Strict mode: {self.strict_mode}\n")

        if len(mesh_files) == 0:
            print("No mesh files found!")
            return None

        # Validate each mesh
        valid_meshes = []
        broken_meshes = []
        failure_reasons = {}

        for mesh_path in tqdm(mesh_files, desc="Validating meshes"):
            is_valid, reason = self.validate_mesh_file(mesh_path)

            if is_valid:
                valid_meshes.append(mesh_path.name)
            else:
                broken_meshes.append(mesh_path.name)
                failure_reasons[mesh_path.name] = reason

        # Compile results
        total = len(mesh_files)
        num_valid = len(valid_meshes)
        num_broken = len(broken_meshes)
        success_rate = (num_valid / total * 100) if total > 0 else 0

        results = {
            'folder': str(folder_path),
            'total_meshes': total,
            'valid_meshes': num_valid,
            'broken_meshes': num_broken,
            'success_rate': success_rate,
            'valid_mesh_list': valid_meshes,
            'broken_mesh_list': broken_meshes,
            'failure_reasons': failure_reasons,
            'strict_mode': self.strict_mode
        }

        # Store in instance
        self.validation_results = results

        # Print summary
        self._print_summary(results)

        # Analyze failure types
        self._analyze_failures(failure_reasons)

        # Save report
        if save_report:
            report_path = report_path or folder_path / "validation_report.json"
            self.save_report(report_path)

        return results

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total meshes:       {results['total_meshes']}")
        print(f"Valid meshes:       {results['valid_meshes']} ({results['success_rate']:.1f}%)")
        print(f"Broken meshes:      {results['broken_meshes']} ({100-results['success_rate']:.1f}%)")
        print(f"{'='*80}\n")

    def _analyze_failures(self, failure_reasons: Dict):
        """Analyze and categorize failure types."""
        if not failure_reasons:
            return

        print(f"{'='*80}")
        print(f"FAILURE ANALYSIS")
        print(f"{'='*80}")

        # Count failure types
        failure_counts = {}
        for reason in failure_reasons.values():
            # Extract base reason (remove specific details)
            base_reason = reason.split('_')[0] if '_' in reason else reason
            failure_counts[base_reason] = failure_counts.get(base_reason, 0) + 1

        # Sort by frequency
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Failure Type':<30} {'Count':<10} {'Percentage':<10}")
        print(f"{'-'*50}")

        for failure_type, count in sorted_failures:
            percentage = (count / len(failure_reasons) * 100)
            print(f"{failure_type:<30} {count:<10} {percentage:.1f}%")

        print(f"\n{'='*80}\n")

        # Show examples of each failure type
        print("Example broken meshes by failure type:")
        print(f"{'-'*80}")

        for failure_type, _ in sorted_failures[:5]:  # Top 5 failure types
            examples = [name for name, reason in failure_reasons.items()
                       if reason.startswith(failure_type)][:3]  # Show 3 examples
            print(f"\n{failure_type}:")
            for example in examples:
                print(f"  - {example}")

        print(f"\n{'='*80}\n")

    def save_report(self, output_path: str):
        """Save detailed validation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"Detailed report saved to: {output_path}")

    def save_broken_mesh_list(self, output_path: str):
        """Save list of broken mesh filenames to text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        broken_meshes = self.validation_results.get('broken_mesh_list', [])

        with open(output_path, 'w') as f:
            f.write(f"# Broken Mesh Files\n")
            f.write(f"# Total: {len(broken_meshes)}\n")
            f.write(f"# Generated: {Path(self.validation_results['folder']).name}\n\n")

            for mesh_name in sorted(broken_meshes):
                reason = self.validation_results['failure_reasons'].get(mesh_name, 'unknown')
                f.write(f"{mesh_name}\t# {reason}\n")

        print(f"Broken mesh list saved to: {output_path}")

    def delete_broken_meshes(self, confirm: bool = False):
        """
        Delete broken mesh files from folder.

        Args:
            confirm: If False, performs dry run only
        """
        if not self.validation_results:
            print("No validation results available. Run validate_folder() first.")
            return

        folder = Path(self.validation_results['folder'])
        broken_meshes = self.validation_results['broken_mesh_list']

        print(f"\n{'='*80}")
        print(f"DELETE BROKEN MESHES")
        print(f"{'='*80}")
        print(f"Folder: {folder}")
        print(f"Meshes to delete: {len(broken_meshes)}")

        if not confirm:
            print(f"\nDRY RUN - No files will be deleted")
            print(f"Set confirm=True to actually delete files")
        else:
            print(f"\nWARNING: This will permanently delete {len(broken_meshes)} files!")

        print(f"\nFiles to be deleted:")
        for i, mesh_name in enumerate(sorted(broken_meshes)[:10]):
            reason = self.validation_results['failure_reasons'].get(mesh_name, 'unknown')
            print(f"  {i+1}. {mesh_name} ({reason})")

        if len(broken_meshes) > 10:
            print(f"  ... and {len(broken_meshes) - 10} more")

        if confirm:
            deleted = 0
            for mesh_name in tqdm(broken_meshes, desc="Deleting"):
                mesh_path = folder / mesh_name
                if mesh_path.exists():
                    mesh_path.unlink()
                    deleted += 1
            print(f"\nDeleted {deleted} broken mesh files")

        print(f"{'='*80}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validate(folder_path: str, strict: bool = False) -> Dict:
    """
    Quick validation of mesh folder.

    Args:
        folder_path: Path to folder with meshes
        strict: Use strict validation mode

    Returns:
        Validation results dictionary
    """
    validator = MeshValidator(strict_mode=strict)
    results = validator.validate_folder(folder_path, save_report=True)

    # Save broken mesh list
    if results and results['broken_meshes'] > 0:
        validator.save_broken_mesh_list(
            Path(folder_path) / "broken_meshes.txt"
        )

    return results


def compare_folders(folder1: str, folder2: str):
    """
    Compare validation results between two folders.

    Useful for comparing your model vs MeshGPT outputs.
    """
    print("\n" + "="*80)
    print("COMPARING MESH QUALITY BETWEEN FOLDERS")
    print("="*80 + "\n")

    validator1 = MeshValidator(strict_mode=False)
    results1 = validator1.validate_folder(folder1, save_report=True)

    print("\n")

    validator2 = MeshValidator(strict_mode=False)
    results2 = validator2.validate_folder(folder2, save_report=True)

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {Path(folder1).name:<30} {Path(folder2).name:<30}")
    print(f"{'-'*90}")
    print(f"{'Total meshes':<30} {results1['total_meshes']:<30} {results2['total_meshes']:<30}")
    print(f"{'Valid meshes':<30} {results1['valid_meshes']:<30} {results2['valid_meshes']:<30}")
    print(f"{'Broken meshes':<30} {results1['broken_meshes']:<30} {results2['broken_meshes']:<30}")
    print(f"{'Success rate':<30} {results1['success_rate']:.1f}%{'':<26} {results2['success_rate']:.1f}%{'':<26}")
    print(f"="*80 + "\n")

    return results1, results2


def filter_valid_meshes(input_folder: str, output_folder: str):
    """
    Copy only valid meshes to a new folder.

    Useful for creating a clean dataset from mixed valid/broken meshes.
    """
    import shutil

    print(f"\nFiltering valid meshes from {input_folder} to {output_folder}...")

    validator = MeshValidator(strict_mode=False)
    results = validator.validate_folder(input_folder, save_report=False)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {results['valid_meshes']} valid meshes...")

    for mesh_name in tqdm(results['valid_mesh_list']):
        src = input_path / mesh_name
        dst = output_path / mesh_name
        if src.exists():
            shutil.copy(src, dst)

    print(f"\nDone! Valid meshes saved to: {output_folder}")
    print(f"Copied: {results['valid_meshes']} meshes")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage scenarios
    """

    # Example 1: Basic validation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Validation")
    print("=" * 80)

    # results = quick_validate("C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\eval_scripts\\evaluation_results\\mano_y_mano\\chair\\Ours")

    # Example 2: Strict validation
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strict Validation")
    print("=" * 80)

    # validator = MeshValidator(strict_mode=True)
    # results = validator.validate_folder(
    #     # "C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\eval_scripts\\evaluation_results\\mano_y_mano\\chair\\Ours",
    #   "C:\\Users\\koth_ak\\Akshat's Thesis\\MeshGPT\\broken_tries_topp_chairs\\chair\\MeshGPT_topp",
    #     save_report=True,
    #     report_path="validation_strict_meshgpt.json",
    #     # report_path="validation_strict_ours.json",
    # )

    # Save broken mesh list
    # if results['broken_meshes'] > 0:
    #     validator.save_broken_mesh_list("broken_meshes_list.txt")
    #
    # # Example 3: Compare two models
    # print("\n" + "=" * 80)
    # print("EXAMPLE 3: Compare Models")
    # print("=" * 80)
    #
    # results1, results2 = compare_folders(
    #     "C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\eval_scripts\\evaluation_results\\mano_y_mano\\chair\\Ours",
    #     "C:\\Users\\koth_ak\\Akshat's Thesis\\MeshGPT\\broken_tries_topp_chairs\\chair\\MeshGPT_topp"
    # )

    # # Example 4: Filter and copy only valid meshes
    # print("\n" + "=" * 80)
    # print("EXAMPLE 4: Filter Valid Meshes")
    # print("=" * 80)
    #
    # filter_valid_meshes(
    #     input_folder="generated_meshes/chair",
    #     output_folder="generated_meshes/chair_valid_only"
    # )
    #
    # # Example 5: Delete broken meshes (DRY RUN)
    # print("\n" + "=" * 80)
    # print("EXAMPLE 5: Delete Broken Meshes (Dry Run)")
    # print("=" * 80)
    #
    validator = MeshValidator()
    validator.validate_folder("C:\\Users\\koth_ak\\Akshat's Thesis\\Text-to-3D-MeshTransformer\\eval_scripts\\evaluation_results\\mano_y_mano\\chair\\Ours")
    validator.delete_broken_meshes(confirm=True)  # Dry run
    #
    # To actually delete: validator.delete_broken_meshes(confirm=True)

    print("\n" + "=" * 80)
    print("MESH VALIDATOR - USAGE GUIDE")
    print("=" * 80)
    print("""
    Quick Start:

    1. Basic validation:
       results = quick_validate("path/to/meshes")

    2. Get broken mesh list:
       validator = MeshValidator()
       results = validator.validate_folder("path/to/meshes")
       validator.save_broken_mesh_list("broken_list.txt")

    3. Compare two folders:
       compare_folders("model1/chair", "model2/chair")

    4. Filter valid meshes:
       filter_valid_meshes("input_folder", "output_folder")

    5. Delete broken meshes:
       validator.delete_broken_meshes(confirm=True)  # BE CAREFUL!
    """)
    print("=" * 80 + "\n")