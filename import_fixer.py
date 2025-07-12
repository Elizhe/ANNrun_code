#!/usr/bin/env python3
"""
Import Fixer for ANNrun_code
Fixes import statements in files to match new modular structure
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class ImportFixer:
    """Fix import statements across ANNrun_code project"""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        self.errors = []
        
        # Mapping of old imports to new imports
        self.import_mappings = {
            # Old data loader imports
            'from core.data.loaders.base_loader import DataLoader': 'from core.data.loaders.base_loader import DataLoader',
            'from core.data.loaders import base_loader': 'from core.data.loaders import base_loader',
            'from core.data.loaders.base_loader import': 'from core.data.loaders.base_loader import',
            
            # Old pangaea loader imports
            'from core.data.loaders.pangaea_loader import PangaeaDataLoader': 'from core.data.loaders.pangaea_loader import PangaeaDataLoader',
            'from core.data.loaders import pangaea_loader': 'from core.data.loaders import pangaea_loader',
            
            # Old model imports
            'from core.models.lumen.model import': 'from core.models.lumen.model import',
            'from core.models.lumen import model as LUMEN': 'from core.models.lumen import model as LUMEN',
            
            # Old bias correction imports
            'from core.preprocessing.bias_correction.corrector import': 'from core.preprocessing.bias_correction.corrector import',
            'from core.preprocessing.bias_correction import corrector as bias': 'from core.preprocessing.bias_correction import corrector as bias',
            
            # Common patterns
            'from ..core.data.loaders.base_loader import': 'from ..core.data.loaders.base_loader import',
            'from ..core.data.loaders.pangaea_loader import': 'from ..core.data.loaders.pangaea_loader import',
        }
    
    def analyze_file_imports(self, filepath: Path) -> Dict[str, List[str]]:
        """Analyze imports in a specific file"""
        if not filepath.exists():
            return {'error': [f"File not found: {filepath}"]}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all import lines
            import_lines = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    import_lines.append({
                        'line_num': i,
                        'content': line,
                        'stripped': stripped
                    })
            
            return {
                'imports': import_lines,
                'total_lines': len(lines),
                'file_size': len(content)
            }
            
        except Exception as e:
            return {'error': [f"Error reading {filepath}: {e}"]}
    
    def fix_imports_in_file(self, filepath: Path, dry_run: bool = False) -> Dict[str, any]:
        """Fix imports in a specific file"""
        print(f"\\nAnalyzing: {filepath.relative_to(self.project_root)}")
        
        analysis = self.analyze_file_imports(filepath)
        if 'error' in analysis:
            print(f"❌ {analysis['error'][0]}")
            return {'success': False, 'errors': analysis['error']}
        
        if not analysis['imports']:
            print("ℹ️  No import statements found")
            return {'success': True, 'changes': 0}
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        changes_log = []
        
        # Apply fixes
        for old_import, new_import in self.import_mappings.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes_made += 1
                changes_log.append(f"  🔧 {old_import} → {new_import}")
                print(f"  🔧 Fixed: {old_import}")
        
        # Check for other problematic patterns
        problematic_patterns = [
            r'from core.data.loaders.base_loader import \\w+',
            r'from core.data.loaders import base_loader',
            r'from pangaea_data_loader import \\w+',
            r'from core.data.loaders import pangaea_loader',
            r'from core.models.lumen.model import \\w+',
            r'from core.models.lumen import model as LUMEN',
            r'from core.preprocessing.bias_correction.corrector import \\w+',
            r'from core.preprocessing.bias_correction import corrector as bias'
        ]
        
        for pattern in problematic_patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"  ⚠️  Potential issue found: {matches}")
        
        # Write changes if not dry run
        if changes_made > 0 and not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Applied {changes_made} fixes")
                
                self.fixes_applied.append({
                    'file': str(filepath),
                    'changes': changes_made,
                    'log': changes_log
                })
                
            except Exception as e:
                error_msg = f"Error writing to {filepath}: {e}"
                print(f"  ❌ {error_msg}")
                self.errors.append(error_msg)
                return {'success': False, 'errors': [error_msg]}
        
        elif changes_made > 0 and dry_run:
            print(f"  📋 Would apply {changes_made} fixes (dry run)")
        
        else:
            print("  ✅ No fixes needed")
        
        return {
            'success': True,
            'changes': changes_made,
            'dry_run': dry_run,
            'changes_log': changes_log
        }
    
    def scan_project_files(self) -> List[Path]:
        """Scan project for Python files that might need import fixes"""
        python_files = []
        
        # Key directories to scan
        scan_dirs = [
            self.project_root / "experiments",
            self.project_root / "core",
            self.project_root / "configs",
            self.project_root / "scripts"
        ]
        
        # Add root level Python files
        for file in self.project_root.glob("*.py"):
            python_files.append(file)
        
        # Scan directories
        for scan_dir in scan_dirs:
            if scan_dir.exists():
                for file in scan_dir.rglob("*.py"):
                    python_files.append(file)
        
        return sorted(python_files)
    
    def fix_all_imports(self, dry_run: bool = False) -> Dict[str, any]:
        """Fix imports in all Python files"""
        print("=" * 60)
        print("IMPORT FIXER FOR ANNRUN_CODE")
        print("=" * 60)
        print(f"Mode: {'DRY RUN' if dry_run else 'APPLY FIXES'}")
        print(f"Project root: {self.project_root.absolute()}")
        
        # Find Python files
        python_files = self.scan_project_files()
        print(f"\\nFound {len(python_files)} Python files to check")
        
        if not python_files:
            print("❌ No Python files found")
            return {'success': False, 'errors': ['No Python files found']}
        
        # Process each file
        total_changes = 0
        successful_fixes = 0
        
        for filepath in python_files:
            result = self.fix_imports_in_file(filepath, dry_run)
            
            if result['success']:
                successful_fixes += 1
                total_changes += result.get('changes', 0)
            else:
                self.errors.extend(result.get('errors', []))
        
        # Summary
        print("\\n" + "=" * 60)
        print("IMPORT FIXING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {len(python_files)}")
        print(f"Files successfully processed: {successful_fixes}")
        print(f"Total changes made: {total_changes}")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\\nErrors:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        if self.fixes_applied:
            print("\\nFixes applied:")
            for fix in self.fixes_applied:
                print(f"  📁 {fix['file']}: {fix['changes']} changes")
        
        if dry_run and total_changes > 0:
            print(f"\\n📋 Run without --dry-run to apply {total_changes} fixes")
        elif not dry_run and total_changes > 0:
            print(f"\\n✅ Applied {total_changes} fixes successfully!")
            print("\\nTry running main.py again:")
            print("python main.py --dry-run")
        elif total_changes == 0:
            print("\\n✅ No import fixes needed")
        
        return {
            'success': len(self.errors) == 0,
            'files_processed': len(python_files),
            'total_changes': total_changes,
            'errors': self.errors,
            'fixes_applied': self.fixes_applied
        }
    
    def check_specific_file(self, filename: str):
        """Check a specific file for import issues"""
        filepath = self.project_root / filename
        
        print(f"Checking specific file: {filename}")
        print("=" * 60)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return
        
        analysis = self.analyze_file_imports(filepath)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error'][0]}")
            return
        
        print(f"File: {filepath}")
        print(f"Size: {analysis['file_size']:,} characters")
        print(f"Total lines: {analysis['total_lines']}")
        print(f"Import statements: {len(analysis['imports'])}")
        
        if analysis['imports']:
            print("\\nImport statements found:")
            for imp in analysis['imports']:
                line_num = imp['line_num']
                content = imp['stripped']
                
                # Check if this import needs fixing
                needs_fix = any(old in content for old in self.import_mappings.keys())
                status = "⚠️ " if needs_fix else "✅"
                
                print(f"  {status} Line {line_num:3d}: {content}")
        else:
            print("\\nNo import statements found")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix import statements in ANNrun_code")
    parser.add_argument('--project-root', default='.',
                       help='Project root directory (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be fixed without making changes')
    parser.add_argument('--check-file', 
                       help='Check specific file (e.g., experiments/architecture_experiment.py)')
    
    args = parser.parse_args()
    
    # Create fixer
    fixer = ImportFixer(args.project_root)
    
    if args.check_file:
        # Check specific file
        fixer.check_specific_file(args.check_file)
    else:
        # Fix all imports
        result = fixer.fix_all_imports(args.dry_run)
        
        # Exit with appropriate code
        if result['success'] and result['total_changes'] == 0:
            exit(0)  # No changes needed
        elif result['success'] and result['total_changes'] > 0:
            exit(0)  # Changes applied successfully
        else:
            exit(1)  # Errors occurred


if __name__ == "__main__":
    main()