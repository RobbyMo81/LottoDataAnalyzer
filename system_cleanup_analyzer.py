#!/usr/bin/env python3
"""
System Cleanup Analyzer
-----------------------
Comprehensive static analysis tool for identifying obsolete code, unused imports,
and legacy system dependencies throughout the Powerball Insights backend.
"""

import os
import ast
import re
import sys
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackendCleanupAnalyzer:
    """Analyzes backend modules for cleanup opportunities."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.core_dir = self.project_root / "core"
        
        # Legacy patterns to identify
        self.legacy_patterns = {
            'joblib_imports': [
                r'import joblib',
                r'from joblib import',
                r'joblib\.',
            ],
            'pickle_imports': [
                r'import pickle',
                r'from pickle import',
                r'pickle\.',
            ],
            'legacy_prediction_system': [
                r'from \.prediction_system import',
                r'PredictionSystem\(',
                r'prediction_system\.',
                r'\.prediction_history',
            ],
            'joblib_files': [
                r'\.joblib',
                r'MODEL_FILE_PATH',
                r'HISTORY_FILE_PATH',
            ],
            'unused_variables': [
                r'prediction_system\s*=',
                r'temp_system\s*=',
            ]
        }
        
        # Obsolete modules and their reasons
        self.obsolete_modules = {
            'prediction_data_cleaner.py': 'Legacy joblib data cleaning - replaced by SQLite',
            'prediction_storage_refactor.py': 'Migration utility - no longer needed',
            'prediction_storage_migration_ui.py': 'Migration UI - temporary tool',
        }
        
        # Analysis results
        self.findings = {
            'legacy_code': {},
            'unused_imports': {},
            'obsolete_files': {},
            'dependency_issues': {},
            'cleanup_opportunities': []
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for cleanup opportunities."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                ast_analysis = self._analyze_ast(tree, content)
            except SyntaxError as e:
                ast_analysis = {'error': f'Syntax error: {e}'}
            
            # Pattern-based analysis
            pattern_analysis = self._analyze_patterns(content)
            
            return {
                'file_path': str(file_path),
                'size': len(content),
                'lines': len(content.splitlines()),
                'ast_analysis': ast_analysis,
                'pattern_analysis': pattern_analysis,
                'legacy_score': self._calculate_legacy_score(pattern_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_ast(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Perform AST-based analysis."""
        analysis = {
            'imports': [],
            'functions': [],
            'classes': [],
            'unused_variables': [],
            'unreachable_code': []
        }
        
        # Collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append({
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    analysis['imports'].append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': len(node.args.args)
                })
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append({
                    'name': node.name,
                    'line': node.lineno
                })
        
        # Check for unused imports
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Find potentially unused imports
        for imp in analysis['imports']:
            import_name = imp.get('alias') or imp.get('name', '').split('.')[0]
            if import_name and import_name not in used_names:
                analysis['unused_variables'].append({
                    'type': 'import',
                    'name': import_name,
                    'line': imp['line']
                })
        
        return analysis
    
    def _analyze_patterns(self, content: str) -> Dict[str, List[Dict]]:
        """Analyze content for legacy patterns."""
        results = {}
        
        for pattern_type, patterns in self.legacy_patterns.items():
            matches = []
            for pattern in patterns:
                for line_num, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line):
                        matches.append({
                            'line': line_num,
                            'content': line.strip(),
                            'pattern': pattern
                        })
            results[pattern_type] = matches
        
        return results
    
    def _calculate_legacy_score(self, pattern_analysis: Dict) -> int:
        """Calculate a legacy score based on pattern matches."""
        score = 0
        weights = {
            'joblib_imports': 10,
            'pickle_imports': 8,
            'legacy_prediction_system': 15,
            'joblib_files': 5,
            'unused_variables': 3
        }
        
        for pattern_type, matches in pattern_analysis.items():
            if pattern_type in weights:
                score += len(matches) * weights[pattern_type]
        
        return score
    
    def analyze_module_dependencies(self) -> Dict[str, List[str]]:
        """Analyze inter-module dependencies."""
        dependencies = {}
        
        for py_file in self.core_dir.glob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports from core modules
                core_imports = []
                for line in content.splitlines():
                    # Match patterns like "from .module import" or "from core.module import"
                    match = re.search(r'from \.(\w+) import|from core\.(\w+) import', line)
                    if match:
                        module = match.group(1) or match.group(2)
                        core_imports.append(module)
                
                dependencies[py_file.stem] = core_imports
                
            except Exception as e:
                logger.warning(f"Could not analyze dependencies for {py_file}: {e}")
        
        return dependencies
    
    def find_orphaned_modules(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Find modules that are not imported by any other module."""
        imported_modules = set()
        all_modules = set(dependencies.keys())
        
        for module, imports in dependencies.items():
            imported_modules.update(imports)
        
        # Modules that exist but are never imported
        orphaned = all_modules - imported_modules
        
        # Filter out known entry points (like app.py equivalents)
        entry_points = {'__init__', 'storage', 'ingest'}  # Known entry points
        orphaned = orphaned - entry_points
        
        return list(orphaned)
    
    def analyze_file_usage(self) -> Dict[str, Dict]:
        """Analyze usage patterns of files."""
        usage_analysis = {}
        
        # Check data directory for obsolete files
        data_dir = self.project_root / "data"
        if data_dir.exists():
            for item in data_dir.rglob("*"):
                if item.is_file():
                    file_info = {
                        'path': str(item),
                        'size': item.stat().st_size,
                        'modified': item.stat().st_mtime,
                        'type': item.suffix
                    }
                    
                    # Check if it's a legacy joblib file
                    if item.suffix == '.joblib':
                        file_info['legacy'] = True
                        file_info['reason'] = 'Legacy joblib storage'
                    
                    usage_analysis[str(item)] = file_info
        
        return usage_analysis
    
    def generate_cleanup_plan(self) -> Dict[str, Any]:
        """Generate a comprehensive cleanup plan."""
        cleanup_plan = {
            'immediate_actions': [],
            'safe_removals': [],
            'requires_review': [],
            'archive_candidates': [],
            'code_refactoring': []
        }
        
        # Analyze all Python files in core
        for py_file in self.core_dir.glob("*.py"):
            if py_file.name.startswith('__'):
                continue
            
            analysis = self.analyze_file(py_file)
            
            # Check if file is obsolete
            if py_file.name in self.obsolete_modules:
                cleanup_plan['safe_removals'].append({
                    'file': str(py_file),
                    'reason': self.obsolete_modules[py_file.name],
                    'action': 'move_to_archive'
                })
            
            # Check legacy score
            legacy_score = analysis.get('legacy_score', 0)
            if legacy_score > 20:
                cleanup_plan['code_refactoring'].append({
                    'file': str(py_file),
                    'legacy_score': legacy_score,
                    'issues': analysis.get('pattern_analysis', {}),
                    'action': 'refactor_legacy_code'
                })
        
        # Analyze dependencies
        dependencies = self.analyze_module_dependencies()
        orphaned = self.find_orphaned_modules(dependencies)
        
        for module in orphaned:
            cleanup_plan['requires_review'].append({
                'module': module,
                'reason': 'No imports found - potential orphan',
                'action': 'review_necessity'
            })
        
        # File usage analysis
        file_usage = self.analyze_file_usage()
        for file_path, info in file_usage.items():
            if info.get('legacy'):
                cleanup_plan['archive_candidates'].append({
                    'file': file_path,
                    'reason': info.get('reason'),
                    'size': info.get('size'),
                    'action': 'archive_legacy_data'
                })
        
        return cleanup_plan
    
    def execute_safe_cleanup(self, cleanup_plan: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Execute safe cleanup operations."""
        results = {
            'actions_taken': [],
            'errors': [],
            'archived_files': [],
            'dry_run': dry_run
        }
        
        # Create archive directory
        archive_dir = self.project_root / "archive" / "legacy_cleanup"
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Process safe removals
        for item in cleanup_plan.get('safe_removals', []):
            file_path = Path(item['file'])
            if file_path.exists():
                if not dry_run:
                    # Move to archive
                    archive_path = archive_dir / file_path.name
                    file_path.rename(archive_path)
                    results['archived_files'].append(str(archive_path))
                
                results['actions_taken'].append({
                    'action': 'archived_file',
                    'file': str(file_path),
                    'reason': item['reason']
                })
        
        # Process legacy data files
        for item in cleanup_plan.get('archive_candidates', []):
            file_path = Path(item['file'])
            if file_path.exists() and '.joblib' in str(file_path):
                if not dry_run:
                    archive_path = archive_dir / "data" / file_path.name
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.rename(archive_path)
                    results['archived_files'].append(str(archive_path))
                
                results['actions_taken'].append({
                    'action': 'archived_legacy_data',
                    'file': str(file_path),
                    'size': item.get('size', 0)
                })
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive cleanup report."""
        cleanup_plan = self.generate_cleanup_plan()
        dependencies = self.analyze_module_dependencies()
        
        report = []
        report.append("=" * 80)
        report.append("BACKEND SYSTEM CLEANUP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        total_files = len(list(self.core_dir.glob("*.py")))
        total_cleanup_items = (
            len(cleanup_plan.get('safe_removals', [])) +
            len(cleanup_plan.get('code_refactoring', [])) +
            len(cleanup_plan.get('archive_candidates', []))
        )
        
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Python files analyzed: {total_files}")
        report.append(f"Total cleanup opportunities: {total_cleanup_items}")
        report.append(f"Obsolete modules identified: {len(cleanup_plan.get('safe_removals', []))}")
        report.append(f"Legacy code patterns: {len(cleanup_plan.get('code_refactoring', []))}")
        report.append(f"Legacy data files: {len(cleanup_plan.get('archive_candidates', []))}")
        report.append("")
        
        # Safe removals
        if cleanup_plan.get('safe_removals'):
            report.append("SAFE REMOVALS (Obsolete Modules)")
            report.append("-" * 40)
            for item in cleanup_plan['safe_removals']:
                report.append(f"• {item['file']}")
                report.append(f"  Reason: {item['reason']}")
                report.append()
        
        # Code refactoring needed
        if cleanup_plan.get('code_refactoring'):
            report.append("CODE REFACTORING REQUIRED")
            report.append("-" * 40)
            for item in cleanup_plan['code_refactoring']:
                report.append(f"• {item['file']} (Legacy Score: {item['legacy_score']})")
                for pattern_type, matches in item['issues'].items():
                    if matches:
                        report.append(f"  - {pattern_type}: {len(matches)} occurrences")
                report.append()
        
        # Archive candidates
        if cleanup_plan.get('archive_candidates'):
            report.append("LEGACY DATA FILES FOR ARCHIVAL")
            report.append("-" * 40)
            for item in cleanup_plan['archive_candidates']:
                size_mb = item.get('size', 0) / (1024 * 1024)
                report.append(f"• {item['file']} ({size_mb:.2f} MB)")
                report.append(f"  Reason: {item['reason']}")
                report.append()
        
        # Dependency analysis
        report.append("MODULE DEPENDENCY ANALYSIS")
        report.append("-" * 40)
        for module, deps in dependencies.items():
            if deps:
                report.append(f"• {module} depends on: {', '.join(deps)}")
        report.append()
        
        # Recommendations
        report.append("CLEANUP RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Archive obsolete modules to preserve version history")
        report.append("2. Refactor modules with high legacy scores")
        report.append("3. Remove or archive legacy joblib data files")
        report.append("4. Update imports to use modernized systems")
        report.append("5. Run comprehensive tests after cleanup")
        report.append()
        
        return "\n".join(report)

def main():
    """Main execution function."""
    print("Starting Backend System Cleanup Analysis...")
    
    analyzer = BackendCleanupAnalyzer()
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save report to file
    report_path = "BACKEND_CLEANUP_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Generate cleanup plan
    cleanup_plan = analyzer.generate_cleanup_plan()
    
    # Ask user for cleanup execution
    print("\nCleanup Plan Generated. Options:")
    print("1. Dry run (show what would be done)")
    print("2. Execute safe cleanup operations")
    print("3. Exit without changes")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        results = analyzer.execute_safe_cleanup(cleanup_plan, dry_run=True)
        print("\nDry Run Results:")
        for action in results['actions_taken']:
            print(f"Would {action['action']}: {action['file']}")
    
    elif choice == "2":
        print("\nExecuting safe cleanup operations...")
        results = analyzer.execute_safe_cleanup(cleanup_plan, dry_run=False)
        print("\nCleanup completed:")
        for action in results['actions_taken']:
            print(f"✓ {action['action']}: {action['file']}")
        
        if results['archived_files']:
            print(f"\nArchived {len(results['archived_files'])} files")
    
    else:
        print("Exiting without making changes.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())