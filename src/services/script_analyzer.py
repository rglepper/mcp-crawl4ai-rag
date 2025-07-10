"""
Script analyzer service for the Crawl4AI MCP server.

This service handles AST-based analysis of Python scripts to extract
imports, class instantiations, method calls, and other code patterns.
"""
import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    name: str
    alias: Optional[str] = None
    is_from_import: bool = False
    line_number: int = 0


@dataclass
class MethodCall:
    """Information about a method call"""
    object_name: str
    method_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    object_type: Optional[str] = None  # Inferred class type


@dataclass
class AttributeAccess:
    """Information about attribute access"""
    object_name: str
    attribute_name: str
    line_number: int
    object_type: Optional[str] = None  # Inferred class type


@dataclass
class FunctionCall:
    """Information about a function call"""
    function_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    full_name: Optional[str] = None  # Module.function_name


@dataclass
class ClassInstantiation:
    """Information about class instantiation"""
    variable_name: str
    class_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    full_class_name: Optional[str] = None  # Module.ClassName


@dataclass
class AnalysisResult:
    """Complete analysis results for a Python script"""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    class_instantiations: List[ClassInstantiation] = field(default_factory=list)
    method_calls: List[MethodCall] = field(default_factory=list)
    attribute_accesses: List[AttributeAccess] = field(default_factory=list)
    function_calls: List[FunctionCall] = field(default_factory=list)
    variable_types: Dict[str, str] = field(default_factory=dict)  # variable_name -> class_type
    errors: List[str] = field(default_factory=list)


class ScriptAnalyzerService:
    """Service for analyzing Python scripts using AST."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the script analyzer service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.import_map: Dict[str, str] = {}  # alias -> actual_module_name
        self.variable_types: Dict[str, str] = {}  # variable_name -> class_type
        self.context_manager_vars: Dict[str, Tuple[int, int, str]] = {}  # var_name -> (start_line, end_line, type)
    
    def analyze_script(self, script_path: str) -> AnalysisResult:
        """
        Analyze a Python script and extract all relevant information.
        
        Args:
            script_path: Path to the Python script to analyze
            
        Returns:
            AnalysisResult containing extracted information
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            result = AnalysisResult(file_path=script_path)
            
            # Reset state for new analysis
            self.import_map.clear()
            self.variable_types.clear()
            self.context_manager_vars.clear()
            
            # Track processed nodes to avoid duplicates
            self.processed_calls = set()
            self.method_call_attributes = set()
            
            # First pass: collect imports and build import map
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._extract_imports(node, result)
            
            # Second pass: analyze usage patterns
            for node in ast.walk(tree):
                self._analyze_node(node, result)
            
            # Set inferred types on method calls and attribute accesses
            self._infer_object_types(result)
            
            result.variable_types = self.variable_types.copy()
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to analyze script {script_path}: {str(e)}"
            logger.error(error_msg)
            result = AnalysisResult(file_path=script_path)
            result.errors.append(error_msg)
            return result
    
    def _extract_imports(self, node: ast.AST, result: AnalysisResult):
        """Extract import information from AST node."""
        line_num = getattr(node, 'lineno', 0)
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = ImportInfo(
                    module=alias.name,
                    name=alias.name,
                    alias=alias.asname,
                    is_from_import=False,
                    line_number=line_num
                )
                result.imports.append(import_info)
                
                # Update import map
                key = alias.asname if alias.asname else alias.name
                self.import_map[key] = alias.name
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                import_info = ImportInfo(
                    module=module,
                    name=alias.name,
                    alias=alias.asname,
                    is_from_import=True,
                    line_number=line_num
                )
                result.imports.append(import_info)
                
                # Update import map
                key = alias.asname if alias.asname else alias.name
                if module:
                    self.import_map[key] = f"{module}.{alias.name}"
                else:
                    self.import_map[key] = alias.name
    
    def _analyze_node(self, node: ast.AST, result: AnalysisResult):
        """Analyze a single AST node for patterns."""
        # Skip already processed calls
        if hasattr(node, '__class__') and id(node) in self.processed_calls:
            return
        
        # Assignments (class instantiations and method call results)
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Call):
                    # Check if it's a class instantiation or method call
                    if isinstance(node.value.func, ast.Name):
                        # Direct function/class call
                        self._extract_class_instantiation(node, result)
                        # Mark this call as processed to avoid duplicate processing
                        self.processed_calls.add(id(node.value))
                    elif isinstance(node.value.func, ast.Attribute):
                        # Method call - track the variable assignment for type inference
                        var_name = node.targets[0].id
                        self._track_method_result_assignment(node.value, var_name)
                        # Still process the method call
                        self._extract_method_call(node.value, result)
        
        # Method calls and function calls
        if isinstance(node, ast.Call) and id(node) not in self.processed_calls:
            if isinstance(node.func, ast.Attribute):
                self._extract_method_call(node, result)
                # Mark this attribute as used in method call to avoid duplicate processing
                self.method_call_attributes.add(id(node.func))
            elif isinstance(node.func, ast.Name):
                # Check if this is likely a class instantiation (based on imported classes)
                func_name = node.func.id
                full_name = self._resolve_full_name(func_name)
                
                # If this is a known imported class, treat as class instantiation
                if self._is_likely_class_instantiation(func_name, full_name):
                    self._extract_nested_class_instantiation(node, result)
                else:
                    self._extract_function_call(node, result)
        
        # Attribute access (not part of method calls)
        if (isinstance(node, ast.Attribute) and 
            id(node) not in self.method_call_attributes):
            self._extract_attribute_access(node, result)
    
    def _extract_class_instantiation(self, assign_node: ast.Assign, result: AnalysisResult):
        """Extract class instantiation from assignment node."""
        target = assign_node.targets[0]
        call = assign_node.value
        line_num = getattr(assign_node, 'lineno', 0)
        
        if isinstance(target, ast.Name) and isinstance(call, ast.Call):
            var_name = target.id
            class_name = self._get_name_from_call(call.func)
            
            if class_name:
                args = [self._get_arg_representation(arg) for arg in call.args]
                kwargs = {
                    kw.arg: self._get_arg_representation(kw.value) 
                    for kw in call.keywords if kw.arg
                }
                
                # Resolve full class name using import map
                full_class_name = self._resolve_full_name(class_name)
                
                instantiation = ClassInstantiation(
                    variable_name=var_name,
                    class_name=class_name,
                    args=args,
                    kwargs=kwargs,
                    line_number=line_num,
                    full_class_name=full_class_name
                )
                
                result.class_instantiations.append(instantiation)
                
                # Track variable type for later method call analysis
                self.variable_types[var_name] = full_class_name or class_name
    
    def _extract_method_call(self, node: ast.Call, result: AnalysisResult):
        """Extract method call information."""
        if isinstance(node.func, ast.Attribute):
            line_num = getattr(node, 'lineno', 0)
            obj_name = self._get_name_from_node(node.func.value)
            method_name = node.func.attr
            
            if obj_name:
                args = [self._get_arg_representation(arg) for arg in node.args]
                kwargs = {
                    kw.arg: self._get_arg_representation(kw.value) 
                    for kw in node.keywords if kw.arg
                }
                
                method_call = MethodCall(
                    object_name=obj_name,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                    line_number=line_num
                )
                
                result.method_calls.append(method_call)
    
    def _extract_function_call(self, node: ast.Call, result: AnalysisResult):
        """Extract function call information."""
        if isinstance(node.func, ast.Name):
            line_num = getattr(node, 'lineno', 0)
            func_name = node.func.id
            
            args = [self._get_arg_representation(arg) for arg in node.args]
            kwargs = {
                kw.arg: self._get_arg_representation(kw.value) 
                for kw in node.keywords if kw.arg
            }
            
            # Resolve full function name using import map
            full_func_name = self._resolve_full_name(func_name)
            
            function_call = FunctionCall(
                function_name=func_name,
                args=args,
                kwargs=kwargs,
                line_number=line_num,
                full_name=full_func_name
            )
            
            result.function_calls.append(function_call)
    
    def _extract_attribute_access(self, node: ast.Attribute, result: AnalysisResult):
        """Extract attribute access information."""
        line_num = getattr(node, 'lineno', 0)
        obj_name = self._get_name_from_node(node.value)
        attr_name = node.attr
        
        if obj_name:
            attr_access = AttributeAccess(
                object_name=obj_name,
                attribute_name=attr_name,
                line_number=line_num
            )
            
            result.attribute_accesses.append(attr_access)
    
    def _get_name_from_call(self, func_node: ast.AST) -> Optional[str]:
        """Get function/class name from call node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None
    
    def _get_name_from_node(self, node: ast.AST) -> Optional[str]:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For chained attributes, get the base name
            base = self._get_name_from_node(node.value)
            return base if base else None
        return None
    
    def _get_arg_representation(self, arg: ast.AST) -> str:
        """Get string representation of an argument."""
        if isinstance(arg, ast.Constant):
            return repr(arg.value)
        elif isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Attribute):
            obj_name = self._get_name_from_node(arg.value)
            return f"{obj_name}.{arg.attr}" if obj_name else arg.attr
        else:
            return "<complex_expression>"
    
    def _resolve_full_name(self, name: str) -> Optional[str]:
        """Resolve a name to its full module.name using import map."""
        # Check if it's a direct import mapping
        if name in self.import_map:
            return self.import_map[name]
        
        # Check if it's a dotted name with first part in import map
        parts = name.split('.')
        if len(parts) > 1 and parts[0] in self.import_map:
            base_module = self.import_map[parts[0]]
            return f"{base_module}.{'.'.join(parts[1:])}"
        
        return None
    
    def _is_likely_class_instantiation(self, func_name: str, full_name: Optional[str]) -> bool:
        """Determine if a function call is likely a class instantiation."""
        # Check if it's a known imported class (classes typically start with uppercase)
        if func_name and func_name[0].isupper():
            return True
        
        # Check if the full name suggests a class (contains known class patterns)
        if full_name:
            # Common class patterns in module names
            class_patterns = [
                'Model', 'Provider', 'Client', 'Agent', 'Manager', 'Handler',
                'Builder', 'Factory', 'Service', 'Controller', 'Processor'
            ]
            return any(pattern in full_name for pattern in class_patterns)
        
        return False
    
    def _extract_nested_class_instantiation(self, node: ast.Call, result: AnalysisResult):
        """Extract class instantiation that's not assigned to a variable."""
        line_num = getattr(node, 'lineno', 0)
        
        if isinstance(node.func, ast.Name):
            class_name = node.func.id
            
            args = [self._get_arg_representation(arg) for arg in node.args]
            kwargs = {
                kw.arg: self._get_arg_representation(kw.value) 
                for kw in node.keywords if kw.arg
            }
            
            # Resolve full class name using import map
            full_class_name = self._resolve_full_name(class_name)
            
            # Use a synthetic variable name since this isn't assigned to a variable
            var_name = f"<{class_name.lower()}_instance>"
            
            instantiation = ClassInstantiation(
                variable_name=var_name,
                class_name=class_name,
                args=args,
                kwargs=kwargs,
                line_number=line_num,
                full_class_name=full_class_name
            )
            
            result.class_instantiations.append(instantiation)
    
    def _track_method_result_assignment(self, call_node: ast.Call, var_name: str):
        """Track variable assignment from method call results."""
        # This could be enhanced to track return types from method calls
        pass
    
    def _infer_object_types(self, result: AnalysisResult):
        """Update object types for method calls and attribute accesses."""
        for method_call in result.method_calls:
            if not method_call.object_type:
                method_call.object_type = self.variable_types.get(method_call.object_name)
        
        for attr_access in result.attribute_accesses:
            if not attr_access.object_type:
                attr_access.object_type = self.variable_types.get(attr_access.object_name)
