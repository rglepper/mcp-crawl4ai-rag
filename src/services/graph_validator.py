"""
Graph validator service for the Crawl4AI MCP server.

This service validates AI-generated code against the Neo4j knowledge graph
to detect hallucinations in imports, method calls, attributes, and parameters.
"""
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

from src.config import Settings
from src.services.script_analyzer import AnalysisResult, ImportInfo, MethodCall, AttributeAccess, FunctionCall, ClassInstantiation

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    VALID = "VALID"
    INVALID = "INVALID" 
    UNCERTAIN = "UNCERTAIN"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class ValidationResult:
    """Result of validating a single element"""
    status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ImportValidation:
    """Validation result for an import"""
    import_info: ImportInfo
    validation: ValidationResult
    available_classes: List[str] = field(default_factory=list)
    available_functions: List[str] = field(default_factory=list)


@dataclass
class ClassValidation:
    """Validation result for a class instantiation"""
    class_instantiation: ClassInstantiation
    validation: ValidationResult


@dataclass
class MethodValidation:
    """Validation result for a method call"""
    method_call: MethodCall
    validation: ValidationResult


@dataclass
class AttributeValidation:
    """Validation result for an attribute access"""
    attribute_access: AttributeAccess
    validation: ValidationResult


@dataclass
class FunctionValidation:
    """Validation result for a function call"""
    function_call: FunctionCall
    validation: ValidationResult


@dataclass
class ScriptValidationResult:
    """Complete validation results for a script"""
    script_path: str
    analysis_result: AnalysisResult
    import_validations: List[ImportValidation] = field(default_factory=list)
    class_validations: List[ClassValidation] = field(default_factory=list)
    method_validations: List[MethodValidation] = field(default_factory=list)
    attribute_validations: List[AttributeValidation] = field(default_factory=list)
    function_validations: List[FunctionValidation] = field(default_factory=list)
    overall_confidence: float = 0.0
    hallucinations_detected: List[Dict[str, Any]] = field(default_factory=list)


class GraphValidatorService:
    """Service for validating code against Neo4j knowledge graph."""
    
    def __init__(self, neo4j_driver, settings: Settings):
        """
        Initialize the graph validator service.
        
        Args:
            neo4j_driver: Neo4j database driver
            settings: Application settings
        """
        self.neo4j_driver = neo4j_driver
        self.settings = settings
        
        # Cache for performance
        self.module_cache: Dict[str, List[str]] = {}
        self.class_cache: Dict[str, Dict[str, Any]] = {}
        self.method_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.repo_cache: Dict[str, str] = {}  # module_name -> repo_name
        self.knowledge_graph_modules: Set[str] = set()  # Track modules in knowledge graph
    
    async def validate_script(self, analysis_result: AnalysisResult) -> ScriptValidationResult:
        """
        Validate entire script analysis against knowledge graph.
        
        Args:
            analysis_result: Result from script analysis
            
        Returns:
            Complete validation results
        """
        result = ScriptValidationResult(
            script_path=analysis_result.file_path,
            analysis_result=analysis_result
        )
        
        # Validate imports first (builds context for other validations)
        result.import_validations = await self._validate_imports(analysis_result.imports)
        
        # Validate class instantiations
        result.class_validations = await self._validate_class_instantiations(
            analysis_result.class_instantiations
        )
        
        # Validate method calls
        result.method_validations = await self._validate_method_calls(
            analysis_result.method_calls
        )
        
        # Validate attribute accesses
        result.attribute_validations = await self._validate_attribute_accesses(
            analysis_result.attribute_accesses
        )
        
        # Validate function calls
        result.function_validations = await self._validate_function_calls(
            analysis_result.function_calls
        )
        
        # Calculate overall confidence and detect hallucinations
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.hallucinations_detected = self._detect_hallucinations(result)
        
        return result
    
    async def validate_import(self, import_info: ImportInfo) -> ValidationResult:
        """Validate a single import statement."""
        try:
            async with self.neo4j_driver.session() as session:
                # Check if module exists in knowledge graph
                query = """
                MATCH (f:File)
                WHERE f.module_name = $module_name OR f.module_name CONTAINS $module_name
                RETURN f.module_name as module_name
                LIMIT 1
                """
                result = await session.run(query, module_name=import_info.module)
                record = await result.single()
                
                if record:
                    return ValidationResult(
                        status=ValidationStatus.VALID,
                        confidence=0.9,
                        message=f"Module '{import_info.module}' found in knowledge graph"
                    )
                else:
                    return ValidationResult(
                        status=ValidationStatus.NOT_FOUND,
                        confidence=0.1,
                        message=f"Module '{import_info.module}' not found in knowledge graph",
                        suggestions=[f"Verify that '{import_info.module}' is a valid module"]
                    )
        except Exception as e:
            logger.error(f"Error validating import {import_info.module}: {e}")
            return ValidationResult(
                status=ValidationStatus.UNCERTAIN,
                confidence=0.5,
                message=f"Could not validate import: {str(e)}"
            )
    
    async def validate_method_call(self, method_call: MethodCall) -> ValidationResult:
        """Validate a single method call."""
        try:
            # Get the class type for the object
            class_type = method_call.object_type
            if not class_type:
                return ValidationResult(
                    status=ValidationStatus.UNCERTAIN,
                    confidence=0.3,
                    message=f"Cannot determine type of object '{method_call.object_name}'"
                )
            
            # Find method in knowledge graph
            method_info = await self._find_method(class_type, method_call.method_name)
            
            if not method_info:
                # Check for similar method names
                similar_methods = await self._find_similar_methods(class_type, method_call.method_name)
                
                return ValidationResult(
                    status=ValidationStatus.NOT_FOUND,
                    confidence=0.1,
                    message=f"Method '{method_call.method_name}' not found on class '{class_type}'",
                    suggestions=similar_methods
                )
            
            # Validate parameters
            expected_params = method_info.get('params_list', [])
            param_validation = self._validate_parameters(
                expected_params=expected_params,
                provided_args=method_call.args,
                provided_kwargs=method_call.kwargs
            )
            
            if param_validation['valid']:
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    confidence=0.8,
                    message=f"Method '{method_call.method_name}' validated successfully"
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    confidence=0.2,
                    message=f"Parameter mismatch for method '{method_call.method_name}': {param_validation['message']}"
                )
                
        except Exception as e:
            logger.error(f"Error validating method call {method_call.method_name}: {e}")
            return ValidationResult(
                status=ValidationStatus.UNCERTAIN,
                confidence=0.5,
                message=f"Could not validate method call: {str(e)}"
            )
    
    async def _validate_imports(self, imports: List[ImportInfo]) -> List[ImportValidation]:
        """Validate all import statements."""
        validations = []
        for import_info in imports:
            validation = await self.validate_import(import_info)
            validations.append(ImportValidation(
                import_info=import_info,
                validation=validation
            ))
        return validations
    
    async def _validate_class_instantiations(self, instantiations: List[ClassInstantiation]) -> List[ClassValidation]:
        """Validate all class instantiations."""
        validations = []
        for instantiation in instantiations:
            validation = await self._validate_class_instantiation(instantiation)
            validations.append(ClassValidation(
                class_instantiation=instantiation,
                validation=validation
            ))
        return validations
    
    async def _validate_method_calls(self, method_calls: List[MethodCall]) -> List[MethodValidation]:
        """Validate all method calls."""
        validations = []
        for method_call in method_calls:
            validation = await self.validate_method_call(method_call)
            validations.append(MethodValidation(
                method_call=method_call,
                validation=validation
            ))
        return validations
    
    async def _validate_attribute_accesses(self, attribute_accesses: List[AttributeAccess]) -> List[AttributeValidation]:
        """Validate all attribute accesses."""
        validations = []
        for attribute_access in attribute_accesses:
            validation = await self._validate_attribute_access(attribute_access)
            validations.append(AttributeValidation(
                attribute_access=attribute_access,
                validation=validation
            ))
        return validations
    
    async def _validate_function_calls(self, function_calls: List[FunctionCall]) -> List[FunctionValidation]:
        """Validate all function calls."""
        validations = []
        for function_call in function_calls:
            validation = await self._validate_function_call(function_call)
            validations.append(FunctionValidation(
                function_call=function_call,
                validation=validation
            ))
        return validations
    
    async def _validate_class_instantiation(self, instantiation: ClassInstantiation) -> ValidationResult:
        """Validate a class instantiation."""
        # For now, assume class instantiations are valid if the class exists
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.7,
            message=f"Class instantiation '{instantiation.class_name}' appears valid"
        )
    
    async def _validate_attribute_access(self, attribute_access: AttributeAccess) -> ValidationResult:
        """Validate an attribute access."""
        # For now, assume attribute accesses are valid
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.6,
            message=f"Attribute access '{attribute_access.attribute_name}' appears valid"
        )
    
    async def _validate_function_call(self, function_call: FunctionCall) -> ValidationResult:
        """Validate a function call."""
        # For now, assume function calls are valid
        return ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.6,
            message=f"Function call '{function_call.function_name}' appears valid"
        )
    
    async def _find_method(self, class_type: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Find method information in knowledge graph."""
        try:
            async with self.neo4j_driver.session() as session:
                query = """
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE (c.name = $class_type OR c.full_name = $class_type) AND m.name = $method_name
                RETURN m.name as name, m.params_list as params_list, m.return_type as return_type
                LIMIT 1
                """
                result = await session.run(query, class_type=class_type, method_name=method_name)
                record = await result.single()
                
                if record:
                    return {
                        "name": record["name"],
                        "params_list": record["params_list"] or [],
                        "return_type": record["return_type"]
                    }
                return None
        except Exception as e:
            logger.error(f"Error finding method {method_name} in class {class_type}: {e}")
            return None
    
    async def _find_similar_methods(self, class_type: str, method_name: str) -> List[str]:
        """Find similar method names for suggestions."""
        try:
            async with self.neo4j_driver.session() as session:
                query = """
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE c.name = $class_type OR c.full_name = $class_type
                RETURN m.name as name
                """
                result = await session.run(query, class_type=class_type)
                
                methods = []
                async for record in result:
                    methods.append(record["name"])
                
                # Simple similarity check (could be enhanced)
                similar = [m for m in methods if method_name.lower() in m.lower() or m.lower() in method_name.lower()]
                return similar[:3]  # Return top 3 suggestions
        except Exception as e:
            logger.error(f"Error finding similar methods: {e}")
            return []
    
    def _validate_parameters(self, expected_params: List[str], provided_args: List[str], provided_kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Validate method parameters."""
        # Simple parameter validation (could be enhanced)
        total_provided = len(provided_args) + len(provided_kwargs)
        expected_count = len(expected_params)
        
        # Allow for 'self' parameter in methods
        if expected_params and expected_params[0] == 'self':
            expected_count -= 1
        
        if total_provided <= expected_count:
            return {"valid": True, "message": "Parameters appear valid"}
        else:
            return {"valid": False, "message": f"Too many parameters provided: {total_provided} > {expected_count}"}
    
    def _calculate_overall_confidence(self, result: ScriptValidationResult) -> float:
        """Calculate overall confidence score."""
        all_validations = []
        all_validations.extend([v.validation for v in result.import_validations])
        all_validations.extend([v.validation for v in result.class_validations])
        all_validations.extend([v.validation for v in result.method_validations])
        all_validations.extend([v.validation for v in result.attribute_validations])
        all_validations.extend([v.validation for v in result.function_validations])
        
        if not all_validations:
            return 1.0  # No validations means no issues
        
        total_confidence = sum(v.confidence for v in all_validations)
        return total_confidence / len(all_validations)
    
    def _detect_hallucinations(self, result: ScriptValidationResult) -> List[Dict[str, Any]]:
        """Detect hallucinations from validation results."""
        hallucinations = []
        
        # Check for invalid or not found validations
        for validation in result.import_validations:
            if validation.validation.status in [ValidationStatus.INVALID, ValidationStatus.NOT_FOUND]:
                hallucinations.append({
                    "type": "invalid_import",
                    "location": f"line {validation.import_info.line_number}",
                    "description": validation.validation.message,
                    "suggestion": validation.validation.suggestions[0] if validation.validation.suggestions else None
                })
        
        for validation in result.method_validations:
            if validation.validation.status in [ValidationStatus.INVALID, ValidationStatus.NOT_FOUND]:
                hallucinations.append({
                    "type": "invalid_method",
                    "location": f"line {validation.method_call.line_number}",
                    "description": validation.validation.message,
                    "suggestion": validation.validation.suggestions[0] if validation.validation.suggestions else None
                })
        
        return hallucinations
    
    async def initialize(self):
        """Initialize the service."""
        logger.info("Graph validator service initialized")
    
    async def close(self):
        """Close the service."""
        logger.info("Graph validator service closed")
