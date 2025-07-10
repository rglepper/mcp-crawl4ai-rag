"""
Hallucination detector service for the Crawl4AI MCP server.

This service orchestrates the detection of AI coding assistant hallucinations
by combining AST analysis, knowledge graph validation, and reporting.
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.config import Settings
from src.models import HallucinationDetectionRequest, HallucinationResult
from src.services.script_analyzer import ScriptAnalyzerService, AnalysisResult
from src.services.graph_validator import GraphValidatorService
from src.services.report_generator import ReportGeneratorService

logger = logging.getLogger(__name__)


class HallucinationDetectorService:
    """Service for detecting AI hallucinations in Python scripts."""
    
    def __init__(self, neo4j_driver, settings: Settings):
        """
        Initialize the hallucination detector service.
        
        Args:
            neo4j_driver: Neo4j database driver
            settings: Application settings
        """
        self.neo4j_driver = neo4j_driver
        self.settings = settings
        self.script_analyzer = ScriptAnalyzerService(settings)
        self.graph_validator = GraphValidatorService(neo4j_driver, settings)
        self.report_generator = ReportGeneratorService(settings)
    
    async def detect_hallucinations(self, request: HallucinationDetectionRequest) -> HallucinationResult:
        """
        Detect hallucinations in a Python script.
        
        Args:
            request: Hallucination detection request
            
        Returns:
            HallucinationResult with detection results
        """
        script_path = str(request.script_path)
        
        try:
            # Validate that the script file exists
            if not Path(script_path).exists():
                return HallucinationResult(
                    success=False,
                    script_path=script_path,
                    total_issues=0,
                    confidence_score=0.0,
                    issues=[],
                    recommendations=[],
                    error=f"Script file not found: {script_path}"
                )
            
            logger.info(f"Starting hallucination detection for: {script_path}")
            
            # Step 1: Analyze the script using AST
            logger.info("Step 1: Analyzing script structure...")
            analysis_result = self._analyze_script(script_path)
            
            if analysis_result.errors:
                logger.warning(f"Analysis warnings: {analysis_result.errors}")
                return HallucinationResult(
                    success=False,
                    script_path=script_path,
                    total_issues=len(analysis_result.errors),
                    confidence_score=0.0,
                    issues=[{"type": "analysis_error", "description": error} for error in analysis_result.errors],
                    recommendations=["Fix syntax errors in the script"],
                    error="Script analysis failed due to syntax errors"
                )
            
            logger.info(f"Found: {len(analysis_result.imports)} imports, "
                       f"{len(analysis_result.class_instantiations)} class instantiations, "
                       f"{len(analysis_result.method_calls)} method calls, "
                       f"{len(analysis_result.function_calls)} function calls, "
                       f"{len(analysis_result.attribute_accesses)} attribute accesses")
            
            # Step 2: Validate against knowledge graph
            logger.info("Step 2: Validating against knowledge graph...")
            validation_result = await self._validate_against_graph(analysis_result)
            
            logger.info(f"Validation complete. Overall confidence: {validation_result.overall_confidence:.1%}")
            
            # Step 3: Generate result
            issues = validation_result.hallucinations_detected
            recommendations = self._generate_recommendations(validation_result)
            
            return HallucinationResult(
                success=True,
                script_path=script_path,
                total_issues=len(issues),
                confidence_score=validation_result.overall_confidence,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error during hallucination detection: {str(e)}")
            return HallucinationResult(
                success=False,
                script_path=script_path,
                total_issues=0,
                confidence_score=0.0,
                issues=[],
                recommendations=[],
                error=str(e)
            )
    
    def _analyze_script(self, script_path: str) -> AnalysisResult:
        """Analyze script using AST analyzer."""
        return self.script_analyzer.analyze_script(script_path)
    
    async def _validate_against_graph(self, analysis_result: AnalysisResult):
        """Validate analysis result against knowledge graph."""
        return await self.graph_validator.validate_script(analysis_result)
    
    def _generate_recommendations(self, validation_result) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Count different types of issues
        invalid_imports = sum(1 for h in validation_result.hallucinations_detected 
                             if h.get('type') == 'invalid_import')
        invalid_methods = sum(1 for h in validation_result.hallucinations_detected 
                             if h.get('type') == 'invalid_method')
        invalid_attributes = sum(1 for h in validation_result.hallucinations_detected 
                                if h.get('type') == 'invalid_attribute')
        
        if invalid_imports > 0:
            recommendations.append(f"Verify {invalid_imports} import statement(s) - some modules may not exist")
        
        if invalid_methods > 0:
            recommendations.append(f"Check {invalid_methods} method call(s) - some methods may not exist on the target classes")
        
        if invalid_attributes > 0:
            recommendations.append(f"Validate {invalid_attributes} attribute access(es) - some attributes may not exist")
        
        if validation_result.overall_confidence < 0.5:
            recommendations.append("Consider reviewing the script against official documentation")
            recommendations.append("Test the script in a development environment before production use")
        
        if not recommendations:
            recommendations.append("Script appears to be valid based on knowledge graph analysis")
        
        return recommendations
    
    async def batch_detect(self, script_paths: List[str]) -> List[HallucinationResult]:
        """
        Detect hallucinations in multiple scripts.
        
        Args:
            script_paths: List of paths to Python scripts
            
        Returns:
            List of hallucination detection results
        """
        logger.info(f"Starting batch detection for {len(script_paths)} scripts")
        
        results = []
        for i, script_path in enumerate(script_paths, 1):
            logger.info(f"Processing script {i}/{len(script_paths)}: {script_path}")
            
            try:
                request = HallucinationDetectionRequest(script_path=Path(script_path))
                result = await self.detect_hallucinations(request)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {script_path}: {str(e)}")
                # Create error result
                error_result = HallucinationResult(
                    success=False,
                    script_path=script_path,
                    total_issues=0,
                    confidence_score=0.0,
                    issues=[],
                    recommendations=[],
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def initialize(self):
        """Initialize the service and its dependencies."""
        await self.graph_validator.initialize()
        logger.info("Hallucination detector service initialized successfully")
    
    async def close(self):
        """Close connections and cleanup resources."""
        await self.graph_validator.close()
        logger.info("Hallucination detector service closed")
