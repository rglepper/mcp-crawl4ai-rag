"""
Report generator service for the Crawl4AI MCP server.

This service handles generating comprehensive reports about AI hallucination
detection results in multiple formats (JSON, Markdown).
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class ReportGeneratorService:
    """Service for generating hallucination detection reports."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the report generator service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.report_timestamp = datetime.now(timezone.utc)
    
    def generate_comprehensive_report(self, validation_result) -> Dict[str, Any]:
        """
        Generate a comprehensive report from validation results.
        
        Args:
            validation_result: Script validation result object
            
        Returns:
            Comprehensive report dictionary
        """
        self.report_timestamp = datetime.now(timezone.utc)
        
        # Extract basic information
        script_path = validation_result.script_path
        overall_confidence = validation_result.overall_confidence
        hallucinations = validation_result.hallucinations_detected
        analysis_result = validation_result.analysis_result
        
        # Calculate validation summary
        validation_summary = self._calculate_validation_summary(validation_result)
        
        # Generate analysis metadata
        analysis_metadata = {
            "script_path": script_path,
            "analysis_timestamp": self.report_timestamp.isoformat(),
            "total_imports": len(analysis_result.imports),
            "total_class_instantiations": len(analysis_result.class_instantiations),
            "total_method_calls": len(analysis_result.method_calls),
            "total_function_calls": len(analysis_result.function_calls),
            "total_attribute_accesses": len(analysis_result.attribute_accesses)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_result)
        
        # Compile comprehensive report
        report = {
            "analysis_metadata": analysis_metadata,
            "validation_summary": validation_summary,
            "hallucinations_detected": hallucinations,
            "recommendations": recommendations,
            "detailed_analysis": {
                "imports": self._format_imports(analysis_result.imports),
                "class_instantiations": self._format_class_instantiations(analysis_result.class_instantiations),
                "method_calls": self._format_method_calls(analysis_result.method_calls),
                "function_calls": self._format_function_calls(analysis_result.function_calls),
                "attribute_accesses": self._format_attribute_accesses(analysis_result.attribute_accesses)
            }
        }
        
        return report
    
    def _calculate_validation_summary(self, validation_result) -> Dict[str, Any]:
        """Calculate validation summary statistics."""
        # Count validation statuses
        valid_count = 0
        invalid_count = 0
        uncertain_count = 0
        not_found_count = 0
        
        # Count from different validation types
        for validation_list in [
            getattr(validation_result, 'import_validations', []),
            getattr(validation_result, 'class_validations', []),
            getattr(validation_result, 'method_validations', []),
            getattr(validation_result, 'attribute_validations', []),
            getattr(validation_result, 'function_validations', [])
        ]:
            for validation in validation_list:
                status = getattr(validation.validation, 'status', 'UNKNOWN')
                if status == 'VALID':
                    valid_count += 1
                elif status == 'INVALID':
                    invalid_count += 1
                elif status == 'UNCERTAIN':
                    uncertain_count += 1
                elif status == 'NOT_FOUND':
                    not_found_count += 1
        
        total_validations = valid_count + invalid_count + uncertain_count + not_found_count
        hallucination_rate = (invalid_count + not_found_count) / total_validations if total_validations > 0 else 0
        
        return {
            "overall_confidence": validation_result.overall_confidence,
            "total_validations": total_validations,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "uncertain_count": uncertain_count,
            "not_found_count": not_found_count,
            "hallucination_rate": hallucination_rate
        }
    
    def _generate_recommendations(self, validation_result) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Count different types of issues
        hallucinations = validation_result.hallucinations_detected
        invalid_imports = sum(1 for h in hallucinations if h.get('type') == 'invalid_import')
        invalid_methods = sum(1 for h in hallucinations if h.get('type') == 'invalid_method')
        invalid_attributes = sum(1 for h in hallucinations if h.get('type') == 'invalid_attribute')
        
        if invalid_imports > 0:
            recommendations.append(f"Verify {invalid_imports} import statement(s) - some modules may not exist")
        
        if invalid_methods > 0:
            recommendations.append(f"Check {invalid_methods} method call(s) - some methods may not exist on the target classes")
        
        if invalid_attributes > 0:
            recommendations.append(f"Validate {invalid_attributes} attribute access(es) - some attributes may not exist")
        
        if validation_result.overall_confidence < 0.5:
            recommendations.append("Consider reviewing the script against official documentation")
            recommendations.append("Test the script in a development environment before production use")
        
        if validation_result.overall_confidence > 0.8:
            recommendations.append("Script appears to be well-structured and valid")
        
        if not recommendations:
            recommendations.append("No specific recommendations - script analysis completed successfully")
        
        return recommendations
    
    def _format_imports(self, imports) -> List[Dict[str, Any]]:
        """Format import information for report."""
        return [
            {
                "module": imp.module,
                "name": imp.name,
                "alias": imp.alias,
                "is_from_import": imp.is_from_import,
                "line_number": imp.line_number
            }
            for imp in imports
        ]
    
    def _format_class_instantiations(self, instantiations) -> List[Dict[str, Any]]:
        """Format class instantiation information for report."""
        return [
            {
                "variable_name": inst.variable_name,
                "class_name": inst.class_name,
                "full_class_name": inst.full_class_name,
                "args": inst.args,
                "kwargs": inst.kwargs,
                "line_number": inst.line_number
            }
            for inst in instantiations
        ]
    
    def _format_method_calls(self, method_calls) -> List[Dict[str, Any]]:
        """Format method call information for report."""
        return [
            {
                "object_name": call.object_name,
                "method_name": call.method_name,
                "object_type": call.object_type,
                "args": call.args,
                "kwargs": call.kwargs,
                "line_number": call.line_number
            }
            for call in method_calls
        ]
    
    def _format_function_calls(self, function_calls) -> List[Dict[str, Any]]:
        """Format function call information for report."""
        return [
            {
                "function_name": call.function_name,
                "full_name": call.full_name,
                "args": call.args,
                "kwargs": call.kwargs,
                "line_number": call.line_number
            }
            for call in function_calls
        ]
    
    def _format_attribute_accesses(self, attribute_accesses) -> List[Dict[str, Any]]:
        """Format attribute access information for report."""
        return [
            {
                "object_name": access.object_name,
                "attribute_name": access.attribute_name,
                "object_type": access.object_type,
                "line_number": access.line_number
            }
            for access in attribute_accesses
        ]
    
    def save_json_report(self, report: Dict[str, Any], output_path: str):
        """
        Save report as JSON file.
        
        Args:
            report: Report dictionary to save
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            raise
    
    def save_markdown_report(self, report: Dict[str, Any], output_path: str):
        """
        Save report as Markdown file.
        
        Args:
            report: Report dictionary to save
            output_path: Path to save the Markdown file
        """
        try:
            md_content = self._generate_markdown_content(report)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Markdown report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save Markdown report: {e}")
            raise
    
    def _generate_markdown_content(self, report: Dict[str, Any]) -> str:
        """Generate Markdown content from report."""
        md = []
        
        # Header
        md.append("# AI Hallucination Detection Report")
        md.append("")
        md.append(f"**Script:** `{report['analysis_metadata']['script_path']}`")
        md.append(f"**Analysis Date:** {report['analysis_metadata']['analysis_timestamp']}")
        md.append(f"**Overall Confidence:** {report['validation_summary']['overall_confidence']:.2%}")
        md.append("")
        
        # Summary
        md.append("## Summary")
        md.append("")
        summary = report['validation_summary']
        md.append(f"- **Total Validations:** {summary['total_validations']}")
        md.append(f"- **Valid:** {summary['valid_count']}")
        md.append(f"- **Invalid:** {summary['invalid_count']}")
        md.append(f"- **Not Found:** {summary['not_found_count']}")
        md.append(f"- **Uncertain:** {summary['uncertain_count']}")
        md.append(f"- **Hallucination Rate:** {summary['hallucination_rate']:.1%}")
        md.append("")
        
        # Hallucinations
        if report['hallucinations_detected']:
            md.append("## 🚨 Hallucinations Detected")
            md.append("")
            for i, hallucination in enumerate(report['hallucinations_detected'], 1):
                md.append(f"### {i}. {hallucination['type'].replace('_', ' ').title()}")
                md.append(f"**Location:** {hallucination.get('location', 'Unknown')}")
                md.append(f"**Description:** {hallucination.get('description', 'No description')}")
                if hallucination.get('suggestion'):
                    md.append(f"**Suggestion:** {hallucination['suggestion']}")
                md.append("")
        else:
            md.append("## ✅ No Hallucinations Detected")
            md.append("")
            md.append("The script appears to be valid based on the knowledge graph analysis.")
            md.append("")
        
        # Recommendations
        if report['recommendations']:
            md.append("## 💡 Recommendations")
            md.append("")
            for rec in report['recommendations']:
                md.append(f"- {rec}")
            md.append("")
        
        # Analysis Details
        md.append("## 📊 Analysis Details")
        md.append("")
        metadata = report['analysis_metadata']
        md.append(f"- **Imports:** {metadata['total_imports']}")
        md.append(f"- **Class Instantiations:** {metadata['total_class_instantiations']}")
        md.append(f"- **Method Calls:** {metadata['total_method_calls']}")
        md.append(f"- **Function Calls:** {metadata['total_function_calls']}")
        md.append(f"- **Attribute Accesses:** {metadata['total_attribute_accesses']}")
        md.append("")
        
        return "\n".join(md)
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Print a concise summary to console.
        
        Args:
            report: Report dictionary to summarize
        """
        print("\n" + "="*80)
        print("🤖 AI HALLUCINATION DETECTION REPORT")
        print("="*80)
        
        print(f"Script: {report['analysis_metadata']['script_path']}")
        print(f"Overall Confidence: {report['validation_summary']['overall_confidence']:.1%}")
        
        summary = report['validation_summary']
        print(f"\nValidation Results:")
        print(f"  ✅ Valid: {summary['valid_count']}")
        print(f"  ❌ Invalid: {summary['invalid_count']}")
        print(f"  🔍 Not Found: {summary['not_found_count']}")
        print(f"  ❓ Uncertain: {summary['uncertain_count']}")
        print(f"  📊 Hallucination Rate: {summary['hallucination_rate']:.1%}")
        
        if report['hallucinations_detected']:
            print(f"\n🚨 {len(report['hallucinations_detected'])} Hallucinations Detected:")
            for hall in report['hallucinations_detected'][:5]:  # Show first 5
                print(f"  - {hall['type'].replace('_', ' ').title()} at {hall.get('location', 'unknown location')}")
                print(f"    {hall.get('description', 'No description')}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations'][:3]:  # Show first 3
                print(f"  - {rec}")
        
        print("="*80)
