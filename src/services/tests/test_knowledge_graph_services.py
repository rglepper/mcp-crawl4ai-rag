"""
Tests for Knowledge Graph Services.

This module tests all knowledge graph functionality including script analysis,
hallucination detection, Neo4j parsing, graph validation, and report generation.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

from src.services.script_analyzer import ScriptAnalyzerService
from src.services.hallucination_detector import HallucinationDetectorService
from src.services.neo4j_parser import Neo4jParserService
from src.services.knowledge_graph import KnowledgeGraphService
from src.services.report_generator import ReportGeneratorService
from src.services.graph_validator import GraphValidatorService
from src.models import HallucinationDetectionRequest, HallucinationResult
from src.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    settings.enable_knowledge_graph = True
    return settings


@pytest.fixture
def mock_neo4j_driver():
    """Create mock Neo4j driver for testing."""
    driver = Mock()
    session = AsyncMock()

    # Create a proper async context manager mock
    class MockAsyncContextManager:
        def __init__(self, session):
            self.session = session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    driver.session = Mock(return_value=MockAsyncContextManager(session))
    return driver


@pytest.fixture
def script_analyzer_service(mock_settings):
    """Create ScriptAnalyzerService with mocked dependencies."""
    return ScriptAnalyzerService(mock_settings)


@pytest.fixture
def hallucination_detector_service(mock_neo4j_driver, mock_settings):
    """Create HallucinationDetectorService with mocked dependencies."""
    return HallucinationDetectorService(mock_neo4j_driver, mock_settings)


@pytest.fixture
def neo4j_parser_service(mock_neo4j_driver, mock_settings):
    """Create Neo4jParserService with mocked dependencies."""
    return Neo4jParserService(mock_neo4j_driver, mock_settings)


@pytest.fixture
def knowledge_graph_service(mock_neo4j_driver, mock_settings):
    """Create KnowledgeGraphService with mocked dependencies."""
    return KnowledgeGraphService(mock_neo4j_driver, mock_settings)


@pytest.fixture
def report_generator_service(mock_settings):
    """Create ReportGeneratorService with mocked dependencies."""
    return ReportGeneratorService(mock_settings)


@pytest.fixture
def graph_validator_service(mock_neo4j_driver, mock_settings):
    """Create GraphValidatorService with mocked dependencies."""
    return GraphValidatorService(mock_neo4j_driver, mock_settings)


class TestScriptAnalyzerService:
    """Test script analysis functionality."""

    def test_analyze_simple_script(self, script_analyzer_service, tmp_path):
        """Test analysis of a simple Python script."""
        # Create a simple test script
        script_content = '''
import os
from pathlib import Path

class TestClass:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

def test_function():
    obj = TestClass("test")
    return obj.get_name()

# Actually call the function
result = test_function()
'''
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        result = script_analyzer_service.analyze_script(str(script_path))

        assert result.file_path == str(script_path)
        assert len(result.imports) == 2  # os and pathlib.Path
        assert len(result.class_instantiations) >= 1  # TestClass instantiation
        assert len(result.method_calls) >= 1  # get_name call
        assert len(result.function_calls) >= 1  # test_function
        assert "TestClass" in result.variable_types.values()

    def test_analyze_script_with_errors(self, script_analyzer_service, tmp_path):
        """Test analysis of script with syntax errors."""
        # Create a script with syntax errors
        script_content = '''
import os
class TestClass
    def __init__(self):  # Missing colon above
        pass
'''
        script_path = tmp_path / "bad_script.py"
        script_path.write_text(script_content)

        result = script_analyzer_service.analyze_script(str(script_path))

        assert result.file_path == str(script_path)
        assert len(result.errors) > 0
        assert "Failed to analyze script" in result.errors[0]

    def test_analyze_nonexistent_script(self, script_analyzer_service):
        """Test analysis of nonexistent script file."""
        result = script_analyzer_service.analyze_script("/nonexistent/script.py")

        assert result.file_path == "/nonexistent/script.py"
        assert len(result.errors) > 0
        assert "Failed to analyze script" in result.errors[0]


class TestHallucinationDetectorService:
    """Test hallucination detection functionality."""

    async def test_detect_hallucinations_success(self, hallucination_detector_service, tmp_path):
        """Test successful hallucination detection."""
        # Create a test script
        script_content = '''
import pydantic
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="test", age=25)
'''
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        request = HallucinationDetectionRequest(script_path=script_path)

        # Mock the validation process
        with patch.object(hallucination_detector_service, '_analyze_script') as mock_analyze, \
             patch.object(hallucination_detector_service, '_validate_against_graph') as mock_validate:

            mock_analyze.return_value = Mock(
                errors=[],
                imports=[],
                class_instantiations=[],
                method_calls=[],
                function_calls=[],
                attribute_accesses=[]
            )
            mock_validate.return_value = Mock(
                overall_confidence=0.85,
                hallucinations_detected=[]
            )

            result = await hallucination_detector_service.detect_hallucinations(request)

        assert isinstance(result, HallucinationResult)
        assert result.success is True
        assert result.script_path == str(script_path)
        assert result.confidence_score >= 0.0
        assert result.total_issues >= 0

    async def test_detect_hallucinations_with_issues(self, hallucination_detector_service, tmp_path):
        """Test hallucination detection with detected issues."""
        script_content = '''
import fake_module
from nonexistent import FakeClass

obj = FakeClass()
obj.nonexistent_method()
'''
        script_path = tmp_path / "bad_script.py"
        script_path.write_text(script_content)

        request = HallucinationDetectionRequest(script_path=script_path)

        # Mock validation with detected issues
        with patch.object(hallucination_detector_service, '_analyze_script') as mock_analyze, \
             patch.object(hallucination_detector_service, '_validate_against_graph') as mock_validate:

            mock_analyze.return_value = Mock(
                errors=[],
                imports=[],
                class_instantiations=[],
                method_calls=[],
                function_calls=[],
                attribute_accesses=[]
            )
            mock_validate.return_value = Mock(
                overall_confidence=0.25,
                hallucinations_detected=[
                    {"type": "invalid_import", "description": "Module 'fake_module' not found"},
                    {"type": "invalid_method", "description": "Method 'nonexistent_method' not found"}
                ]
            )

            result = await hallucination_detector_service.detect_hallucinations(request)

        assert isinstance(result, HallucinationResult)
        assert result.success is True
        assert result.total_issues == 2
        assert result.confidence_score == 0.25
        assert len(result.issues) == 2

    async def test_detect_hallucinations_file_not_found(self, hallucination_detector_service):
        """Test hallucination detection with nonexistent file."""
        request = HallucinationDetectionRequest(script_path=Path("/nonexistent/script.py"))

        result = await hallucination_detector_service.detect_hallucinations(request)

        assert isinstance(result, HallucinationResult)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestNeo4jParserService:
    """Test Neo4j repository parsing functionality."""

    async def test_parse_repository_success(self, neo4j_parser_service):
        """Test successful repository parsing."""
        repo_url = "https://github.com/test/repo.git"

        # Mock the parsing process
        with patch.object(neo4j_parser_service, '_clone_repository') as mock_clone, \
             patch.object(neo4j_parser_service, '_analyze_python_files') as mock_analyze, \
             patch.object(neo4j_parser_service, '_store_in_neo4j') as mock_store:

            mock_clone.return_value = "/tmp/test_repo"
            mock_analyze.return_value = [
                {
                    "file_path": "test.py",
                    "module_name": "test",
                    "classes": [{"name": "TestClass", "methods": []}],
                    "functions": [{"name": "test_func"}]
                }
            ]
            mock_store.return_value = {"nodes_created": 10, "relationships_created": 5}

            result = await neo4j_parser_service.parse_repository(repo_url)

        assert result["success"] is True
        assert result["repo_url"] == repo_url
        assert result["nodes_created"] == 10
        assert result["relationships_created"] == 5

    async def test_parse_repository_clone_failure(self, neo4j_parser_service):
        """Test repository parsing with clone failure."""
        repo_url = "https://github.com/invalid/repo.git"

        with patch.object(neo4j_parser_service, '_clone_repository') as mock_clone:
            mock_clone.side_effect = Exception("Clone failed")

            result = await neo4j_parser_service.parse_repository(repo_url)

        assert result["success"] is False
        assert "error" in result
        assert "Clone failed" in result["error"]


class TestKnowledgeGraphService:
    """Test knowledge graph querying functionality."""

    async def test_query_repositories(self, knowledge_graph_service, mock_neo4j_driver):
        """Test querying repositories from knowledge graph."""
        # Mock Neo4j session and results
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = iter([
            {"name": "repo1", "file_count": 10},
            {"name": "repo2", "file_count": 15}
        ])
        mock_session.run.return_value = mock_result

        # Update the mock to return our session
        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_session
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        mock_neo4j_driver.session.return_value = MockAsyncContextManager()

        repositories = await knowledge_graph_service.list_repositories()

        assert len(repositories) == 2
        assert repositories[0]["name"] == "repo1"
        assert repositories[1]["name"] == "repo2"

    async def test_query_classes_in_repository(self, knowledge_graph_service, mock_neo4j_driver):
        """Test querying classes in a specific repository."""
        mock_session = AsyncMock()
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session
        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = iter([
            {"name": "TestClass", "full_name": "module.TestClass"},
            {"name": "AnotherClass", "full_name": "module.AnotherClass"}
        ])
        mock_session.run.return_value = mock_result

        classes = await knowledge_graph_service.list_classes("test_repo")

        assert len(classes) == 2
        assert classes[0]["name"] == "TestClass"
        assert classes[1]["name"] == "AnotherClass"

    async def test_search_method(self, knowledge_graph_service, mock_neo4j_driver):
        """Test searching for methods in knowledge graph."""
        mock_session = AsyncMock()
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session
        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = iter([
            {
                "class_name": "TestClass",
                "method_name": "test_method",
                "params_list": ["self", "param1"],
                "return_type": "str"
            }
        ])
        mock_session.run.return_value = mock_result

        methods = await knowledge_graph_service.search_method("test_method")

        assert len(methods) == 1
        assert methods[0]["method_name"] == "test_method"
        assert methods[0]["class_name"] == "TestClass"


class TestReportGeneratorService:
    """Test report generation functionality."""

    def test_generate_comprehensive_report(self, report_generator_service):
        """Test generation of comprehensive hallucination report."""
        # Mock validation result
        validation_result = Mock()
        validation_result.script_path = "/test/script.py"
        validation_result.overall_confidence = 0.75
        validation_result.hallucinations_detected = [
            {
                "type": "invalid_method",
                "location": "line 10",
                "description": "Method 'fake_method' not found",
                "suggestion": "Use 'real_method' instead"
            }
        ]
        validation_result.analysis_result = Mock()
        validation_result.analysis_result.imports = []
        validation_result.analysis_result.class_instantiations = []
        validation_result.analysis_result.method_calls = []

        # Add validation lists that the report generator expects
        validation_result.import_validations = []
        validation_result.class_validations = []
        validation_result.method_validations = []
        validation_result.attribute_validations = []
        validation_result.function_validations = []

        report = report_generator_service.generate_comprehensive_report(validation_result)

        assert "analysis_metadata" in report
        assert "validation_summary" in report
        assert "hallucinations_detected" in report
        assert report["analysis_metadata"]["script_path"] == "/test/script.py"
        assert report["validation_summary"]["overall_confidence"] == 0.75
        assert len(report["hallucinations_detected"]) == 1

    def test_save_json_report(self, report_generator_service, tmp_path):
        """Test saving report as JSON file."""
        report = {
            "analysis_metadata": {"script_path": "/test/script.py"},
            "validation_summary": {"overall_confidence": 0.8},
            "hallucinations_detected": []
        }

        output_path = tmp_path / "test_report.json"
        report_generator_service.save_json_report(report, str(output_path))

        assert output_path.exists()
        # Verify JSON content can be loaded
        import json
        with open(output_path) as f:
            loaded_report = json.load(f)
        assert loaded_report["validation_summary"]["overall_confidence"] == 0.8

    def test_save_markdown_report(self, report_generator_service, tmp_path):
        """Test saving report as Markdown file."""
        report = {
            "analysis_metadata": {
                "script_path": "/test/script.py",
                "analysis_timestamp": "2024-01-01T00:00:00Z"
            },
            "validation_summary": {"overall_confidence": 0.8},
            "hallucinations_detected": []
        }

        output_path = tmp_path / "test_report.md"
        report_generator_service.save_markdown_report(report, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "# AI Hallucination Detection Report" in content
        assert "/test/script.py" in content


class TestGraphValidatorService:
    """Test graph validation functionality."""

    async def test_validate_imports(self, graph_validator_service, mock_neo4j_driver):
        """Test validation of import statements."""
        mock_session = AsyncMock()
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session
        mock_result = AsyncMock()
        mock_result.single.return_value = {"exists": True}
        mock_session.run.return_value = mock_result

        # Mock import info
        import_info = Mock()
        import_info.module = "pydantic"
        import_info.name = "BaseModel"

        validation = await graph_validator_service.validate_import(import_info)

        assert validation.status == "VALID"
        assert validation.confidence > 0.5

    async def test_validate_method_call(self, graph_validator_service, mock_neo4j_driver):
        """Test validation of method calls."""
        mock_session = AsyncMock()
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session
        mock_result = AsyncMock()
        mock_result.single.return_value = {
            "name": "test_method",
            "params_list": ["self", "param1"],
            "return_type": "str"
        }
        mock_session.run.return_value = mock_result

        # Mock method call
        method_call = Mock()
        method_call.object_name = "obj"
        method_call.method_name = "test_method"
        method_call.args = ["value"]
        method_call.object_type = "TestClass"

        validation = await graph_validator_service.validate_method_call(method_call)

        assert validation.status == "VALID"
        assert validation.confidence > 0.5

    async def test_validate_nonexistent_method(self, graph_validator_service, mock_neo4j_driver):
        """Test validation of nonexistent method calls."""
        mock_session = AsyncMock()
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        # Mock method call for nonexistent method
        method_call = Mock()
        method_call.object_name = "obj"
        method_call.method_name = "fake_method"
        method_call.args = []
        method_call.object_type = "TestClass"

        validation = await graph_validator_service.validate_method_call(method_call)

        assert validation.status == "NOT_FOUND"
        assert validation.confidence < 0.5
