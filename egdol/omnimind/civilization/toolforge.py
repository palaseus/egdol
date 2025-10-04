"""
Toolforge Subsystem
Autonomous tool creation, testing, and integration system.
"""

import ast
import inspect
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import json
import subprocess
import tempfile
import os
from pathlib import Path

from ..conversational.personality_framework import Personality, PersonalityType


class ToolType(Enum):
    """Tool type enumeration."""
    REASONING = auto()
    PARSER = auto()
    SIMULATION = auto()
    ANALYSIS = auto()
    OPTIMIZATION = auto()
    PREDICTION = auto()
    VISUALIZATION = auto()
    COMMUNICATION = auto()


class ToolStatus(Enum):
    """Tool status enumeration."""
    PROPOSED = auto()
    DESIGNING = auto()
    TESTING = auto()
    INTEGRATING = auto()
    ACTIVE = auto()
    DEPRECATED = auto()
    FAILED = auto()


@dataclass
class ToolSpecification:
    """Tool specification for creation."""
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    proposer_agent_id: str
    requirements: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)  # param_name -> type
    outputs: Dict[str, str] = field(default_factory=dict)  # output_name -> type
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.name,
            "proposer_agent_id": self.proposer_agent_id,
            "requirements": self.requirements,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ToolImplementation:
    """Tool implementation."""
    tool_id: str
    code: str
    tests: str
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "code": self.code,
            "tests": self.tests,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat()
        }


@dataclass
class ToolTestResult:
    """Tool test result."""
    tool_id: str
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "test_name": test_name,
            "passed": self.passed,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ToolUsage:
    """Tool usage tracking."""
    tool_id: str
    user_agent_id: str
    success: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "user_agent_id": self.user_agent_id,
            "success": self.success,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class Toolforge:
    """Toolforge subsystem for autonomous tool creation."""
    
    def __init__(self):
        self.tools: Dict[str, ToolImplementation] = {}
        self.specifications: Dict[str, ToolSpecification] = {}
        self.test_results: List[ToolTestResult] = []
        self.usage_history: List[ToolUsage] = []
        self.tool_status: Dict[str, ToolStatus] = {}
        self.tool_directory = Path("toolforge")
        self.tool_directory.mkdir(exist_ok=True)
        
    def propose_tool(self, 
                    name: str,
                    description: str,
                    tool_type: ToolType,
                    proposer_agent_id: str,
                    requirements: List[str] = None,
                    inputs: Dict[str, str] = None,
                    outputs: Dict[str, str] = None,
                    constraints: List[str] = None,
                    success_criteria: List[str] = None) -> str:
        """Propose a new tool."""
        tool_id = str(uuid.uuid4())
        
        specification = ToolSpecification(
            tool_id=tool_id,
            name=name,
            description=description,
            tool_type=tool_type,
            proposer_agent_id=proposer_agent_id,
            requirements=requirements or [],
            inputs=inputs or {},
            outputs=outputs or {},
            constraints=constraints or [],
            success_criteria=success_criteria or []
        )
        
        self.specifications[tool_id] = specification
        self.tool_status[tool_id] = ToolStatus.PROPOSED
        
        return tool_id
    
    def design_tool(self, tool_id: str, designer_agent_id: str) -> bool:
        """Design tool implementation."""
        if tool_id not in self.specifications:
            return False
        
        specification = self.specifications[tool_id]
        self.tool_status[tool_id] = ToolStatus.DESIGNING
        
        # Generate tool implementation based on specification
        implementation = self._generate_tool_implementation(specification, designer_agent_id)
        
        if implementation:
            self.tools[tool_id] = implementation
            self.tool_status[tool_id] = ToolStatus.TESTING
            return True
        
        self.tool_status[tool_id] = ToolStatus.FAILED
        return False
    
    def _generate_tool_implementation(self, 
                                   specification: ToolSpecification, 
                                   designer_agent_id: str) -> Optional[ToolImplementation]:
        """Generate tool implementation."""
        # This is a simplified implementation - in reality, this would use AI/ML
        # to generate actual code based on the specification
        
        tool_type = specification.tool_type
        name = specification.name
        
        if tool_type == ToolType.REASONING:
            code = self._generate_reasoning_tool(specification)
        elif tool_type == ToolType.PARSER:
            code = self._generate_parser_tool(specification)
        elif tool_type == ToolType.SIMULATION:
            code = self._generate_simulation_tool(specification)
        elif tool_type == ToolType.ANALYSIS:
            code = self._generate_analysis_tool(specification)
        elif tool_type == ToolType.OPTIMIZATION:
            code = self._generate_optimization_tool(specification)
        elif tool_type == ToolType.PREDICTION:
            code = self._generate_prediction_tool(specification)
        elif tool_type == ToolType.VISUALIZATION:
            code = self._generate_visualization_tool(specification)
        elif tool_type == ToolType.COMMUNICATION:
            code = self._generate_communication_tool(specification)
        else:
            return None
        
        # Generate tests
        tests = self._generate_tests(specification)
        
        # Generate dependencies
        dependencies = self._extract_dependencies(code)
        
        return ToolImplementation(
            tool_id=specification.tool_id,
            code=code,
            tests=tests,
            dependencies=dependencies,
            metadata={
                "designer_agent_id": designer_agent_id,
                "specification": specification.to_dict()
            }
        )
    
    def _generate_reasoning_tool(self, specification: ToolSpecification) -> str:
        """Generate reasoning tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(*args, **kwargs):
    """
    {specification.description}
    """
    # Reasoning tool implementation
    result = {{}}
    
    # Process inputs
    for key, value in kwargs.items():
        if key in {list(specification.inputs.keys())}:
            result[key] = value
    
    # Perform reasoning
    reasoning_result = perform_reasoning(kwargs)
    result["reasoning"] = reasoning_result
    
    return result

def perform_reasoning(inputs):
    """Perform reasoning on inputs."""
    # Simplified reasoning logic
    return {{"conclusion": "Reasoning completed", "confidence": 0.8}}
'''
    
    def _generate_parser_tool(self, specification: ToolSpecification) -> str:
        """Generate parser tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(text, *args, **kwargs):
    """
    {specification.description}
    """
    # Parser tool implementation
    result = {{}}
    
    # Parse text
    parsed_elements = parse_text(text)
    result["parsed"] = parsed_elements
    
    # Extract structured data
    structured_data = extract_structured_data(parsed_elements)
    result["structured"] = structured_data
    
    return result

def parse_text(text):
    """Parse text into elements."""
    # Simplified parsing logic
    return text.split()
    
def extract_structured_data(elements):
    """Extract structured data from elements."""
    return {{"elements": elements, "count": len(elements)}}
'''
    
    def _generate_simulation_tool(self, specification: ToolSpecification) -> str:
        """Generate simulation tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(*args, **kwargs):
    """
    {specification.description}
    """
    # Simulation tool implementation
    result = {{}}
    
    # Run simulation
    simulation_result = run_simulation(kwargs)
    result["simulation"] = simulation_result
    
    return result

def run_simulation(params):
    """Run simulation with parameters."""
    # Simplified simulation logic
    return {{"status": "completed", "steps": 100, "result": "success"}}
'''
    
    def _generate_analysis_tool(self, specification: ToolSpecification) -> str:
        """Generate analysis tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(data, *args, **kwargs):
    """
    {specification.description}
    """
    # Analysis tool implementation
    result = {{}}
    
    # Perform analysis
    analysis_result = perform_analysis(data, kwargs)
    result["analysis"] = analysis_result
    
    return result

def perform_analysis(data, params):
    """Perform analysis on data."""
    # Simplified analysis logic
    return {{"insights": ["insight1", "insight2"], "confidence": 0.9}}
'''
    
    def _generate_optimization_tool(self, specification: ToolSpecification) -> str:
        """Generate optimization tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(*args, **kwargs):
    """
    {specification.description}
    """
    # Optimization tool implementation
    result = {{}}
    
    # Perform optimization
    optimization_result = perform_optimization(kwargs)
    result["optimization"] = optimization_result
    
    return result

def perform_optimization(params):
    """Perform optimization."""
    # Simplified optimization logic
    return {{"optimal_solution": "solution", "fitness": 0.95}}
'''
    
    def _generate_prediction_tool(self, specification: ToolSpecification) -> str:
        """Generate prediction tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(*args, **kwargs):
    """
    {specification.description}
    """
    # Prediction tool implementation
    result = {{}}
    
    # Make prediction
    prediction_result = make_prediction(kwargs)
    result["prediction"] = prediction_result
    
    return result

def make_prediction(params):
    """Make prediction based on parameters."""
    # Simplified prediction logic
    return {{"prediction": "future_state", "confidence": 0.8}}
'''
    
    def _generate_visualization_tool(self, specification: ToolSpecification) -> str:
        """Generate visualization tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(data, *args, **kwargs):
    """
    {specification.description}
    """
    # Visualization tool implementation
    result = {{}}
    
    # Create visualization
    visualization_result = create_visualization(data, kwargs)
    result["visualization"] = visualization_result
    
    return result

def create_visualization(data, params):
    """Create visualization from data."""
    # Simplified visualization logic
    return {{"type": "chart", "data": data}}
'''
    
    def _generate_communication_tool(self, specification: ToolSpecification) -> str:
        """Generate communication tool code."""
        return f'''
def {specification.name.lower().replace(" ", "_")}(message, *args, **kwargs):
    """
    {specification.description}
    """
    # Communication tool implementation
    result = {{}}
    
    # Process communication
    communication_result = process_communication(message, kwargs)
    result["communication"] = communication_result
    
    return result

def process_communication(message, params):
    """Process communication message."""
    # Simplified communication logic
    return {{"response": "processed", "sentiment": "positive"}}
'''
    
    def _generate_tests(self, specification: ToolSpecification) -> str:
        """Generate tests for tool."""
        tool_name = specification.name.lower().replace(" ", "_")
        
        return f'''
import unittest
import sys
import os

# Add tool directory to path
sys.path.append(os.path.dirname(__file__))

class Test{specification.name.replace(" ", "")}(unittest.TestCase):
    """Test cases for {specification.name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tool = __import__("{tool_name}")
    
    def test_basic_functionality(self):
        """Test basic tool functionality."""
        # Test with sample inputs
        result = self.tool.{tool_name}()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
    
    def test_input_validation(self):
        """Test input validation."""
        # Test with invalid inputs
        with self.assertRaises(Exception):
            self.tool.{tool_name}(invalid_input="test")
    
    def test_output_format(self):
        """Test output format."""
        result = self.tool.{tool_name}()
        # Verify output format matches specification
        for output_name in {list(specification.outputs.keys())}:
            self.assertIn(output_name, result)
    
    def test_performance(self):
        """Test tool performance."""
        import time
        
        start_time = time.time()
        result = self.tool.{tool_name}()
        execution_time = time.time() - start_time
        
        # Tool should complete within reasonable time
        self.assertLess(execution_time, 5.0)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with edge cases
        try:
            result = self.tool.{tool_name}(edge_case="test")
            # Should not raise exception
        except Exception as e:
            # If exception is raised, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, TypeError, KeyError))

if __name__ == "__main__":
    unittest.main()
'''
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []
        
        # Simple dependency extraction
        if "import " in code:
            lines = code.split('\n')
            for line in lines:
                if line.strip().startswith('import '):
                    module = line.strip().split('import ')[1].split(' ')[0]
                    dependencies.append(module)
                elif line.strip().startswith('from '):
                    module = line.strip().split('from ')[1].split(' ')[0]
                    dependencies.append(module)
        
        return list(set(dependencies))
    
    def test_tool(self, tool_id: str) -> List[ToolTestResult]:
        """Test tool implementation."""
        if tool_id not in self.tools:
            return []
        
        implementation = self.tools[tool_id]
        test_results = []
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(implementation.code)
            f.write('\n\n')
            f.write(implementation.tests)
            test_file = f.name
        
        try:
            # Run tests
            result = subprocess.run(
                ['python', '-m', 'unittest', test_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse test results
            test_output = result.stdout
            test_errors = result.stderr
            
            # Create test result
            test_result = ToolTestResult(
                tool_id=tool_id,
                test_name="full_test_suite",
                passed=result.returncode == 0,
                error_message=test_errors if result.returncode != 0 else None,
                execution_time=0.0,  # Would need to measure
                memory_usage=0.0   # Would need to measure
            )
            
            test_results.append(test_result)
            
        except subprocess.TimeoutExpired:
            test_result = ToolTestResult(
                tool_id=tool_id,
                test_name="full_test_suite",
                passed=False,
                error_message="Test timeout",
                execution_time=30.0
            )
            test_results.append(test_result)
        
        except Exception as e:
            test_result = ToolTestResult(
                tool_id=tool_id,
                test_name="full_test_suite",
                passed=False,
                error_message=str(e),
                execution_time=0.0
            )
            test_results.append(test_result)
        
        finally:
            # Clean up
            os.unlink(test_file)
        
        # Store test results
        self.test_results.extend(test_results)
        
        return test_results
    
    def integrate_tool(self, tool_id: str) -> bool:
        """Integrate tool into system."""
        if tool_id not in self.tools:
            return False
        
        # Check if tool passed all tests
        tool_test_results = [r for r in self.test_results if r.tool_id == tool_id]
        if not all(r.passed for r in tool_test_results):
            return False
        
        # Save tool to filesystem
        implementation = self.tools[tool_id]
        tool_file = self.tool_directory / f"{tool_id}.py"
        
        with open(tool_file, 'w') as f:
            f.write(implementation.code)
        
        # Update status
        self.tool_status[tool_id] = ToolStatus.ACTIVE
        
        return True
    
    def use_tool(self, tool_id: str, user_agent_id: str, **kwargs) -> Dict[str, Any]:
        """Use tool and track usage."""
        if tool_id not in self.tools or self.tool_status.get(tool_id) != ToolStatus.ACTIVE:
            return {"error": "Tool not available"}
        
        start_time = time.time()
        
        try:
            # Import and use tool
            tool_file = self.tool_directory / f"{tool_id}.py"
            if not tool_file.exists():
                return {"error": "Tool file not found"}
            
            # Execute tool
            exec(open(tool_file).read())
            tool_name = self.specifications[tool_id].name.lower().replace(" ", "_")
            result = locals()[tool_name](**kwargs)
            
            execution_time = time.time() - start_time
            
            # Track usage
            usage = ToolUsage(
                tool_id=tool_id,
                user_agent_id=user_agent_id,
                success=True,
                execution_time=execution_time,
                metadata=kwargs
            )
            self.usage_history.append(usage)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Track failed usage
            usage = ToolUsage(
                tool_id=tool_id,
                user_agent_id=user_agent_id,
                success=False,
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
            self.usage_history.append(usage)
            
            return {"error": str(e)}
    
    def get_tool_statistics(self, tool_id: str) -> Dict[str, Any]:
        """Get tool usage statistics."""
        tool_usages = [u for u in self.usage_history if u.tool_id == tool_id]
        
        if not tool_usages:
            return {"error": "No usage data"}
        
        total_uses = len(tool_usages)
        successful_uses = sum(1 for u in tool_usages if u.success)
        success_rate = successful_uses / total_uses if total_uses > 0 else 0.0
        avg_execution_time = sum(u.execution_time for u in tool_usages) / total_uses
        
        return {
            "tool_id": tool_id,
            "total_uses": total_uses,
            "successful_uses": successful_uses,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "status": self.tool_status.get(tool_id, ToolStatus.PROPOSED).name
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        available_tools = []
        
        for tool_id, status in self.tool_status.items():
            if status == ToolStatus.ACTIVE:
                spec = self.specifications.get(tool_id)
                if spec:
                    available_tools.append({
                        "tool_id": tool_id,
                        "name": spec.name,
                        "description": spec.description,
                        "tool_type": spec.tool_type.name,
                        "statistics": self.get_tool_statistics(tool_id)
                    })
        
        return available_tools
