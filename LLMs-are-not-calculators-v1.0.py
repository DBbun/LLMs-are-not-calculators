#!/usr/bin/env python3
"""
LLM Education Impact Simulator
Based on "LLMs are not calculators" by Daniel Jackson (2025)

Copyright (c) 2026 DBbun LLC. All rights reserved.

This simulation generates synthetic observational data capturing how students
interact with AI tools (search, explicit-context LLMs, and agentic LLMs) while
completing educational tasks.

Key theoretical foundations from Jackson (2025):
- Context engineering is more critical than prompt engineering
- Intentionality in tool usage affects learning outcomes
- Reading documentation vs. relying on LLMs creates different learning paths
- Module boundaries and separation of concerns matter for complex work
- Partial automation risks and agency preservation

Version: 1.0
Owner: DBbun LLC
Author: Based on Jackson's educational research
License: MIT

Dependencies:
    - Python 3.9+
    - numpy (pip install numpy)
    - pandas (pip install pandas)
    - matplotlib (pip install matplotlib)
    - seaborn (optional, for enhanced visualizations)

Usage:
    python llm_education_simulator.py --config config.json
    python llm_education_simulator.py  # uses default config

Outputs:
    - CSV files: students.csv, tasks.csv, runs.csv, events.csv
    - JSON: config.json, summary.json, metadata.json
    - Figures: Multiple PNG visualizations
    - README.md: Documentation for the dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Figures will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    Simulation configuration with parameters grounded in educational research.
    
    All probability and behavioral parameters are based on observations from
    Jackson (2025) and related educational AI literature.
    """
    # Metadata
    version: str = "2.0"
    description: str = "LLM Education Impact Simulator based on Jackson (2025)"
    
    # Reproducibility
    seed: int = 42
    
    # Output settings
    output_dir: str = "output"
    write_metadata: bool = True
    write_readme: bool = True
    
    # Simulation scale
    n_students: int = 120
    n_tasks: int = 15
    tasks_per_student: int = 5
    
    # Task type distribution (Jackson: coding 55%, writing 30%, reading 15%)
    task_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "coding": 0.55,
        "writing": 0.30,
        "reading": 0.15,
    })
    
    # Tool usage distribution
    # search: traditional documentation-driven approach
    # llm_explicit: LLM with careful context selection (safer)
    # llm_agentic: LLM with automated context (higher risk)
    tool_distribution: Dict[str, float] = field(default_factory=lambda: {
        "search": 0.20,
        "llm_explicit": 0.50,
        "llm_agentic": 0.30,
    })
    
    # Student behavioral traits (mean, std)
    # Based on Jackson's observations about student behavior with LLMs
    
    # Effort minimization: tendency to take shortcuts (Jackson: "urge overwhelming")
    effort_minimization_mean: float = 0.62
    effort_minimization_sd: float = 0.18
    
    # Documentation discipline: reading docs before coding
    # Jackson: novices equipped with LLMs don't read documentation
    doc_discipline_mean: float = 0.42
    doc_discipline_sd: float = 0.18
    
    # Context care: attention to context selection
    # Jackson: "context engineering replacing prompt engineering"
    context_care_mean: float = 0.55
    context_care_sd: float = 0.20
    
    # Base skill level
    skill_mean: float = 0.52
    skill_sd: float = 0.18
    
    # Time pressure: urgency/deadline stress
    time_pressure_mean: float = 0.58
    time_pressure_sd: float = 0.20
    
    # Intentionality: deliberate vs passive tool use
    # Jackson: intentional usage preserves agency
    intentionality_mean: float = 0.48
    intentionality_sd: float = 0.22
    
    # Tool characteristics
    llm_latency_sec_mean: float = 4.0
    llm_latency_sec_sd: float = 1.3
    
    search_latency_sec_mean: float = 2.2
    search_latency_sec_sd: float = 0.8
    
    # Error rates (Jackson: LLMs hallucinate, omit, violate boundaries)
    # Base rates modified by skill and context care
    llm_hallucination_base: float = 0.20
    llm_omission_base: float = 0.25
    
    # Boundary violation rates (Jackson: agentic tools "tread on toes")
    # Especially problematic with poor modularity
    agentic_boundary_violation_base: float = 0.32
    explicit_boundary_violation_base: float = 0.10
    search_boundary_violation_base: float = 0.05
    
    # Grading thresholds
    pass_score_threshold: float = 0.70
    
    # Testing parameters (for coding tasks)
    coding_tests_mean: int = 18
    coding_tests_sd: int = 5
    
    # Outcome weights (how behaviors affect quality)
    # Jackson: documentation reading, testing, avoiding hallucinations all matter
    weight_docs_on_quality: float = 0.22
    weight_testing_on_quality: float = 0.26
    weight_hallucinations_on_quality: float = 0.32
    weight_boundary_violations_on_quality: float = 0.28
    weight_effort_minimization_on_quality: float = 0.18
    weight_time_pressure_on_quality: float = 0.16
    weight_intentionality_on_quality: float = 0.24
    weight_reading_on_quality: float = 0.20
    
    # Event granularity caps
    max_prompt_rounds: int = 7
    max_edit_rounds: int = 10
    max_doc_opens: int = 12
    max_test_runs: int = 8
    max_reading_sessions: int = 6
    
    # Visualization settings
    make_figures: bool = True
    figure_dpi: int = 150
    figure_format: str = "png"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_distributions()
        
    def _validate_distributions(self):
        """Ensure probability distributions sum to 1.0."""
        for name, dist in [
            ("task_type_weights", self.task_type_weights),
            ("tool_distribution", self.tool_distribution)
        ]:
            total = sum(dist.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"{name} must sum to 1.0, got {total:.3f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling nested dataclasses."""
        return asdict(self)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# DOMAIN MODELS
# =============================================================================

@dataclass
class Student:
    """
    Represents a student with behavioral and skill characteristics.
    
    Traits are based on Jackson's observations about how students interact
    with LLMs in educational settings.
    """
    student_id: str
    skill: float  # Base ability (0-1)
    effort_minimization: float  # Tendency to take shortcuts (0-1)
    doc_discipline: float  # Reads documentation before acting (0-1)
    context_care: float  # Attention to context selection (0-1)
    time_pressure: float  # Stress/urgency level (0-1)
    intentionality: float  # Deliberate vs passive tool use (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


@dataclass
class Task:
    """
    Represents an educational task (coding, writing, or reading).
    
    Tasks vary in difficulty, structure, and grading criteria.
    """
    task_id: str
    task_type: str  # coding, writing, reading
    difficulty: float  # 0-1 scale
    has_module_boundaries: bool  # Relevant for coding (Jackson: modularity matters)
    rubric_strictness: float  # Grading precision (0-1)
    requires_prior_reading: bool  # Whether background docs needed
    conceptual_complexity: float  # How many interconnected ideas (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


@dataclass
class RunResult:
    """
    Complete record of a student completing a task with a specific tool.
    
    Captures both observable behaviors and outcome metrics.
    """
    # Identifiers
    run_id: str
    student_id: str
    task_id: str
    tool_mode: str
    timestamp: str
    
    # Observable behaviors
    prompts: int
    doc_opens: int
    reading_sessions: int  # Sustained doc reading (Jackson: critical skill)
    edits: int
    test_runs: int
    verification_checks: int  # Checking LLM output (Jackson: checking matters)
    boundary_violations: int
    
    # Error tracking
    hallucinations: int
    hallucinations_caught: int  # Detected and corrected
    omissions: int
    omissions_caught: int
    
    # Time tracking
    latency_sec_total: float
    reading_time_sec: float
    
    # Outcomes
    score: float  # 0-1 rubric score
    passed: int  # 0 or 1
    tests_total: int  # For coding tasks
    tests_failed: int
    
    # Agency and learning proxies
    agency_proxy: float  # Student ownership (0-1)
    cognitive_engagement: float  # Deep vs surface learning (0-1)
    context_quality: float  # How well context was selected (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


@dataclass
class Event:
    """
    Fine-grained event in a student's workflow.
    
    Events provide the raw observational data for analysis.
    """
    timestamp: str
    run_id: str
    student_id: str
    task_id: str
    tool_mode: str
    event_type: str
    value: float
    metadata: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, x))


def truncated_normal(
    rng: random.Random,
    mean: float,
    sd: float,
    low: float = 0.0,
    high: float = 1.0,
    max_attempts: int = 30
) -> float:
    """
    Generate truncated normal random variable.
    
    Falls back to clamped mean if sampling fails.
    """
    for _ in range(max_attempts):
        value = rng.gauss(mean, sd)
        if low <= value <= high:
            return value
    return max(low, min(high, mean))


def positive_normal(
    rng: random.Random,
    mean: float,
    sd: float,
    minimum: float = 0.0
) -> float:
    """Generate positive normal variable with floor."""
    for _ in range(30):
        value = rng.gauss(mean, sd)
        if value >= minimum:
            return value
    return max(minimum, mean)


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    """Select item according to probability weights."""
    items = list(weights.items())
    total = sum(w for _, w in items)
    if total <= 0:
        raise ValueError("Weights must sum to positive value")
    
    r = rng.random() * total
    cumulative = 0.0
    for key, weight in items:
        cumulative += weight
        if r <= cumulative:
            return key
    return items[-1][0]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp(base_time: datetime, offset_seconds: float = 0) -> str:
    """Generate ISO timestamp with optional offset."""
    return (base_time + timedelta(seconds=offset_seconds)).isoformat()


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class EducationSimulator:
    """
    Main simulation engine for generating educational AI usage data.
    
    Implements the theoretical model from Jackson (2025) as a generative
    process that produces observable event traces.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)  # For pandas/numpy operations
        
        self.students: List[Student] = []
        self.tasks: List[Task] = []
        self.runs: List[RunResult] = []
        self.events: List[Event] = []
        
        logger.info(f"Initialized simulator with seed={config.seed}")
    
    def generate_students(self) -> List[Student]:
        """
        Generate student population with varying characteristics.
        
        Behavioral traits follow distributions observed in Jackson's study.
        """
        logger.info(f"Generating {self.config.n_students} students...")
        students = []
        
        for i in range(self.config.n_students):
            student = Student(
                student_id=f"S{i:04d}",
                skill=truncated_normal(
                    self.rng,
                    self.config.skill_mean,
                    self.config.skill_sd
                ),
                effort_minimization=truncated_normal(
                    self.rng,
                    self.config.effort_minimization_mean,
                    self.config.effort_minimization_sd
                ),
                doc_discipline=truncated_normal(
                    self.rng,
                    self.config.doc_discipline_mean,
                    self.config.doc_discipline_sd
                ),
                context_care=truncated_normal(
                    self.rng,
                    self.config.context_care_mean,
                    self.config.context_care_sd
                ),
                time_pressure=truncated_normal(
                    self.rng,
                    self.config.time_pressure_mean,
                    self.config.time_pressure_sd
                ),
                intentionality=truncated_normal(
                    self.rng,
                    self.config.intentionality_mean,
                    self.config.intentionality_sd
                ),
            )
            students.append(student)
        
        return students
    
    def generate_tasks(self) -> List[Task]:
        """
        Generate task pool with varying characteristics.
        
        Task types and properties reflect those in Jackson's course.
        """
        logger.info(f"Generating {self.config.n_tasks} tasks...")
        tasks = []
        
        for i in range(self.config.n_tasks):
            task_type = weighted_choice(self.rng, self.config.task_type_weights)
            
            # Coding tasks tend to be harder and have module boundaries
            if task_type == "coding":
                difficulty_mean = 0.58
                has_boundaries = self.rng.random() < 0.75
                requires_reading = self.rng.random() < 0.80
                complexity = truncated_normal(self.rng, 0.62, 0.15)
            elif task_type == "writing":
                difficulty_mean = 0.50
                has_boundaries = False
                requires_reading = self.rng.random() < 0.60
                complexity = truncated_normal(self.rng, 0.52, 0.18)
            else:  # reading
                difficulty_mean = 0.42
                has_boundaries = False
                requires_reading = True
                complexity = truncated_normal(self.rng, 0.45, 0.16)
            
            task = Task(
                task_id=f"T{i:03d}",
                task_type=task_type,
                difficulty=truncated_normal(self.rng, difficulty_mean, 0.18),
                has_module_boundaries=has_boundaries,
                rubric_strictness=truncated_normal(self.rng, 0.60, 0.15),
                requires_prior_reading=requires_reading,
                conceptual_complexity=complexity,
            )
            tasks.append(task)
        
        return tasks
    
    def simulate_run(
        self,
        student: Student,
        task: Task,
        tool_mode: str,
        run_id: str,
        base_time: datetime
    ) -> RunResult:
        """
        Simulate a single student completing a task with a specific tool.
        
        This is the core generative model that produces observable events
        and outcome metrics based on Jackson's theoretical framework.
        """
        events = []
        time_offset = 0.0
        
        # Track metrics
        prompts = 0
        doc_opens = 0
        reading_sessions = 0
        reading_time = 0.0
        edits = 0
        test_runs = 0
        verification_checks = 0
        boundary_violations = 0
        hallucinations = 0
        hallucinations_caught = 0
        omissions = 0
        omissions_caught = 0
        
        # Start event
        events.append(Event(
            timestamp=get_timestamp(base_time, time_offset),
            run_id=run_id,
            student_id=student.student_id,
            task_id=task.task_id,
            tool_mode=tool_mode,
            event_type="start",
            value=1,
            metadata=""
        ))
        time_offset += 1.0
        
        # === READING PHASE ===
        # Jackson: "novices don't read documentation"
        # Reading behavior depends on doc_discipline and intentionality
        
        if task.requires_prior_reading:
            reading_prob = clamp01(
                0.25 + 0.50 * student.doc_discipline +
                0.30 * student.intentionality -
                0.25 * student.effort_minimization -
                0.20 * student.time_pressure
            )
            
            if tool_mode.startswith("llm"):
                # LLM users read less (Jackson's observation)
                reading_prob *= 0.55
            
            # Expected reading sessions
            if self.rng.random() < reading_prob:
                expected_sessions = (
                    1.0 + 2.5 * student.doc_discipline +
                    1.5 * task.conceptual_complexity +
                    1.0 * student.intentionality
                )
                
                if tool_mode == "llm_agentic":
                    expected_sessions *= 0.60  # Agentic tools reduce reading
                
                reading_sessions = min(
                    self.config.max_reading_sessions,
                    max(0, int(round(expected_sessions)))
                )
                
                for i in range(reading_sessions):
                    session_time = positive_normal(
                        self.rng,
                        3.0 + 4.0 * task.conceptual_complexity,
                        2.0,
                        minimum=0.5
                    )
                    reading_time += session_time
                    time_offset += session_time
                    
                    events.append(Event(
                        timestamp=get_timestamp(base_time, time_offset),
                        run_id=run_id,
                        student_id=student.student_id,
                        task_id=task.task_id,
                        tool_mode=tool_mode,
                        event_type="reading_session",
                        value=session_time,
                        metadata=f"session={i+1}|depth={task.conceptual_complexity:.2f}"
                    ))
        
        # === DOCUMENTATION LOOKUP PHASE ===
        # Quick reference lookups during work (different from sustained reading)
        
        doc_open_prob = clamp01(
            0.30 + 0.45 * student.doc_discipline +
            0.25 * task.difficulty -
            0.20 * student.effort_minimization
        )
        
        if tool_mode.startswith("llm"):
            doc_open_prob *= 0.65  # LLM users consult docs less
        
        expected_opens = (
            2.0 + 6.0 * student.doc_discipline +
            2.5 * task.difficulty
        )
        
        if student.time_pressure > 0.7:
            expected_opens *= 0.70
        
        doc_opens = min(
            self.config.max_doc_opens,
            max(0, int(round(expected_opens * doc_open_prob)))
        )
        
        for i in range(doc_opens):
            lookup_time = positive_normal(
                self.rng,
                self.config.search_latency_sec_mean,
                self.config.search_latency_sec_sd,
                minimum=0.2
            )
            time_offset += lookup_time
            
            events.append(Event(
                timestamp=get_timestamp(base_time, time_offset),
                run_id=run_id,
                student_id=student.student_id,
                task_id=task.task_id,
                tool_mode=tool_mode,
                event_type="doc_lookup",
                value=lookup_time,
                metadata=f"lookup={i+1}"
            ))
        
        # === PROMPT PHASE (LLM tools only) ===
        # Jackson: prompt rounds vary by task and user approach
        
        if tool_mode != "search":
            expected_prompts = (
                1.0 + 3.5 * task.difficulty +
                2.0 * student.effort_minimization -
                1.5 * student.skill +
                1.0 * task.conceptual_complexity
            )
            
            # Intentional users may use more prompts strategically
            if student.intentionality > 0.6:
                expected_prompts += 1.0
            
            expected_prompts *= (1.0 + 0.20 * student.time_pressure)
            
            prompts = min(
                self.config.max_prompt_rounds,
                max(1, int(round(expected_prompts)))
            )
            
            # Process each prompt
            for i in range(prompts):
                latency = positive_normal(
                    self.rng,
                    self.config.llm_latency_sec_mean,
                    self.config.llm_latency_sec_sd,
                    minimum=0.4
                )
                time_offset += latency
                
                # Calculate error probabilities
                # Jackson: errors depend on skill, context care, task difficulty
                
                halluc_prob = self._prob_hallucination(student, task, tool_mode)
                omit_prob = self._prob_omission(student, task, tool_mode)
                
                halluc_occurred = int(self.rng.random() < halluc_prob)
                omit_occurred = int(self.rng.random() < omit_prob)
                
                hallucinations += halluc_occurred
                omissions += omit_occurred
                
                events.append(Event(
                    timestamp=get_timestamp(base_time, time_offset),
                    run_id=run_id,
                    student_id=student.student_id,
                    task_id=task.task_id,
                    tool_mode=tool_mode,
                    event_type="prompt",
                    value=latency,
                    metadata=f"round={i+1}|halluc={halluc_occurred}|omit={omit_occurred}"
                ))
                
                # Verification behavior (Jackson: checking LLM output matters)
                # Intentional users more likely to verify
                verify_prob = clamp01(
                    0.20 + 0.50 * student.skill +
                    0.40 * student.intentionality -
                    0.35 * student.effort_minimization -
                    0.25 * student.time_pressure
                )
                
                if self.rng.random() < verify_prob:
                    verification_checks += 1
                    verify_time = positive_normal(self.rng, 1.5, 0.6, 0.2)
                    time_offset += verify_time
                    
                    events.append(Event(
                        timestamp=get_timestamp(base_time, time_offset),
                        run_id=run_id,
                        student_id=student.student_id,
                        task_id=task.task_id,
                        tool_mode=tool_mode,
                        event_type="verify_output",
                        value=verify_time,
                        metadata=f"after_prompt={i+1}"
                    ))
                    
                    # Verification can catch errors
                    if halluc_occurred and self.rng.random() < (0.60 * student.skill + 0.20):
                        hallucinations_caught += 1
                        events.append(Event(
                            timestamp=get_timestamp(base_time, time_offset),
                            run_id=run_id,
                            student_id=student.student_id,
                            task_id=task.task_id,
                            tool_mode=tool_mode,
                            event_type="correct_hallucination",
                            value=1,
                            metadata=""
                        ))
                    
                    if omit_occurred and self.rng.random() < (0.50 * student.skill + 0.15):
                        omissions_caught += 1
                        events.append(Event(
                            timestamp=get_timestamp(base_time, time_offset),
                            run_id=run_id,
                            student_id=student.student_id,
                            task_id=task.task_id,
                            tool_mode=tool_mode,
                            event_type="recover_omission",
                            value=1,
                            metadata=""
                        ))
        
        # === EDITING PHASE ===
        # Jackson: editing shows engagement and agency
        
        expected_edits = (
            2.0 + 5.5 * student.skill +
            2.8 * task.difficulty +
            1.5 * student.intentionality
        )
        
        expected_edits *= (1.20 - 0.75 * student.effort_minimization)
        
        if tool_mode == "llm_agentic":
            expected_edits *= 0.75  # Agentic tools reduce manual editing
        
        edits = min(
            self.config.max_edit_rounds,
            max(0, int(round(expected_edits)))
        )
        
        # Boundary violation probability (Jackson: agentic tools "tread on toes")
        boundary_prob = self._prob_boundary_violation(student, task, tool_mode)
        
        for i in range(edits):
            edit_time = positive_normal(
                self.rng,
                1.5 + 2.2 * task.difficulty,
                0.8,
                minimum=0.2
            )
            time_offset += edit_time
            
            boundary_violation = int(self.rng.random() < boundary_prob)
            boundary_violations += boundary_violation
            
            events.append(Event(
                timestamp=get_timestamp(base_time, time_offset),
                run_id=run_id,
                student_id=student.student_id,
                task_id=task.task_id,
                tool_mode=tool_mode,
                event_type="edit",
                value=edit_time,
                metadata=f"round={i+1}|boundary_violation={boundary_violation}"
            ))
        
        # === TESTING PHASE (coding tasks only) ===
        
        tests_total = 0
        tests_failed = 0
        
        if task.task_type == "coding":
            # Total tests available
            tests_total = int(round(positive_normal(
                self.rng,
                self.config.coding_tests_mean,
                self.config.coding_tests_sd,
                minimum=6
            )))
            tests_total = max(6, min(40, tests_total))
            
            # How many times student runs tests
            expected_test_runs = (
                1.0 + 5.0 * student.skill +
                2.5 * task.difficulty +
                1.0 * student.intentionality
            )
            
            expected_test_runs *= (1.15 - 0.65 * student.effort_minimization)
            
            if student.time_pressure > 0.7:
                expected_test_runs *= 0.75
            
            test_runs = min(
                self.config.max_test_runs,
                max(0, int(round(expected_test_runs)))
            )
            
            for i in range(test_runs):
                test_time = positive_normal(
                    self.rng,
                    1.2 + 1.5 * task.difficulty,
                    0.6,
                    minimum=0.2
                )
                time_offset += test_time
                
                events.append(Event(
                    timestamp=get_timestamp(base_time, time_offset),
                    run_id=run_id,
                    student_id=student.student_id,
                    task_id=task.task_id,
                    tool_mode=tool_mode,
                    event_type="run_tests",
                    value=test_time,
                    metadata=f"round={i+1}"
                ))
            
            # Calculate test failures
            tests_failed = self._calculate_test_failures(
                student, task, tool_mode,
                tests_total, test_runs,
                hallucinations, omissions,
                boundary_violations, prompts
            )
        
        # === OUTCOME CALCULATION ===
        
        score, passed = self._calculate_score_and_pass(
            student=student,
            task=task,
            tool_mode=tool_mode,
            doc_opens=doc_opens,
            reading_sessions=reading_sessions,
            reading_time=reading_time,
            prompts=prompts,
            edits=edits,
            test_runs=test_runs,
            verification_checks=verification_checks,
            boundary_violations=boundary_violations,
            hallucinations=hallucinations,
            hallucinations_caught=hallucinations_caught,
            omissions=omissions,
            omissions_caught=omissions_caught,
            tests_total=tests_total,
            tests_failed=tests_failed,
        )
        
        # Agency proxy (Jackson: intentionality preserves agency)
        agency_proxy = self._calculate_agency(
            student, tool_mode, edits, prompts,
            reading_sessions, verification_checks
        )
        
        # Cognitive engagement (deep vs surface learning)
        cognitive_engagement = self._calculate_cognitive_engagement(
            student, task, tool_mode,
            reading_sessions, verification_checks, edits
        )
        
        # Context quality (how well context was selected)
        context_quality = self._calculate_context_quality(
            student, tool_mode, doc_opens,
            reading_sessions, boundary_violations
        )
        
        # Submit event
        events.append(Event(
            timestamp=get_timestamp(base_time, time_offset),
            run_id=run_id,
            student_id=student.student_id,
            task_id=task.task_id,
            tool_mode=tool_mode,
            event_type="submit",
            value=1,
            metadata=f"score={score:.3f}|passed={passed}|time={time_offset:.1f}s"
        ))
        
        # Store events
        self.events.extend(events)
        
        # Return complete run result
        return RunResult(
            run_id=run_id,
            student_id=student.student_id,
            task_id=task.task_id,
            tool_mode=tool_mode,
            timestamp=get_timestamp(base_time, 0),
            prompts=prompts,
            doc_opens=doc_opens,
            reading_sessions=reading_sessions,
            edits=edits,
            test_runs=test_runs,
            verification_checks=verification_checks,
            boundary_violations=boundary_violations,
            hallucinations=hallucinations,
            hallucinations_caught=hallucinations_caught,
            omissions=omissions,
            omissions_caught=omissions_caught,
            latency_sec_total=time_offset,
            reading_time_sec=reading_time,
            score=score,
            passed=passed,
            tests_total=tests_total,
            tests_failed=tests_failed,
            agency_proxy=agency_proxy,
            cognitive_engagement=cognitive_engagement,
            context_quality=context_quality,
        )
    
    # -------------------------------------------------------------------------
    # Probability models (based on Jackson's observations)
    # -------------------------------------------------------------------------
    
    def _prob_hallucination(
        self,
        student: Student,
        task: Task,
        tool_mode: str
    ) -> float:
        """
        Calculate probability of LLM hallucination.
        
        Jackson: hallucinations depend on skill, context care, and pressure.
        """
        if tool_mode == "search":
            return clamp01(0.05 * (1.0 - student.skill) * (0.7 + task.difficulty))
        
        base = self.config.llm_hallucination_base
        
        prob = base * (1.1 - 0.6 * student.skill)
        prob *= (1.1 - 0.5 * student.context_care)
        prob *= (0.8 + 0.6 * task.difficulty)
        prob *= (1.0 + 0.5 * student.time_pressure)
        prob *= (1.0 + 0.4 * student.effort_minimization)
        prob *= (1.1 - 0.3 * student.intentionality)
        
        return clamp01(prob)
    
    def _prob_omission(
        self,
        student: Student,
        task: Task,
        tool_mode: str
    ) -> float:
        """
        Calculate probability of LLM omission error.
        
        Jackson: omissions happen when students don't verify carefully.
        """
        if tool_mode == "search":
            return clamp01(
                0.10 * (1.0 - student.doc_discipline) * (0.8 + task.difficulty)
            )
        
        base = self.config.llm_omission_base
        
        prob = base * (1.1 - 0.5 * student.skill)
        prob *= (1.2 - 0.6 * student.doc_discipline)
        prob *= (0.8 + 0.7 * task.difficulty)
        prob *= (1.0 + 0.5 * student.time_pressure)
        prob *= (1.0 + 0.3 * student.effort_minimization)
        
        return clamp01(prob)
    
    def _prob_boundary_violation(
        self,
        student: Student,
        task: Task,
        tool_mode: str
    ) -> float:
        """
        Calculate probability of module boundary violation.
        
        Jackson: agentic tools "tread on each other's toes" without modularity.
        Only relevant for coding tasks with module boundaries.
        """
        if task.task_type != "coding" or not task.has_module_boundaries:
            return 0.0
        
        if tool_mode == "llm_agentic":
            base = self.config.agentic_boundary_violation_base
        elif tool_mode == "llm_explicit":
            base = self.config.explicit_boundary_violation_base
        else:
            base = self.config.search_boundary_violation_base
        
        prob = base * (1.15 - 0.6 * student.context_care)
        prob *= (1.10 - 0.5 * student.skill)
        prob *= (0.8 + 0.5 * task.difficulty)
        prob *= (1.0 + 0.4 * student.time_pressure)
        prob *= (1.2 - 0.3 * student.intentionality)
        
        return clamp01(prob)
    
    def _calculate_test_failures(
        self,
        student: Student,
        task: Task,
        tool_mode: str,
        tests_total: int,
        test_runs: int,
        hallucinations: int,
        omissions: int,
        boundary_violations: int,
        prompts: int
    ) -> int:
        """Calculate how many tests fail based on work quality."""
        if tests_total == 0:
            return 0
        
        # Failure risk factors
        halluc_rate = hallucinations / max(1, prompts) if prompts > 0 else 0
        omit_rate = omissions / max(1, prompts) if prompts > 0 else 0
        boundary_rate = boundary_violations / max(1, 10)  # normalized
        
        risk = 0.25 + 0.35 * task.difficulty
        risk += 0.20 * halluc_rate
        risk += 0.18 * omit_rate
        risk += 0.15 * boundary_rate
        risk *= (1.15 - 0.55 * student.skill)
        
        # Testing reduces failures
        test_coverage = test_runs / max(1, self.config.max_test_runs)
        risk *= (1.10 - 0.45 * test_coverage)
        
        risk = clamp01(risk)
        
        # Expected failures with noise
        expected_failures = risk * tests_total
        noisy_failures = self.rng.gauss(expected_failures, 0.12 * tests_total)
        
        failures = int(round(max(0.0, noisy_failures)))
        return max(0, min(tests_total, failures))
    
    def _calculate_score_and_pass(
        self,
        student: Student,
        task: Task,
        tool_mode: str,
        doc_opens: int,
        reading_sessions: int,
        reading_time: float,
        prompts: int,
        edits: int,
        test_runs: int,
        verification_checks: int,
        boundary_violations: int,
        hallucinations: int,
        hallucinations_caught: int,
        omissions: int,
        omissions_caught: int,
        tests_total: int,
        tests_failed: int,
    ) -> Tuple[float, int]:
        """
        Calculate rubric score and pass/fail.
        
        Based on Jackson's observations about what leads to quality work.
        """
        # Base score from skill and task
        base = 0.55 + 0.55 * student.skill
        base -= 0.35 * task.difficulty
        base -= 0.18 * task.rubric_strictness
        base = clamp01(base)
        
        score = base
        
        # Documentation reading (Jackson: critical for novices)
        reading_term = reading_sessions / max(1, self.config.max_reading_sessions)
        score += self.config.weight_reading_on_quality * (reading_term - 0.30)
        
        # Quick doc lookups
        docs_term = doc_opens / max(1, self.config.max_doc_opens)
        score += self.config.weight_docs_on_quality * (docs_term - 0.35)
        
        # Testing behavior
        if tests_total > 0:
            testing_term = test_runs / max(1, self.config.max_test_runs)
            score += self.config.weight_testing_on_quality * (testing_term - 0.30)
        
        # Error penalties (uncaught errors hurt more)
        uncaught_halluc = hallucinations - hallucinations_caught
        uncaught_omit = omissions - omissions_caught
        
        if prompts > 0:
            halluc_term = uncaught_halluc / prompts
            omit_term = uncaught_omit / prompts
            score -= self.config.weight_hallucinations_on_quality * (
                0.70 * halluc_term + 0.60 * omit_term
            )
        
        # Boundary violations
        if edits > 0 and task.has_module_boundaries:
            boundary_term = boundary_violations / edits
            score -= self.config.weight_boundary_violations_on_quality * boundary_term
        
        # Behavioral penalties
        score -= self.config.weight_effort_minimization_on_quality * (
            student.effort_minimization - 0.40
        )
        score -= self.config.weight_time_pressure_on_quality * (
            student.time_pressure - 0.45
        )
        
        # Intentionality bonus (Jackson: intentional use preserves learning)
        score += self.config.weight_intentionality_on_quality * (
            student.intentionality - 0.45
        )
        
        # Engagement through editing
        edit_term = edits / max(1, self.config.max_edit_rounds)
        score += 0.12 * (math.sqrt(edit_term) - 0.45)
        
        # Verification bonus
        verify_term = verification_checks / max(1, prompts if prompts > 0 else 1)
        score += 0.10 * verify_term
        
        # Tool-specific adjustments
        if tool_mode == "search":
            score += 0.03  # Documentation-driven slightly better
        elif tool_mode == "llm_agentic":
            score -= 0.05  # Agentic tools riskier
        
        score = clamp01(score)
        
        # Pass determination
        pass_by_score = score >= self.config.pass_score_threshold
        pass_by_tests = True
        
        if tests_total > 0:
            # Allow up to 12% test failures
            pass_by_tests = tests_failed <= max(1, int(0.12 * tests_total))
        
        passed = 1 if (pass_by_score and pass_by_tests) else 0
        
        return score, passed
    
    def _calculate_agency(
        self,
        student: Student,
        tool_mode: str,
        edits: int,
        prompts: int,
        reading_sessions: int,
        verification_checks: int
    ) -> float:
        """
        Calculate student agency proxy.
        
        Jackson: agency preserved through intentionality and active engagement.
        """
        edit_term = edits / max(1, self.config.max_edit_rounds)
        prompt_term = prompts / max(1, self.config.max_prompt_rounds)
        reading_term = reading_sessions / max(1, self.config.max_reading_sessions)
        verify_term = verification_checks / max(1, prompts if prompts > 0 else 1)
        
        agency = 0.40
        agency += 0.30 * clamp01(edit_term)
        agency += 0.20 * clamp01(reading_term)
        agency += 0.15 * clamp01(verify_term)
        agency -= 0.18 * clamp01(prompt_term)
        agency -= 0.22 * student.effort_minimization
        agency += 0.25 * student.intentionality
        
        if tool_mode == "llm_agentic":
            agency -= 0.08  # Agentic tools reduce agency
        
        return clamp01(agency)
    
    def _calculate_cognitive_engagement(
        self,
        student: Student,
        task: Task,
        tool_mode: str,
        reading_sessions: int,
        verification_checks: int,
        edits: int
    ) -> float:
        """
        Calculate cognitive engagement (deep vs surface learning).
        
        Jackson: deep engagement requires reading, verifying, editing.
        """
        reading_term = reading_sessions / max(1, self.config.max_reading_sessions)
        verify_term = verification_checks / max(1, 3)  # Normalize to ~3 checks
        edit_term = edits / max(1, self.config.max_edit_rounds)
        
        engagement = 0.35
        engagement += 0.30 * reading_term
        engagement += 0.25 * verify_term
        engagement += 0.20 * edit_term
        engagement += 0.25 * student.intentionality
        engagement -= 0.30 * student.effort_minimization
        engagement += 0.15 * student.skill  # Skilled students engage more deeply
        
        # Task complexity encourages deeper engagement (if not rushed)
        if student.time_pressure < 0.6:
            engagement += 0.10 * task.conceptual_complexity
        
        if tool_mode == "llm_agentic":
            engagement -= 0.10  # Agentic use reduces cognitive engagement
        
        return clamp01(engagement)
    
    def _calculate_context_quality(
        self,
        student: Student,
        tool_mode: str,
        doc_opens: int,
        reading_sessions: int,
        boundary_violations: int
    ) -> float:
        """
        Calculate context selection quality.
        
        Jackson: "context engineering replacing prompt engineering"
        Good context = reading docs, avoiding boundary violations
        """
        reading_term = reading_sessions / max(1, self.config.max_reading_sessions)
        doc_term = doc_opens / max(1, self.config.max_doc_opens)
        
        quality = 0.40
        quality += 0.35 * student.context_care
        quality += 0.25 * reading_term
        quality += 0.20 * doc_term
        quality += 0.15 * student.skill
        
        # Boundary violations indicate poor context selection
        if boundary_violations > 0:
            violation_penalty = min(0.30, 0.10 * boundary_violations)
            quality -= violation_penalty
        
        if tool_mode == "llm_explicit":
            quality += 0.08  # Explicit context selection is better
        elif tool_mode == "llm_agentic":
            quality -= 0.12  # Agentic context selection is worse
        
        return clamp01(quality)
    
    def run(self) -> None:
        """Execute complete simulation."""
        logger.info("Starting simulation...")
        
        # Generate population and tasks
        self.students = self.generate_students()
        self.tasks = self.generate_tasks()
        
        # Simulate runs
        logger.info(f"Simulating {self.config.n_students} students × "
                   f"{self.config.tasks_per_student} tasks...")
        
        run_counter = 0
        base_time = datetime(2026, 1, 15, 9, 0, 0)  # Arbitrary start time
        
        for student in self.students:
            # Each student gets random sample of tasks
            student_tasks = self.rng.sample(
                self.tasks,
                k=min(self.config.tasks_per_student, len(self.tasks))
            )
            
            for task in student_tasks:
                # Assign tool mode
                tool_mode = weighted_choice(self.rng, self.config.tool_distribution)
                
                run_id = f"R{run_counter:06d}"
                
                # Simulate this run
                result = self.simulate_run(
                    student=student,
                    task=task,
                    tool_mode=tool_mode,
                    run_id=run_id,
                    base_time=base_time + timedelta(hours=run_counter * 0.5)
                )
                
                self.runs.append(result)
                run_counter += 1
        
        logger.info(f"Completed {len(self.runs)} runs with {len(self.events)} events")


# =============================================================================
# OUTPUT & ANALYSIS
# =============================================================================

class OutputManager:
    """Handles writing simulation outputs and generating visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
        ensure_dir(config.output_dir)
    
    def write_all(
        self,
        students: List[Student],
        tasks: List[Task],
        runs: List[RunResult],
        events: List[Event]
    ) -> None:
        """Write all simulation outputs."""
        logger.info(f"Writing outputs to {self.config.output_dir}")
        
        # CSV files
        self._write_csv("students.csv", students)
        self._write_csv("tasks.csv", tasks)
        self._write_csv("runs.csv", runs)
        self._write_csv("events.csv", events)
        
        # Config
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Wrote config to {config_path}")
        
        # Summary statistics
        summary = self._generate_summary(runs)
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote summary to {summary_path}")
        
        # Metadata for dataset
        if self.config.write_metadata:
            self._write_metadata(students, tasks, runs, events)
        
        # README for dataset
        if self.config.write_readme:
            self._write_readme(summary)
        
        # Figures
        if self.config.make_figures and HAS_PLOTTING:
            self._make_all_figures(runs, events)
    
    def _write_csv(self, filename: str, data: List) -> None:
        """Write list of dataclass objects to CSV."""
        if not data:
            logger.warning(f"No data to write for {filename}")
            return
        
        filepath = os.path.join(self.config.output_dir, filename)
        fieldnames = list(data[0].to_dict().keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                writer.writerow(item.to_dict())
        
        logger.info(f"Wrote {len(data)} rows to {filename}")
    
    def _generate_summary(self, runs: List[RunResult]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not runs:
            return {"n_runs": 0, "error": "No runs completed"}
        
        df = pd.DataFrame([r.to_dict() for r in runs])
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "n_runs": len(runs),
            "n_students": df['student_id'].nunique(),
            "n_tasks": df['task_id'].nunique(),
            
            "overall": {
                "pass_rate": float(df['passed'].mean()),
                "avg_score": float(df['score'].mean()),
                "avg_agency": float(df['agency_proxy'].mean()),
                "avg_cognitive_engagement": float(df['cognitive_engagement'].mean()),
                "avg_context_quality": float(df['context_quality'].mean()),
            },
            
            "by_tool": {}
        }
        
        # By tool mode
        for tool in df['tool_mode'].unique():
            tool_df = df[df['tool_mode'] == tool]
            summary["by_tool"][tool] = {
                "n": len(tool_df),
                "pass_rate": float(tool_df['passed'].mean()),
                "avg_score": float(tool_df['score'].mean()),
                "avg_doc_opens": float(tool_df['doc_opens'].mean()),
                "avg_reading_sessions": float(tool_df['reading_sessions'].mean()),
                "avg_prompts": float(tool_df['prompts'].mean()),
                "avg_edits": float(tool_df['edits'].mean()),
                "avg_verification_checks": float(tool_df['verification_checks'].mean()),
                "avg_boundary_violations": float(tool_df['boundary_violations'].mean()),
                "avg_hallucinations": float(tool_df['hallucinations'].mean()),
                "avg_agency": float(tool_df['agency_proxy'].mean()),
                "avg_cognitive_engagement": float(tool_df['cognitive_engagement'].mean()),
                "avg_context_quality": float(tool_df['context_quality'].mean()),
            }
        
        return summary
    
    def _write_metadata(
        self,
        students: List[Student],
        tasks: List[Task],
        runs: List[RunResult],
        events: List[Event]
    ) -> None:
        """Write dataset metadata for Hugging Face."""
        metadata = {
            "dataset_name": "llm-education-impact-simulator",
            "version": self.config.version,
            "description": self.config.description,
            "based_on": "Jackson, D. (2025). LLMs are not calculators: Why educators should embrace AI (and fear it)",
            "generated_at": datetime.now().isoformat(),
            "license": "MIT",
            "language": "en",
            
            "statistics": {
                "n_students": len(students),
                "n_tasks": len(tasks),
                "n_runs": len(runs),
                "n_events": len(events),
            },
            
            "files": {
                "students.csv": "Student characteristics and behavioral traits",
                "tasks.csv": "Educational tasks with varying difficulty and structure",
                "runs.csv": "Complete records of students completing tasks with different tools",
                "events.csv": "Fine-grained event log of student activities",
                "config.json": "Simulation configuration parameters",
                "summary.json": "Summary statistics and aggregations",
            },
            
            "schema": {
                "students": {
                    "student_id": "string (unique identifier)",
                    "skill": "float [0-1] (base ability)",
                    "effort_minimization": "float [0-1] (tendency to take shortcuts)",
                    "doc_discipline": "float [0-1] (reads documentation)",
                    "context_care": "float [0-1] (attention to context selection)",
                    "time_pressure": "float [0-1] (stress/urgency)",
                    "intentionality": "float [0-1] (deliberate tool use)",
                },
                "tasks": {
                    "task_id": "string (unique identifier)",
                    "task_type": "string (coding|writing|reading)",
                    "difficulty": "float [0-1]",
                    "has_module_boundaries": "bool (modularity requirement)",
                    "rubric_strictness": "float [0-1]",
                    "requires_prior_reading": "bool",
                    "conceptual_complexity": "float [0-1]",
                },
                "runs": {
                    "run_id": "string (unique identifier)",
                    "student_id": "string (foreign key)",
                    "task_id": "string (foreign key)",
                    "tool_mode": "string (search|llm_explicit|llm_agentic)",
                    "score": "float [0-1] (rubric score)",
                    "passed": "int (0|1)",
                    "agency_proxy": "float [0-1] (student ownership)",
                    "cognitive_engagement": "float [0-1] (deep learning)",
                    "context_quality": "float [0-1] (context selection quality)",
                },
            },
            
            "citation": {
                "text": "If you use this dataset, please cite: Jackson, D. (2025). LLMs are not calculators: Why educators should embrace AI (and fear it).",
                "bibtex": """@article{jackson2025llms,
  title={LLMs are not calculators: Why educators should embrace AI (and fear it)},
  author={Jackson, Daniel},
  year={2025},
  month={December}
}"""
            }
        }
        
        metadata_path = os.path.join(self.config.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata to {metadata_path}")
    
    def _write_readme(self, summary: Dict[str, Any]) -> None:
        """Generate README for the dataset."""
        readme = f"""# LLM Education Impact Simulator Dataset

**Version:** {self.config.version}  
**Generated:** {datetime.now().strftime('%Y-%m-%d')}  
**Based on:** Jackson, D. (2025). "LLMs are not calculators: Why educators should embrace AI (and fear it)"

## Overview

This dataset contains synthetic observational data simulating how students interact with different AI tools (search engines, explicit-context LLMs, and agentic LLMs) while completing educational tasks.

The simulation is grounded in educational research, particularly Daniel Jackson's observations about LLM usage in undergraduate software development courses.

## Key Findings from Source Research

Jackson's paper identifies several critical patterns:

1. **Context selection matters more than prompting** - How students select context for LLM queries is more important than how they phrase prompts
2. **Reading documentation declines** - Students using LLMs read less background documentation
3. **Intentionality preserves agency** - Deliberate, strategic LLM use maintains student ownership of work
4. **Agentic tools risk boundary violations** - Automated context selection often breaks module boundaries
5. **Verification is critical** - Checking LLM output significantly reduces errors

## Dataset Statistics

- **Students:** {summary.get('n_students', 'N/A')}
- **Tasks:** {summary.get('n_tasks', 'N/A')}
- **Runs:** {summary.get('n_runs', 'N/A')}
- **Overall pass rate:** {summary.get('overall', {}).get('pass_rate', 0):.1%}
- **Average agency proxy:** {summary.get('overall', {}).get('avg_agency', 0):.3f}

## Files

- `students.csv` - Student characteristics (n={summary.get('n_students', 'N/A')})
- `tasks.csv` - Educational tasks (n={summary.get('n_tasks', 'N/A')})
- `runs.csv` - Complete run records (n={summary.get('n_runs', 'N/A')})
- `events.csv` - Fine-grained event log
- `config.json` - Simulation parameters
- `summary.json` - Summary statistics
- `metadata.json` - Dataset metadata

## Tool Modes

### search
Traditional documentation-driven approach. Students consult search engines and documentation rather than LLMs.

### llm_explicit
LLM usage with explicit context selection. Students carefully choose what context to provide to the LLM, maintaining awareness of scope and boundaries.

### llm_agentic
LLM usage with automated context selection. Tools automatically determine context, reducing student control but potentially increasing efficiency.

## Key Variables

### Student Traits
- `skill` - Base ability (0-1)
- `effort_minimization` - Tendency to take shortcuts (0-1)
- `doc_discipline` - Propensity to read documentation (0-1)
- `context_care` - Attention to context selection (0-1)
- `intentionality` - Deliberate vs passive tool use (0-1)

### Outcomes
- `score` - Rubric-based quality score (0-1)
- `passed` - Binary pass/fail (0/1)
- `agency_proxy` - Student ownership of work (0-1)
- `cognitive_engagement` - Deep vs surface learning (0-1)
- `context_quality` - Quality of context selection (0-1)

### Observable Behaviors
- `reading_sessions` - Sustained documentation reading
- `doc_opens` - Quick reference lookups
- `prompts` - LLM prompt rounds
- `verification_checks` - Checking LLM output
- `edits` - Manual editing rounds
- `boundary_violations` - Module boundary breaks (coding only)
- `hallucinations` - LLM factual errors
- `omissions` - LLM omission errors

## Usage Examples

### Load the data (Python)

```python
import pandas as pd

students = pd.read_csv('students.csv')
tasks = pd.read_csv('tasks.csv')
runs = pd.read_csv('runs.csv')
events = pd.read_csv('events.csv')

# Analyze pass rates by tool mode
pass_rates = runs.groupby('tool_mode')['passed'].mean()
print(pass_rates)

# Compare agency across tools
agency_by_tool = runs.groupby('tool_mode')['agency_proxy'].mean()
print(agency_by_tool)
```

### Research Questions

This dataset can help explore:

1. How does tool choice (search vs LLM) affect learning outcomes?
2. What student traits predict successful LLM usage?
3. How does context selection quality relate to errors and outcomes?
4. Does intentionality mediate the relationship between tool use and agency?
5. What behaviors distinguish students who maintain agency from those who don't?

## Citation

If you use this dataset, please cite:

```bibtex
@article{{jackson2025llms,
  title={{LLMs are not calculators: Why educators should embrace AI (and fear it)}},
  author={{Jackson, Daniel}},
  year={{2025}},
  month={{December}}
}}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions about this dataset or the underlying simulation:
- Based on research by Daniel Jackson, MIT
- Simulation code: See `llm_education_simulator.py`

## Acknowledgments

This simulation incorporates insights from:
- Daniel Jackson's course experiments at MIT
- Lorena Barba's experiences with LLMs in programming courses
- Andrés Salazar-Gómez and Sanjay Sarma's AI in education analysis
"""
        
        readme_path = os.path.join(self.config.output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme)
        logger.info(f"Wrote README to {readme_path}")
    
    def _make_all_figures(self, runs: List[RunResult], events: List[Event]) -> None:
        """Generate all visualization figures."""
        logger.info("Generating figures...")
        
        df_runs = pd.DataFrame([r.to_dict() for r in runs])
        
        # Figure 1: Pass rates by tool mode
        self._fig_pass_rates_by_tool(df_runs)
        
        # Figure 2: Agency by tool mode
        self._fig_agency_by_tool(df_runs)
        
        # Figure 3: Reading behavior by tool mode
        self._fig_reading_by_tool(df_runs)
        
        # Figure 4: Error rates
        self._fig_error_rates(df_runs)
        
        # Figure 5: Cognitive engagement vs agency
        self._fig_engagement_vs_agency(df_runs)
        
        # Figure 6: Context quality impact
        self._fig_context_quality_impact(df_runs)
        
        # Figure 7: Boundary violations (coding tasks)
        self._fig_boundary_violations(df_runs)
        
        # Figure 8: Verification effectiveness
        self._fig_verification_effectiveness(df_runs)
        
        logger.info("Completed all figures")
    
    def _save_figure(self, name: str) -> None:
        """Save current figure to file."""
        filepath = os.path.join(
            self.config.output_dir,
            f"{name}.{self.config.figure_format}"
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved figure: {name}")
    
    def _fig_pass_rates_by_tool(self, df: pd.DataFrame) -> None:
        """Figure 1: Pass rates by tool mode."""
        pass_rates = df.groupby('tool_mode')['passed'].agg(['mean', 'sem'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(pass_rates))
        ax.bar(x, pass_rates['mean'], yerr=pass_rates['sem'],
               capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_xticks(x)
        ax.set_xticklabels(pass_rates.index)
        ax.set_ylabel('Pass Rate')
        ax.set_title('Pass Rates by Tool Mode')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        self._save_figure('fig1_pass_rates_by_tool')
    
    def _fig_agency_by_tool(self, df: pd.DataFrame) -> None:
        """Figure 2: Agency proxy by tool mode."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        tool_order = ['search', 'llm_explicit', 'llm_agentic']
        data = [df[df['tool_mode'] == tool]['agency_proxy'].values
                for tool in tool_order]
        
        bp = ax.boxplot(data, labels=tool_order, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Agency Proxy')
        ax.set_title('Student Agency by Tool Mode')
        ax.set_ylim([0, 1])
        
        self._save_figure('fig2_agency_by_tool')
    
    def _fig_reading_by_tool(self, df: pd.DataFrame) -> None:
        """Figure 3: Reading behavior by tool mode."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Reading sessions
        reading_data = df.groupby('tool_mode')['reading_sessions'].mean()
        ax1.bar(range(len(reading_data)), reading_data.values,
                color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        ax1.set_xticks(range(len(reading_data)))
        ax1.set_xticklabels(reading_data.index)
        ax1.set_ylabel('Average Reading Sessions')
        ax1.set_title('Documentation Reading by Tool Mode')
        
        # Doc lookups
        doc_data = df.groupby('tool_mode')['doc_opens'].mean()
        ax2.bar(range(len(doc_data)), doc_data.values,
                color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        ax2.set_xticks(range(len(doc_data)))
        ax2.set_xticklabels(doc_data.index)
        ax2.set_ylabel('Average Doc Lookups')
        ax2.set_title('Quick Reference Lookups by Tool Mode')
        
        self._save_figure('fig3_reading_by_tool')
    
    def _fig_error_rates(self, df: pd.DataFrame) -> None:
        """Figure 4: Error rates by tool mode."""
        # Filter to LLM modes only
        llm_df = df[df['tool_mode'].str.startswith('llm')].copy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tools = llm_df['tool_mode'].unique()
        x = np.arange(len(tools))
        width = 0.25
        
        halluc_rates = [llm_df[llm_df['tool_mode'] == t]['hallucinations'].mean()
                        for t in tools]
        omit_rates = [llm_df[llm_df['tool_mode'] == t]['omissions'].mean()
                      for t in tools]
        boundary_rates = [llm_df[llm_df['tool_mode'] == t]['boundary_violations'].mean()
                         for t in tools]
        
        ax.bar(x - width, halluc_rates, width, label='Hallucinations', alpha=0.8)
        ax.bar(x, omit_rates, width, label='Omissions', alpha=0.8)
        ax.bar(x + width, boundary_rates, width, label='Boundary Violations', alpha=0.8)
        
        ax.set_ylabel('Average Count')
        ax.set_title('Error Rates by Tool Mode (LLM only)')
        ax.set_xticks(x)
        ax.set_xticklabels(tools)
        ax.legend()
        
        self._save_figure('fig4_error_rates')
    
    def _fig_engagement_vs_agency(self, df: pd.DataFrame) -> None:
        """Figure 5: Cognitive engagement vs agency."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for tool, color in [('search', '#2ecc71'),
                           ('llm_explicit', '#3498db'),
                           ('llm_agentic', '#e74c3c')]:
            tool_df = df[df['tool_mode'] == tool]
            ax.scatter(tool_df['cognitive_engagement'],
                      tool_df['agency_proxy'],
                      alpha=0.5, s=50, color=color, label=tool)
        
        ax.set_xlabel('Cognitive Engagement')
        ax.set_ylabel('Agency Proxy')
        ax.set_title('Cognitive Engagement vs Student Agency')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        self._save_figure('fig5_engagement_vs_agency')
    
    def _fig_context_quality_impact(self, df: pd.DataFrame) -> None:
        """Figure 6: Context quality impact on outcomes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Context quality vs score
        for tool, color in [('search', '#2ecc71'),
                           ('llm_explicit', '#3498db'),
                           ('llm_agentic', '#e74c3c')]:
            tool_df = df[df['tool_mode'] == tool]
            ax1.scatter(tool_df['context_quality'], tool_df['score'],
                       alpha=0.4, s=40, color=color, label=tool)
        
        ax1.set_xlabel('Context Quality')
        ax1.set_ylabel('Score')
        ax1.set_title('Context Quality vs Score')
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Context quality bins vs pass rate
        df['context_bin'] = pd.cut(df['context_quality'], bins=5)
        bin_pass_rates = df.groupby('context_bin')['passed'].mean()
        
        ax2.bar(range(len(bin_pass_rates)), bin_pass_rates.values,
                color='#3498db', alpha=0.7)
        ax2.set_xticks(range(len(bin_pass_rates)))
        ax2.set_xticklabels([f"{i.left:.2f}-{i.right:.2f}"
                            for i in bin_pass_rates.index], rotation=45)
        ax2.set_xlabel('Context Quality Bin')
        ax2.set_ylabel('Pass Rate')
        ax2.set_title('Pass Rate by Context Quality')
        ax2.set_ylim([0, 1])
        
        self._save_figure('fig6_context_quality_impact')
    
    def _fig_boundary_violations(self, df: pd.DataFrame) -> None:
        """Figure 7: Boundary violations (coding tasks only)."""
        coding_df = df[df['tests_total'] > 0].copy()
        
        if len(coding_df) == 0:
            logger.warning("No coding tasks found, skipping boundary violations figure")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        violation_data = coding_df.groupby('tool_mode')['boundary_violations'].agg(['mean', 'sem'])
        
        x = range(len(violation_data))
        ax.bar(x, violation_data['mean'], yerr=violation_data['sem'],
               capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_xticks(x)
        ax.set_xticklabels(violation_data.index)
        ax.set_ylabel('Average Boundary Violations')
        ax.set_title('Module Boundary Violations by Tool (Coding Tasks)')
        
        self._save_figure('fig7_boundary_violations')
    
    def _fig_verification_effectiveness(self, df: pd.DataFrame) -> None:
        """Figure 8: Verification check effectiveness."""
        llm_df = df[df['tool_mode'].str.startswith('llm')].copy()
        
        if len(llm_df) == 0:
            logger.warning("No LLM runs found, skipping verification figure")
            return
        
        # Calculate caught rates
        llm_df['halluc_caught_rate'] = np.where(
            llm_df['hallucinations'] > 0,
            llm_df['hallucinations_caught'] / llm_df['hallucinations'],
            0
        )
        llm_df['omit_caught_rate'] = np.where(
            llm_df['omissions'] > 0,
            llm_df['omissions_caught'] / llm_df['omissions'],
            0
        )
        
        # Bin by verification checks
        llm_df['verify_bin'] = pd.cut(llm_df['verification_checks'], bins=4)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hallucinations caught
        halluc_caught = llm_df.groupby('verify_bin')['halluc_caught_rate'].mean()
        ax1.bar(range(len(halluc_caught)), halluc_caught.values,
                color='#e74c3c', alpha=0.7)
        ax1.set_xticks(range(len(halluc_caught)))
        ax1.set_xticklabels([f"{int(i.left)}-{int(i.right)}"
                            for i in halluc_caught.index], rotation=45)
        ax1.set_xlabel('Verification Checks')
        ax1.set_ylabel('Hallucination Catch Rate')
        ax1.set_title('Verification Effectiveness: Hallucinations')
        ax1.set_ylim([0, 1])
        
        # Omissions caught
        omit_caught = llm_df.groupby('verify_bin')['omit_caught_rate'].mean()
        ax2.bar(range(len(omit_caught)), omit_caught.values,
                color='#f39c12', alpha=0.7)
        ax2.set_xticks(range(len(omit_caught)))
        ax2.set_xticklabels([f"{int(i.left)}-{int(i.right)}"
                            for i in omit_caught.index], rotation=45)
        ax2.set_xlabel('Verification Checks')
        ax2.set_ylabel('Omission Catch Rate')
        ax2.set_title('Verification Effectiveness: Omissions')
        ax2.set_ylim([0, 1])
        
        self._save_figure('fig8_verification_effectiveness')


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="LLM Education Impact Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_education_simulator.py
  python llm_education_simulator.py --config custom_config.json
  python llm_education_simulator.py --seed 123 --students 200 --tasks 20
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--students',
        type=int,
        help='Number of students to simulate'
    )
    parser.add_argument(
        '--tasks',
        type=int,
        help='Number of tasks to generate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = Config.from_json(args.config)
    else:
        config = Config()
    
    # Override with command-line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.students is not None:
        config.n_students = args.students
    if args.tasks is not None:
        config.n_tasks = args.tasks
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_figures:
        config.make_figures = False
    
    # Run simulation
    logger.info("=" * 70)
    logger.info("LLM Education Impact Simulator v2.0")
    logger.info("Based on Jackson (2025) 'LLMs are not calculators'")
    logger.info("=" * 70)
    
    simulator = EducationSimulator(config)
    simulator.run()
    
    # Write outputs
    output_manager = OutputManager(config)
    output_manager.write_all(
        students=simulator.students,
        tasks=simulator.tasks,
        runs=simulator.runs,
        events=simulator.events
    )
    
    logger.info("=" * 70)
    logger.info(f"Simulation complete!")
    logger.info(f"Outputs saved to: {os.path.abspath(config.output_dir)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
