STAGE_METRICS = {
    "Planning and Design": [
        "Modularity",
        "Cohesion",
        "Requirement Traceability",
        "Architectural Complexity"
    ],
    "Development": [
        "Code Smells",
        "Maintainability Index",
        "Cognitive Complexity",
        "Cyclomatic Complexity",
        "Code Duplication",
        "Technical Debt",
        "Dependency Management",
        "Comment Density",
        "Software Size (LoC)",
        "Percentage of Assertions"
    ],
    "Testing": [
        "Unit Tests",
        "Test Success Rate",
        "Code Reproducibility",
        "Defect Rate"
    ],
    "Maintenance": [
        "Presence of License",
        "Publicly Accessible Repository",
        "Rich Metadata",
        "Documentation Quality",
        "User Satisfaction",
        "No Leaked Private Credentials",
        "Security Vulnerabilities"
    ]
}

# function that returns the metrics for a selected stage
def get_metrics_for_stage(stage_name):
    return STAGE_METRICS.get(stage_name, [])