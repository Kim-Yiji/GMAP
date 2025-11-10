"""
Entry point for threat_score_viz package when run as a module.

Usage:
    python -m threat_score_viz --help
    python -m threat_score_viz --annotations <path> --list-candidates
    python -m threat_score_viz --video <path> --annotations <path> --output-video <path>
"""

from .main import main

if __name__ == '__main__':
    main()

