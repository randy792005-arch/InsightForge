"""
InsightForge: An Agentic AI-Based Retail Business Intelligence and Decision Support System
Main Application Entrypoint
"""

import sys
import os

# Ensure InsightForge root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dashboard.dashboard import main_dashboard_app

# Set wide layout and branding without emoji encoding dependencies
st.set_page_config(
    page_title="InsightForge | Enterprise Retail Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    main_dashboard_app()
