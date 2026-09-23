"""
InsightForge Base Agent Architecture
Defines common execution lifecycle, structured telemetry, standardized AgentResult schemas,
and automatic logging for all specialized retail AI agents.
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from database.db_manager import DatabaseManager


@dataclass
class AgentResult:
    """Standardized output container across all multi-agent stages."""
    agent_name: str
    status: str  # 'SUCCESS', 'WARNING', 'ERROR'
    message: str
    execution_time_sec: float
    data: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status in ["SUCCESS", "WARNING"]


class BaseAgent:
    """
    Abstract base class for all InsightForge specialized agents.
    Provides timing, structured logging, database audit integration, and safe error capture.
    """

    def __init__(self, name: str, db_manager: Optional[DatabaseManager] = None):
        self.name = name
        self.db = db_manager if db_manager is not None else DatabaseManager()
        self.logs: List[str] = []

    def log(self, message: str):
        """Record an timestamped execution log message."""
        entry = f"[{self.name}] {message}"
        self.logs.append(entry)

    def run(self, *args, **kwargs) -> AgentResult:
        """
        Executes the agent lifecycle with timing, exception handling, and database logging.
        """
        self.logs = []
        start_time = time.time()
        self.log("Agent execution initiated.")

        session_id = kwargs.get("session_id", "default_session")

        try:
            # Execute agent-specific core logic
            data, status, summary_msg = self._execute(*args, **kwargs)
            exec_time = round(time.time() - start_time, 3)
            self.log(f"Agent finished in {exec_time}s with status: {status}.")

            # Record in SQLite audit log
            self.db.log_agent_step(
                session_id=session_id,
                agent_name=self.name,
                status=status,
                message=summary_msg,
                execution_time_sec=exec_time,
                details={"logs_count": len(self.logs)}
            )

            return AgentResult(
                agent_name=self.name,
                status=status,
                message=summary_msg,
                execution_time_sec=exec_time,
                data=data,
                logs=self.logs
            )

        except Exception as e:
            exec_time = round(time.time() - start_time, 3)
            error_msg = f"Agent failed with error: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())

            self.db.log_agent_step(
                session_id=session_id,
                agent_name=self.name,
                status="ERROR",
                message=str(e),
                execution_time_sec=exec_time,
                details={"error_trace": traceback.format_exc()}
            )

            return AgentResult(
                agent_name=self.name,
                status="ERROR",
                message=error_msg,
                execution_time_sec=exec_time,
                data={},
                logs=self.logs
            )

    def _execute(self, *args, **kwargs) -> tuple:
        """Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement _execute")
