from multi_agent_system.tools.knowledge_base import mock_knowledge_base
from multi_agent_system.memory.working_memory import WorkingMemory
from multi_agent_system.logging.logger import log_agent_action

class ResearcherAgent:
    def research(self, query: str, memory: WorkingMemory, simulate_failure: bool = True, force_failure: bool = False) -> str:
        """
        Queries the mock knowledge base to research a query and stores findings in WorkingMemory.
        Handles failures gracefully and logs events.
        """
        log_agent_action("Researcher", "Search Knowledge Base", f"Searching for query: '{query}'")

        try:
            findings = mock_knowledge_base(query, simulate_failure=simulate_failure, force_failure=force_failure)
            log_agent_action("Researcher", "Search Completed", f"Successfully retrieved findings for query: '{query}'")
            memory.add_research_finding(findings)
            return findings
        except Exception as e:
            # Safely handle exception and log it
            error_msg = f"Tool Execution Failed: {str(e)}"
            log_agent_action("Researcher", "Handle Tool Failure", f"Gracefully caught exception: '{str(e)}'")
            
            # Save fallback state to memory to prevent system crash
            fallback_finding = f"[Researcher Warning: Tool failure occurred while researching '{query}'. Fallback state active.]"
            memory.add_research_finding(fallback_finding)
            return fallback_finding
