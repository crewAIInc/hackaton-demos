import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, before_kickoff, crew, task

from src.contract_analysis.services import ContractsService
from src.contract_analysis.tools.qdrant_vector_search_tool import QdrantVectorSearchTool


@CrewBase
class AnalyzingContractClausesForConflictsAndSimilaritiesCrew:
    """AnalyzingContractClausesForConflictsAndSimilarities crew"""

    vector_search_tool = QdrantVectorSearchTool(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )

    @before_kickoff
    def load_and_classify_contracts(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ContractsService().load_and_classify_contracts()
        return inputs

    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )

    @agent
    def source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["source_citer_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )

    @agent
    def conflicts_of_interest_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["conflicts_of_interest_specialist"],  # type: ignore
        )

    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )

    @task
    def retrieve_contracts_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_contracts_task"],  # type: ignore
        )

    @task
    def source_citer_task(self) -> Task:
        return Task(
            config=self.tasks_config["source_citer_task"],  # type: ignore
        )

    @task
    def conflicts_of_interest_task(self) -> Task:
        return Task(
            config=self.tasks_config["conflicts_of_interest_task"],  # type: ignore
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_report_task"],  # type: ignore
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalyzingContractClausesForConflictsAndSimilarities crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=self.tasks,  # type: ignore
            process=Process.sequential,
            verbose=True,
        )
