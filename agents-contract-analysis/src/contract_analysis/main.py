#!/usr/bin/env python
import sys

from contract_analysis.crew import (
    AnalyzingContractClausesForConflictsAndSimilaritiesCrew,
)


def run():
    """
    Run the crew.
    """
    inputs = {
        "query": """
            What are the differences in how contracts
            define warranties within creditcardscominc and digitalcinemadestination
        """,
    }
    AnalyzingContractClausesForConflictsAndSimilaritiesCrew().crew().kickoff(
        inputs=inputs
    )


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "database_credentials": "sample_value",
        "database_type": "sample_value",
        "specific_clauses": "sample_value",
    }
    try:
        AnalyzingContractClausesForConflictsAndSimilaritiesCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AnalyzingContractClausesForConflictsAndSimilaritiesCrew().crew().replay(
            task_id=sys.argv[1]
        )

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "database_credentials": "sample_value",
        "database_type": "sample_value",
        "specific_clauses": "sample_value",
    }
    try:
        AnalyzingContractClausesForConflictsAndSimilaritiesCrew().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()
