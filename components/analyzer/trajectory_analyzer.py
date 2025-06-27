from .base_analyzer import BaseAnalyzer
from langchain_core.prompts import ChatPromptTemplate
from utils.utils import load_prompt_template

import logging
from pydantic import BaseModel, Field
from typing import Dict, Union, List, Tuple

class FailureAnalysis(BaseModel):
    """Analysis of the failure cases in evolution trajectory."""
    result: int = Field(description="Result of the evolution. If the evolution is passed, please output 1. If the evolution is failed, please output 0.")
    reasons: Dict[str, str] = Field(description="Reason of why the evolution is failed. Identify cases that failed to evolve, and provide their case ID as key and reasons as value.")
    feedback: str = Field(description="Feedback to improve the evolution process based on the reasons.")
    
class SuccessAnalysis(BaseModel):
    """Analysis of the success cases in evolution trajectory."""
    result: int = Field(description="Result of the evolution. If the evolution is passed, please output 1. If the evolution is failed, please output 0.")
    feedback: str = Field(description="Feedback to improve the evolution process. Please provide feedbacks in the context of not only complexity but also diversity and coherence.")

class TrajectoryAnalysis(BaseModel):
    """Analysis of the evolution trajectory."""
    response: Union[FailureAnalysis, SuccessAnalysis] = Field(description="Response of the evolution trajectory analysis.")
    
    
class TrajectoryAnalyzer(BaseAnalyzer):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.system_prompt = load_prompt_template("components/analyzer/prompts/system_prompt.prompt")
        self.trajectory_analysis_prompt = load_prompt_template("components/analyzer/prompts/trajectory_analysis_prompt.prompt")
        
        
    def preprocess_trajectory(self, trajectory : List[List[str]]) -> str:
        whole_trajectory = ""
        for trace in trajectory:
            for idx, stage in enumerate(trace):
                whole_trajectory += f"Stage {idx}: {stage}\n"
            whole_trajectory += "\n"
        return whole_trajectory
        
        
    def analyze_trajectory(self, trajectory : List[List[str]]) -> Tuple[str, str | None]:
        analyzer = self.llm.with_structured_output(TrajectoryAnalysis)
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", self.trajectory_analysis_prompt),
            ]
        )
        chain = prompt | analyzer
        whole_trajectory = self.preprocess_trajectory(trajectory)
        print("Trajectory Analysis has started...")
        if whole_trajectory:
            logging.info(f"Whole Trajectory : \n{whole_trajectory}")
        else:
            logging.warning("No trajectory Detected.")
        feedback = chain.invoke({"trajectory": whole_trajectory})
        logging.info(f"Trajectory Analysis Result : \n{feedback}")

        try:
            if feedback.response.result:
                logging.info(f"Current method is suitable for the evolution process.")
                return 'Passed', feedback.response.feedback
            else:
                logging.info(f"Current method is not suitable:\n{feedback.response.reasons}")
                return 'Failed', feedback.response.feedback
        except Exception as e:
            logging.error(f"Error in analyzing trajectory:\n{e}")
            return 'Error', None

    def analyze(self, trajectory : List[List[str]]) -> Tuple[str, str | None]:
        return self.analyze_trajectory(trajectory)