from typing import List
from pydantic import BaseModel, Field

class Query(BaseModel):
    """
    Represents a financial query involving placeholders for key financial attributes: 
    <BUIS>, <DATE>, <NET>, <FACTOR>, <PROF_LOSS>, <CUR>, <PF>, and <DSK>.  

    Example Queries:  
    - What are the <FACTOR>s that contributed the highest <NET> profit/loss?  
    - Which <CUR> currencies are driving the top-performing portfolios <PF>?  
    """
    query: str = Field(..., title="Financial query using placeholders <BUIS>, <DATE>, <NET>, <FACTOR>, <PROF_LOSS>, <CUR>, <PF>, and <DSK>.")

class FinancialQueries(BaseModel):
    """
    A collection of structured queries designed to generate financial summaries.  
    Each query should use placeholders (<FIELD>) instead of actual values.  
    """
    queries: List[Query] = Field(..., title="List of financial queries using placeholders <FIELD> instead of actual values.")


class SQLScript(BaseModel):
    """
    SQL Script to query data from the given table. You have to use this tool to generate the SQL script.
    """
    sql_script: str = Field(..., title="SQL Script to query data from the given table.")
    columns: List[str] = Field(..., title="Which columns are being projected in the SQL script.")
    description: str = Field(..., title="What does the SQL script do in Finance Analyst's perspective")

