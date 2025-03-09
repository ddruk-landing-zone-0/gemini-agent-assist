from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .agent_functions import start_agent, end_agent, refine_old_summary_agent, generate_subj_query_agent, generate_stat_query_agent, register_data, generate_sql_script_agent, sql_result_agent, generate_bucket_query_agent, generate_final_result
from ..models.agent_state import AgentState


class MyAgent:
    def __init__(self, thread_id=None):
        self.config = None
        self.app = None
        self.build(thread_id)

    def build(self, thread_id):
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node('start', start_agent)
        workflow.add_node('end', end_agent)
        workflow.add_node('refine_old_summaries', refine_old_summary_agent)
        workflow.add_node('generate_subj_queries', generate_subj_query_agent)
        workflow.add_node('generate_stat_queries', generate_stat_query_agent)
        workflow.add_node('register_data', register_data)
        workflow.add_node('sql_script_generation', generate_sql_script_agent)
        workflow.add_node('sql_result_generation', sql_result_agent)
        workflow.add_node('bucket_query_generation', generate_bucket_query_agent)
        workflow.add_node('final_result', generate_final_result)

        # Edges
        workflow.add_edge('start', 'refine_old_summaries')
        workflow.add_edge('refine_old_summaries', 'generate_subj_queries')
        workflow.add_edge('generate_subj_queries', 'generate_stat_queries')
        workflow.add_edge('generate_stat_queries', 'register_data')
        workflow.add_edge('register_data', 'sql_script_generation')
        workflow.add_edge('sql_script_generation', 'sql_result_generation')
        workflow.add_edge('sql_result_generation', 'bucket_query_generation')
        workflow.add_edge('bucket_query_generation', 'final_result')
        workflow.add_edge('final_result', 'end')

        # Compile
        workflow.set_entry_point('start')
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        self.config = {"configurable":{"thread_id":str(thread_id)}}

    def get_recent_state_snap(self):
        snap = self.app.get_state(self.config).values.copy()
        return snap
    
    def get_graph(self):
        graph = self.app.get_graph(xray=True)
        return graph
    
    def continue_flow(self, state):
        self.app.invoke(state, config=self.config)
        return self.get_recent_state_snap()
    


MY_AGENT = MyAgent(thread_id=1)
graph = MY_AGENT.get_graph()
print(graph.draw_ascii())