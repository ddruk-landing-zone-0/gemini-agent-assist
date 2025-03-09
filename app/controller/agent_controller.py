# from ..service.agent_build_service import MY_AGENT

# init_state_snap = {
#     'state': 'start',
#     'model':{
#         'model_name':'gemini-2.0-flash-001',
#         'temperature': 0.5,
#         'max_output_tokens': 512,
#         'max_retries':5,
#         'wait_time':30
#     },
#     'results':{
#         'refine_old_summaries':[],
#         'subj_query_generation':[],
#         'stat_query_generation':[],
#         'sql_script_generation':[],
#         'sql_result':[],
#         'bucket_query_generation':[],
#         'final_result':[]
#     },
#     'cache_location':{
#         "sample_summarized_pnl_commentaries":"../sample_data/sample_summarized_pnl_commentaries.json",
#         "rule_based_title_comment_data":"../sample_data/rule_based_title_comment_data.json",
        
#         "refine_old_summaries":"../sample_data/cached/refine_old_summaries.json",
#         "subj_query_generation":"../sample_data/cached/subj_query_generation.json",
#         "stat_query_generation":"../sample_data/cached/stat_query_generation.json",
#         "sql_script_generation":"../sample_data/cached/sql_script_generation.json",
#         "sql_result":"../sample_data/cached/sql_result.json",
#         "final_result":"../sample_data/cached/final_result.json"
#     },
#     'cache_flag':{
#         "refine_old_summaries":True,
#         "subj_query_generation":True,
#         "stat_query_generation":True,
#         "sql_script_generation":True,
#         "sql_result":True,
#         "final_result":True
#     }
# }

# MY_AGENT.continue_flow(init_state_snap);