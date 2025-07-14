from airflow import DAG
from datetime import datetime
from airflow.decorators import task
from langchain.agents import create_react_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated
import operator

#import os
#os.environ['OPENAI_API_KEY'] = ""
#os.environ['EXA_API_KEY'] = ""
#os.environ['SERPER_API_KEY'] = ""
#os.environ['LINKUP_API_KEY'] = ""

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool
search = GoogleSerperAPIWrapper()

serp_tool = Tool(
    name="Intermediate_Answer",
    func=search.run,
    description=(
        "Useful for researching companies that meet specific criteria, such as not using AWS, "
        "having revenue under $100 million, and being based in the USA. Use this tool to look up "
        "company details, cloud service providers, financial data, and headquarters location from public sources."
    ),
)

from langchain_openai import ChatOpenAI
llm_1 = ChatOpenAI(model='gpt-4.1-mini-2025-04-14', temperature=0)
llm_2 = ChatOpenAI(model='gpt-4.1-mini-2025-04-14', temperature=0)
llm_3 = ChatOpenAI(model='gpt-4.1-mini-2025-04-14', temperature=0)

from langchain.prompts import PromptTemplate

prompt_1 = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

RESEARCH METHODOLOGY - Streamlined 2-phase approach for efficiency:

PHASE 1 - CRUNCHBASE DISCOVERY (Use serper_tool):
Primary focus on Crunchbase as it's highly reliable:
- Search "site:crunchbase.com companies USA revenue under 100 million"
- Search "site:crunchbase.com startups USA funding under 50 million"
- Search "Crunchbase small companies not AWS alternative cloud"
- Search "Crunchbase manufacturing companies USA under 100M"
- Search "Crunchbase SaaS companies USA under 100M Azure Google Cloud"

PHASE 2 - QUICK TECHNOLOGY VERIFICATION (Use serper_tool):
Light verification for final confirmation:
- Search "[company_name] uses Azure" or "[company_name] Google Cloud customer"
- Search "[company_name] NOT AWS alternative cloud provider"
- Search "[company_name] on-premises infrastructure"
- If unclear, search "[company_name] technology stack" for quick confirmation

QUALIFYING CRITERIA CHECKLIST - Each company must meet ALL criteria:
✓ Based in USA (headquarters location)
✓ Revenue less than $100 million (verified through multiple sources)
✓ NOT using AWS (confirmed via technology stack or explicit statements)
✓ Using alternative cloud providers (Azure, GCP, on-premises, or hybrid)

SEARCH STRATEGY TIPS:
- Use specific Boolean operators: "company AND revenue AND -AWS"
- Target industry-specific searches to find niche companies
- Look for technology case studies and customer success stories
- Check multiple sources for revenue verification
- Prioritize companies with explicit non-AWS mentions

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Find 10 companies based in the USA that are not using AWS and have revenue less than 100 million dollars. Use the `serper_tool` to systematically search for smaller US companies, checking their cloud infrastructure preferences and financial information. Focus on companies that explicitly mention using alternative cloud providers (like Google Cloud, Microsoft Azure, or on-premises solutions) or are small enough to likely avoid AWS due to cost considerations.

IMPORTANT: Follow the streamlined 2-phase methodology above. Leverage Crunchbase's reliable data in Phase 1 to build a strong initial list (aim for 15-20 companies), then do light verification in Phase 2 (spend max 1-2 searches per company). Prioritize speed and quantity over perfect verification - basic confirmation is sufficient.

{input}
Thought:{agent_scratchpad}"""
)

agent_1 = create_react_agent(llm=llm_1, prompt=prompt_1, tools=[serp_tool])

agent_1 = AgentExecutor(
    agent=agent_1,
    tools=[serp_tool],
    handle_parsing_errors=True,
    max_execution_time=1800,  # 30 minutes (increased from 5 minutes)
    max_iterations=50,        # Reduced from 200 to prevent infinite loops
    verbose=True,
    return_intermediate_steps=True     # Keep track of progress
)

prompt_2 = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

RESEARCH METHODOLOGY - Streamlined 2-phase approach for efficiency:

PHASE 1 - CRUNCHBASE DISCOVERY (Use serper_tool):
Primary focus on Crunchbase as it's highly reliable:
- Search "site:crunchbase.com companies USA revenue under 100 million"
- Search "site:crunchbase.com startups USA funding under 50 million"
- Search "Crunchbase small companies not AWS alternative cloud"
- Search "Crunchbase manufacturing companies USA under 100M"
- Search "Crunchbase SaaS companies USA under 100M Azure Google Cloud"

PHASE 2 - QUICK TECHNOLOGY VERIFICATION (Use serper_tool):
Light verification for final confirmation:
- Search "[company_name] uses Azure" or "[company_name] Google Cloud customer"
- Search "[company_name] NOT AWS alternative cloud provider"
- Search "[company_name] on-premises infrastructure"
- If unclear, search "[company_name] technology stack" for quick confirmation

QUALIFYING CRITERIA CHECKLIST - Each company must meet ALL criteria:
✓ Based in USA (headquarters location)
✓ Revenue less than $100 million (verified through multiple sources)
✓ NOT using AWS (confirmed via technology stack or explicit statements)
✓ Using alternative cloud providers (Azure, GCP, on-premises, or hybrid)

SEARCH STRATEGY TIPS:
- Use specific Boolean operators: "company AND revenue AND -AWS"
- Target industry-specific searches to find niche companies
- Look for technology case studies and customer success stories
- Check multiple sources for revenue verification
- Prioritize companies with explicit non-AWS mentions

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: ONLY provide a clean numbered list of companies found. Format as:
1. Company Name - Industry - Revenue Range - Cloud Provider
2. Company Name - Industry - Revenue Range - Cloud Provider
(etc.)

Do NOT include explanations, methodology descriptions, or additional commentary in the Final Answer.

Begin!

Question: Find 10 companies based in the USA that are not using AWS and have revenue less than 100 million dollars. Use the `serper_tool` to systematically search for smaller US companies, checking their cloud infrastructure preferences and financial information. Focus on companies that explicitly mention using alternative cloud providers (like Google Cloud, Microsoft Azure, or on-premises solutions) or are small enough to likely avoid AWS due to cost considerations.

IMPORTANT: Follow the streamlined 2-phase methodology above. Leverage Crunchbase's reliable data in Phase 1 to build a strong initial list (aim for 15-20 companies), then do light verification in Phase 2 (spend max 1-2 searches per company). Prioritize speed and quantity over perfect verification - basic confirmation is sufficient.

CRITICAL FORMATTING RULES:
- Always follow Thought with Action
- Always follow Action with Action Input
- Never skip any step in the format
- If you don't know what action to take, use the serper_tool to search

{input}
Thought: {agent_scratchpad}"""
)

agent_2 = create_react_agent(llm=llm_2, prompt=prompt_2, tools=[serp_tool])

agent_2 = AgentExecutor(
    agent=agent_2,
    tools=[serp_tool],
    handle_parsing_errors=True,
    max_execution_time=1800,  # 30 minutes (increased from 5 minutes)
    max_iterations=50,        # Reduced from 200 to prevent infinite loops
    verbose=True,
    return_intermediate_steps=True     # Keep track of progress
)

prompt_3 = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

RESEARCH METHODOLOGY - Streamlined 2-phase approach for efficiency:

PHASE 1 - CRUNCHBASE DISCOVERY (Use serper_tool):
Primary focus on Crunchbase as it's highly reliable:
- Search "site:crunchbase.com companies USA revenue under 100 million"
- Search "site:crunchbase.com startups USA funding under 50 million"
- Search "Crunchbase small companies not AWS alternative cloud"
- Search "Crunchbase manufacturing companies USA under 100M"
- Search "Crunchbase SaaS companies USA under 100M Azure Google Cloud"

PHASE 2 - QUICK TECHNOLOGY VERIFICATION (Use serper_tool):
Light verification for final confirmation:
- Search "[company_name] uses Azure" or "[company_name] Google Cloud customer"
- Search "[company_name] NOT AWS alternative cloud provider"
- Search "[company_name] on-premises infrastructure"
- If unclear, search "[company_name] technology stack" for quick confirmation

QUALIFYING CRITERIA CHECKLIST - Each company must meet ALL criteria:
✓ Based in USA (headquarters location)
✓ Revenue less than $100 million (verified through multiple sources)
✓ NOT using AWS (confirmed via technology stack or explicit statements)
✓ Using alternative cloud providers (Azure, GCP, on-premises, or hybrid)

SEARCH STRATEGY TIPS:
- Use specific Boolean operators: "company AND revenue AND -AWS"
- Target industry-specific searches to find niche companies
- Look for technology case studies and customer success stories
- Check multiple sources for revenue verification
- Prioritize companies with explicit non-AWS mentions

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: ONLY provide a clean numbered list of companies found. Format as:
1. Company Name - Industry - Revenue Range - Cloud Provider
2. Company Name - Industry - Revenue Range - Cloud Provider
(etc.)

Do NOT include explanations, methodology descriptions, or additional commentary in the Final Answer.

Begin!

Question: Find 10 companies based in the USA that are not using AWS and have revenue less than 100 million dollars. Use the `serper_tool` to systematically search for smaller US companies, checking their cloud infrastructure preferences and financial information. Focus on companies that explicitly mention using alternative cloud providers (like Google Cloud, Microsoft Azure, or on-premises solutions) or are small enough to likely avoid AWS due to cost considerations.

IMPORTANT: Follow the streamlined 2-phase methodology above. Leverage Crunchbase's reliable data in Phase 1 to build a strong initial list (aim for 15-20 companies), then do light verification in Phase 2 (spend max 1-2 searches per company). Prioritize speed and quantity over perfect verification - basic confirmation is sufficient.

CRITICAL FORMATTING RULES:
- Always follow Thought with Action
- Always follow Action with Action Input
- Never skip any step in the format
- If you don't know what action to take, use the serper_tool to search

{input}
Thought: {agent_scratchpad}"""
)

agent_3 = create_react_agent(llm=llm_2, prompt=prompt_2, tools=[serp_tool])

agent_3 = AgentExecutor(
    agent=agent_3,
    tools=[serp_tool],
    handle_parsing_errors=True,
    max_execution_time=1800,  # 30 minutes (increased from 5 minutes)
    max_iterations=50,        # Reduced from 200 to prevent infinite loops
    verbose=True,
    return_intermediate_steps=True     # Keep track of progress
)

class State(TypedDict):
    result: Annotated[list[str], operator.add] # Explicitly type the list content

def agent_1_node(state: State):
    response = agent_1.invoke({"input": "Find me companies not using AWS and having a revenue of less than 100 million dollars"})
    # Ensure the response contains 'output' and return it as a list
    return {"result": [response['output']]}

def agent_2_node(state: State):
    response = agent_2.invoke({"input": "Find me companies not using AWS and having a revenue of less than 100 million dollars"})
    # Ensure the response contains 'output' and return it as a list
    return {"result": [response['output']]}

def agent_3_node(state: State):
    response = agent_3.invoke({"input": "Find me companies not using AWS and having a revenue of less than 100 million dollars"})
    # Ensure the response contains 'output' and return it as a list
    print()
    return {"result": [response['output']]}

graph = StateGraph(State)

graph.add_node("agent_1_node", agent_1_node)
graph.add_node("agent_2_node", agent_2_node)
graph.add_node("agent_3_node", agent_3_node)

graph.add_edge(START, "agent_1_node")
graph.add_edge("agent_1_node", "agent_2_node")
graph.add_edge("agent_2_node", "agent_3_node")
graph.add_edge("agent_3_node", END)

graph = graph.compile()

with DAG(
    dag_id="sky_switch_dag",
    start_date=datetime(2025,6,26),
    schedule="@hourly"
) as dag:
    
    @task
    def generate_company():
        response = graph.invoke({"result": []})
        return response
    
    @task
    def format_data(response: list):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "from the provided corpus extract the company names,revenue and the cloud used"),
            ("system", "Make sure no company is left out"),
            ("user", "{input}")
        ])
        chain = prompt | llm_1
        result = chain.invoke({"input": response})
        return result.content
    
    @task
    def data_db_add(result:str):
        embeddings = OpenAIEmbeddings()
        docs = [Document(page_content=result)]
        rcts = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        split_docs = rcts.split_documents(docs)
        db = Neo4jVector.from_documents(
            documents=split_docs,
            embedding=embeddings,
            #url="",
            #username="",
            #password=""
        )

    generate_company_task=generate_company()
    format_data_task=format_data(generate_company_task)
    data_db_add_task=data_db_add(format_data_task)

    