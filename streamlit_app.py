import streamlit as st
from openai import OpenAI
import os
from autogen import AssistantAgent, UserProxyAgent, register_function, ConversableAgent
import asyncio
import autogen
import pyodbc
import json 
from tools import search_component, search_sous_function, search_function, init_context, search_component_interconnections

 
# Show title and description.
st.title("💬 Agent Prototype ")
st.write(
    "This model is in alpha version "
)

# Show title and description.

if "messages" not in st.session_state:
    st.session_state.messages = []

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        st.session_state.messages.append({"name": sender.name, "content": message})
        st.chat_message(sender.name).write(message)
        return super()._process_received_message(message, sender, silent)


if "chatgroup" not in st.session_state and "user_proxy" not in st.session_state and "agent" not in st.session_state :
    llm_config = {
        "config_list": [{"model": "gpt-4o", "api_key": st.secrets["OPENAI_API_KEY"]}],
    }

    

    agent = ConversableAgent(
    name="assistant",
    system_message=f"""You are an expert in industrial maintenance. 
    Your task is to help the user identify the component reference causing an issue in a machine. 
    
    Think step by step: 
    You are smart and have intuition, base on the symptom you can map the function, subfunction or symptome of a machine. 
    Keep in mind that some components have a direct impact (e.g., a circuit breaker failing directly stops the system from functioning) while others have indirect effects (e.g., a failing fan leading to overheating and subsequent problems in other components).
    In general the user describe an incident like 'The IHM is not working'. You need to get more data, to understand the problem and some time identify the linked component. 
    Exemple : The IHM is not turning on. The component is the IHM than if it's not turning on it can be power outage or screen issue. So you check to which component the IHM is interconnected to find if we have something for the power or the screen.
    
    Always explain your reasonning process and chain of thought

Machine Structure:

The machine is composed of functions, each containing sub-functions, which in turn link directly to components.
Examples:
Preheating Function: Includes sub-functions heating, regulation, control.
Group Raboutage Function: Includes sub-functions cutting, pressing, welding.
Group Usinage Function: Includes sub-functions milling, drilling, cutting.
Your Objective:

Identify the Relevant Function/Sub-Function:
The user can talk about function, sub function or component. If it's a component you can fetch sub function and component to determine what are the component. 
The structure of the machine is logic and you need to determine sub function or component based on parent node name or description.
Determine the function or sub-function related to the reported issue. For instance, if there’s a calibration issue tied to a heating component, focus on the Preheating function.

Differentiate Direct vs. Indirect Impacts:

Direct Impact (e.g., Circuit Breaker): If a circuit breaker fails, it has an immediate and direct effect, possibly shutting down the machine or part of it.
Indirect Impact (e.g., Fan Overheating): If a cooling fan malfunctions, it may cause overheating, which in turn affects other components indirectly. Always trace through these indirect links to find the root cause.
Use Available Tools:
search_function: Get details on a given function (name, description, ID, sub-functions).
search_sous_function: Get details on a sub-function (name, description, ID, components).
search_component: Get details on a component (name, description, ID, type, family).
search_component_interconnections: Understand how one component relates to others, revealing indirect impacts.
Use these tools to find the root cause by examining both direct and indirect relationships.

Always Provide the Component Reference:
In your final recommendation, explicitly name the component reference (not id) so the user can locate and fix it.

Be Resourceful and Clear:
Draw upon your maintenance expertise, the provided context, and the results of your tool usage to arrive at the best possible solution. Your ultimate goal is to name the correct component reference that will fix the user’s problem.

Always explain you reason

Context of the Machine:
{init_context("U7.5C")}""",
    llm_config=llm_config,
    )

    user_proxy = TrackableUserProxyAgent(
    name="user",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=0
    )   

    def execute(recipient, messages, sender, config): 
        if "tool_calls" in messages[-1]:
             print('tool execution')
             print(f"Messages sent to: {recipient.name} | num messages: {messages}")
             tool = recipient.generate_tool_calls_reply(messages, sender)
             messages.append(tool)
             recipient.send(tool[1],sender)
        return False, None  # required to ensure the agent communication flow continues
    
    user_proxy.register_reply(
        [autogen.Agent, None],
        reply_func=execute, 
        config={"callback": None},
    )

    register_function(search_component,
                      caller=agent,
                      executor=user_proxy,
                      name="search_component",
                      description="Search for a component in the database. Return the name of the component, the description, the id, the type and the family"
                      )
    register_function(search_sous_function,
                      caller=agent,
                      executor=user_proxy,
                      name="search_sous_function",
                      description="Search for a sous function in the database, Return the name of the sous function, the description, the id and the composants of the subfunction"
                      )
    register_function(search_function,
                      caller=agent,
                      executor=user_proxy,
                      name="search_function",
                      description="Search for a function in the database.  Return the name of the function, the description, the id and the subfunction of the function"
                      )
    register_function(search_component_interconnections,
                      caller=agent,
                      executor=user_proxy,
                      name="search_component_interconnections",
                      description="Search for a component interconnections in the database. Return component that are linked together"
                      )
    
    st.session_state["agent"]=agent
    st.session_state["user_proxy"]=user_proxy
    st.session_state["loop"] = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state["loop"])
    async def initiate_chat():
            await user_proxy.a_initiate_chat(
                agent,
                message="Bonjour comment ca va ?",
            )

    st.session_state["loop"].run_until_complete(initiate_chat())


    # Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["name"]):
        st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"name":"user_proxy", "content": prompt})
            st.chat_message("user_proxy").write(prompt)
            st.session_state["user_proxy"].send(prompt, st.session_state["agent"])
        # Store and display the current prompt.



