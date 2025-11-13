import time
# from defs import *
import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
try:
    load_dotenv(find_dotenv())
except:
    load_dotenv(Path("my/path/.env"))
import anthropic
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
MAX_TOKEN=5000
TEMPERATURE=0.0
MODEL="claude-3-7-sonnet-20250219"
MAX_TRY_OVERLOAD = 3

# g_message_history = []
# def clear_message_history():
#     global g_message_history
#     g_message_history = []


# def call_llm(systemMsg, messages):
#     for i in range(MAX_TRY_OVERLOAD):
#         try:
#             message = client.messages.create( model=MODEL, max_tokens=MAX_TOKEN, temperature=TEMPERATURE, system=systemMsg, messages= messages)
            
#             # print('[User]\n', messages[-1]["content"], "\n[END User]")
#             # print('[LLM]\n', message.content[0].text, "\n[END LLM]")
            
#             return message
#         except Exception as err:
#             if err.args[0].find("not resolve authentication method"):
#                 raise err
#             if err.args[0].find("overloaded_error"):
#                 if i<MAX_TRY_OVERLOAD-1:
#                     mprint("API Overloaded, trying again in 2 seconds...")
#                     time.sleep(2)
#                 else:
#                     raise err
                
# decomposeSystemMsg="Your role is to decompose natural language constraints into natural language lower-level constraints to later apply them to a given PDDL problem. Doing so consists in rephrasing and decomposing the initial contraint into one or several to remove ambiguities and more importantly to match the predicates defined in a given PDDL problem. It's important to note that constraints cannot directly affect an action, only predicates describing the state of the world. The lower-level constraints should be in natural language, no PDDL, and refer to predicates."
# def decompose(domain, problem, constraint):
#     global g_message_history
    
    
#     messages=[
#             {"role": "user", "content": "I will share a PDDL domain followed by a corresponding PDDL problem. After that, I will share the constraint to decompose."},
#             {"role": "assistant", "content": "Got it now share with me the PDDL domain and problem."},
#             {"role": "user", "content": domain + '\n' + problem},   
#             {"role": "assistant", "content": "Got it now share with me the constraints to decompose. I will format my answer in a very clear and consise numbered list where the lower-level constraints referring to predicates are the main items and the subitems are explanations or descriptions. I should not put any PDDL code in the main items."},
#             {"role": "user", "content": constraint},
#         ]
    
#     # Call LLM
#     message = call_llm(decomposeSystemMsg, messages)
    
#     # Update history with request and LLM answer
#     g_message_history += messages + [{"role": "assistant", "content": message.content[0].text}]
    
#     return message.content[0].text
# def redecompose(feedback):
#     global g_message_history
#     messages=[
#             {"role": "user", "content": feedback},
#         ]
    
#      # Call LLM
#     message = call_llm(decomposeSystemMsg, g_message_history + messages)

#     # Update history with request and LLM answer
#     g_message_history += messages + [{"role": "assistant", "content": message.content[0].text},]
    
#     return message.content[0].text

# def removeFormating(text):
    
#     # Remove initial white spaces and empty lines
#     newtext = ""
#     for l in text.splitlines():
#         if l=='':
#             continue
#         i = 0
#         while i<len(l) and l[i]==' ':
#             i+=1
#         l = l[i:]
#         newtext += l + '\n'
        
        
#     # Get main items
#     main_items = []
#     for l in newtext.splitlines():
#         try:
#             int(l[0])
#             main_items.append(l)
#         except:
#             continue
    
#     symbols = [
#         '#',
#         '*',
#         '=',
#         '-',
#     ]
    
#     newtext = ""
#     for l in main_items:
#         for s in symbols:
#             l = l.replace(s, '')
#         i = 0
#         while not l[i].isalpha():
#             i+=1
#         l = l[i:]
#         newtext += l + '\n'
#     newtext = newtext[:-1]
        
#     return newtext

# encodingSystemMsg="You are a PDDL planning expert. Your purpose is to translate natural language constraints into PDDL3.0 constraints to be used in a classical PDDL planner. Respond to the requested translations only with consise and accurate PDDL language. PDDL3.0 constraints should only concer predicates and functions, they cannot refer directly to actions."
# def encodePrefs(domain, problem, constraint):
#     global g_message_history
    
#     # Setup request
#     messages=[
#             {"role": "user", "content": "When translating natural language inputs into PDDL3.0 constraints, only use the following keywords: 'and','or','not','=','<','<=','>','>=','+','-','*','/','forall','exists','always','sometime','within','at-most-once','sometime-after','sometime-before','always-within','holding-during','hold-after','at-end'. After, I will share a PDDL domain followed by a corresponding PDDL problem. After that, I will share a contraint to translate."},
        
#             {"role": "assistant", "content": "Got it now share with me the PDDL domain and problem."},
#             {"role": "user", "content": domain + '\n' + problem},   
#             {"role": "assistant", "content": "Got it now share with me the constraint to translate."},
#             {"role": "user", "content": constraint},
#             # {"role": "assistant", "content": "When translating, only when possible and relevant, I should forall instead of enumerating all objects."},
#             # {"role": "assistant", "content": "I will now carefully translate the given constraint."},
#         ]
    
#     # Call LLM
#     message = call_llm(encodingSystemMsg, messages)
    
#     # Update history with request and LLM answer
#     g_message_history += messages + [{"role": "assistant", "content": message.content[0].text}]
    
#     return message.content[0].text
# def reencodePrefs(feedback):
#     global g_message_history
#     messages=[
#             {"role": "user", "content": feedback},
#         ]
    
#      # Call LLM
#     message = call_llm(encodingSystemMsg, g_message_history + messages)

#     # Update history with request and LLM answer
#     g_message_history += messages + [{"role": "assistant", "content": message.content[0].text},]
    
#     return message.content[0].text

### HDDL:
with open("./HDDL_method_description.txt", "r") as f:
    hddl_method_description = f.read()

g_message_history = []
def clear_message_history():
    global g_message_history
    g_message_history = []

def call_llm(systemMsg, messages, save_history=False, use_history=False):
    global g_message_history
    for i in range(MAX_TRY_OVERLOAD):
        try:
            if use_history:
                print("Using message history", g_message_history)
                parse_messages = g_message_history + messages
            else:
                parse_messages = messages
            message = client.messages.create( model=MODEL, max_tokens=MAX_TOKEN, temperature=TEMPERATURE, system=systemMsg, messages= parse_messages)
            
            # print('[User]\n', messages[-1]["content"], "\n[END User]")
            # print('[LLM]\n', message.content[0].text, "\n[END LLM]")
            if save_history:
                print("Saving message to history")
                g_message_history += messages + [{"role": "assistant", "content": message.content[0].text}]
                print("Saved message to history", g_message_history)
            return message.content[0].text
        except Exception as err:
            if err.args[0].find("not resolve authentication method"):
                raise err
            if err.args[0].find("overloaded_error"):
                if i<MAX_TRY_OVERLOAD-1:
                    mprint("API Overloaded, trying again in 2 seconds...")
                    time.sleep(2)
                else:
                    raise err

# def decompose_hddl(domain, problem, constraint):
#     global g_message_history
    
    
#     messages=[
#             {"role": "user", "content": "I will share a HDDL domain followed by a corresponding HDDL problem. After that, I will share the hierarchical strategy to decompose."},
#             {"role": "assistant", "content": "Got it now share with me the HDDL domain and problem."},
#             {"role": "user", "content": domain + '\n' + problem},   
#             {"role": "assistant", "content": "Got it now share with me the strategy to decompose. I will format my answer in a very clear and consise list of (1) method parameters, (2) relevant task, (3) a numbered list of ordered subtasks in a more human interpretable langauge. I should not put any HDDL code in the main items."},
#             {"role": "user", "content": constraint},
#         ]
    
#     # Call LLM
#     message = call_llm(decomposeSystemMsg, messages)
    
#     # Update history with request and LLM answer
#     g_message_history += messages + [{"role": "assistant", "content": message.content[0].text}]
    
#     return message.content[0].text

def encodePrefs_hddl(domain, problem, preferences):
    global g_message_history
    
    # Setup request
    systemMsg="You are a HDDL planning expert. Your purpose is to translate natural language human preferences (preferred strategy) in how to perform a task into a new HDDL method to be used in a classical HDDL planner. Respond to the requested translations only with consise and accurate HDDL language."
    messages=[
            {"role": "user", "content": "I will share with you a text describing how HDDL method works. After I will share a HDDL domain followed by a corresponding HDDL problem. After that I will start to share human preferred strategy to translate."},
            {"role": "user", "content": hddl_method_description},
            
        #     {"role": "user", "content": "When translating natural language inputs into PDDL3.0 constraints, only use the following keywords: 'and','or','not','=','<','<=','>','>=','+','-','*','/','forall','exists','always','sometime','within','at-most-once','sometime-after','sometime-before','always-within','holding-during','hold-after','at-end'. After, I will share a PDDL domain followed by a corresponding PDDL problem. After that, I will share human preferences to translate."},
        
            {"role": "assistant", "content": "Got it now share with me the HDDL domain and problem."},
            {"role": "user", "content": domain + '\n' + problem},   
            {"role": "assistant", "content": "Got it now share with me the human preferences to translate. Note that if the preferred strategy is similar to an existing method, I will return the name of the method."},
            {"role": "user", "content": preferences},
            # {"role": "user", "content":  "First, reformule and rephrase the preferences in two different manners to better understand it and remove ambiguities. After, give me the encoding for the given preferences."},
        ]
    
    # Call LLM
    message = client.messages.create( model=MODEL, max_tokens=MAX_TOKEN, temperature=TEMPERATURE, system=systemMsg, messages= messages)
    
    # Update history with request and LLM answer
    g_message_history += messages + [{"role": "assistant", "content": message.content[0].text}]
    
    print('LLM:', message.content[0].text, "[END LLM]")
    
    return message.content[0].text

def reencodePrefs_hddl(feedback):
    global g_message_history
    
    # Setup request
    system="You are a HDDL planning expert. Your purpose is to translate natural language human preferences into HDDL method to be used in a classical HDDL planner. Respond to the requested translations only with consise and accurate HDDL language."
    messages=[
            {"role": "user", "content": "Your last translation is incorrect. Carefully generate a new correct translation considering the following feedback: "+feedback},
        ]
    
    # Call LLM
    message = client.messages.create( model=MODEL, max_tokens=MAX_TOKEN, temperature=TEMPERATURE, system=system, messages=g_message_history + messages)
    
    # Update history with request and LLM answer
    g_message_history += messages + [{"role": "assistant", "content": message.content[0].text},]
    print('LLM:', message.content[0].text, "[END LLM]")
    
    return message.content[0].text




if __name__=='__main__':
    pass

