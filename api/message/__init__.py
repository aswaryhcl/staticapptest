
import logging
import os
import json
import requests
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
import azure.functions as func
import snowflake.connector
import os,openai
from openai import AzureOpenAI
import pandas as pd
import re
from snowflake.connector import DictCursor
# import inspect
# from langchain.agents import create_pandas_dataframe_agent


snowflake_account = os.getenv("snowflake_account")
snowflake_user = os.getenv("snowflake_user")
snowflake_password = os.getenv("snowflake_password")
snowflake_warehouse = os.getenv("snowflake_warehouse")
snowflake_database = os.getenv("snowflake_database")
snowflake_schema = os.getenv("snowflake_schema")

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY") 
search_api_version = '2023-07-01-Preview'
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

AOAI_chat_endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AOAI_chat_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
AOAI_embd_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AOAI_embd_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AOAI_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

sql_db_server = os.getenv("SQL_DB_SERVER")
sql_db_user = os.getenv("SQL_DB_USER")
sql_db_password = os.getenv("SQL_DB_PASSWORD")
sql_db_name = os.getenv("SQL_DB_NAME")

blob_sas_url = os.getenv("BLOB_SAS_URL")

QRANT_HOST= os.getenv("QRANT_HOST")
QRANT_PORT= os.getenv("QRANT_PORT")
QDRANT_COLLECTION_NAME= os.getenv("QDRANT_COLLECTION_NAME")

collection_name = QDRANT_COLLECTION_NAME
vector_name = "content_vector"
top_k = 6

# font color adjustments
blue, end_blue = '\033[36m', '\033[0m'

place_orders = False


AzureOpenAIclient = AzureOpenAI(
            azure_endpoint = AOAI_chat_endpoint, 
            api_key = AOAI_chat_key,  
            api_version = AOAI_api_version,
            azure_deployment=chat_deployment
            )


functions = [
        {
            "name": "referCkaKnowledgeBase",
            "description": "Find information from CKA knowledgebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {
                        "type": "string",
                        "description": "User question (i.e., Any technical help etc.)"
                        },
                    },
                "required": ["user_question"],
                }
        },
        {
            "name": "analyticsinsights",
            "description": "Answer User questions related to a snowflake data model as a Snowflake SQL data analyst expert. Use only if the requested information if not already available in the conversation context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {
                        "type": "string",
                        "description": "User question related to orders, products or accounts (i.e., what were the most sold products?, who has the most loyalty points, how many products were sold in last 1 month, how many orders were placed etc.)"
                    },
                },
                "required": ["user_question"],
            }
        }
    ]


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('Python HTTP trigger function processed a request.')

        inputdata = json.loads(req.get_body())

        #messages=inputdata['messages']
        #skill=inputdata['skill']
        messages = inputdata.get("messages")
        skill = inputdata.get("skill")

        products = []

        available_functions = {
                    "referCkaKnowledgeBase": referCkaKnowledgeBase,
                    "analyticsinsights": analyticsinsights
            }
        
        if skill =='cka':
            function_name = {"name": "referCkaKnowledgeBase"}
            print('got function to call: {0}'.format(function_name['name']))
        elif skill == 'analyticsinsights':
            function_name = {"name": "analyticsinsights"} 
            print('got function to call: {0}'.format(function_name['name']))
        else:
            function_name = "none"
            print('got function to call: {0}'.format(function_name))

    
        print('\n**** Calling Chat Completions # 1 ****')
        #response = chat_complete(messages, functions= functions, function_call= function_name) 

        response = AzureOpenAIclient.chat.completions.create(
            model=chat_deployment,
            messages=messages,
            functions=functions,
            function_call=function_name, 
        )

        response_message = response.choices[0].message
        print(response_message)
    

        if response_message.function_call:
            print('\nwe got functions')

            function_name = response_message.function_call.name

            # verify function exists
            if function_name not in available_functions:
                return "Function " + function_name + " does not exist"
            function_to_call = available_functions[function_name]  
            
            # verify function has correct number of arguments
            function_args = json.loads(response_message.function_call.arguments)
            '''
            if check_args(function_to_call, function_args) is False:
                print('Invalid number of arguments for function')
                return "Invalid number of arguments for function: " + function_name
            '''

            print('\n**** Response **** : \n{}'.format(response))
            function_args = json.loads(response_message.function_call.arguments)

            response_message = response.choices[0].message
            print('\n**** response_message 1 **** : \n{0}'.format(response_message))
            function_args = json.loads(response_message.function_call.arguments)

            print('\n**** function_args **** : \n{}'.format(str(function_args)))

            # calling function
            print('\n *** Calling Function:*** ')
            function_response = function_to_call(**function_args)

            print("Output of function call:")
            print(function_response)
            print()

            if len(function_response) < 1:
                function_response="I am sorry, I could not find you an answer. Kindly either ask questions based on data in the skill OR change the skill from the drop-down"

            # adding assistant response to messages
            messages.append( 
                {
                    "role": response_message.role,
                    "function_call": {
                        "name": function_name,
                        "arguments": response_message.function_call.arguments,
                    },
                    "content": None
                }
            )
            print('\n *** message # 1 :*** \n{0}'.format(str(messages)))

            print('\n**** function_args **** : \n{}'.format(str(function_args)))

            # adding function response to messages
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            ) 

            print("\nMessages in second request:")
            for message in messages:
                print(message)
            print()

            print('\n**** Calling Chat Completions # 2 ****')
            #response = chat_complete(messages, functions=functions, function_call='none')
            second_response = AzureOpenAIclient.chat.completions.create(
                messages=messages,
                model=chat_deployment
            )
            print('\n *** Response # 2 :*** \n{0}'.format(second_response))
            
            response_message = second_response.choices[0].message
            print('\n*** response_message 2: {0}'.format(str(response_message)))

    
        messages.append({'role' : response_message.role, 'content' : response_message.content})

        print('\n *** Final message :*** \n{0}'.format(str(messages)))    

        response_object = {
                "messages": messages,
                "products": products
            }
        
        return func.HttpResponse(
            json.dumps(response_object),
            status_code=200
        )

    except Exception as e:
        error_message = {"error":str(e)}
        return func.HttpResponse(
            json.dumps(error_message),
            status_code=200
        )
    
 
def connect_to_snowflake():
        try:
            snowflake.connector.paramstyle='qmark'
            conn = snowflake.connector.connect(
                user=snowflake_user,
                password=snowflake_password,
                account=snowflake_account,
                warehouse=snowflake_warehouse,
                database=snowflake_database,
                schema=snowflake_schema
            )
            #st.success("Connection to Snowflake successful!")
            return conn
        except Exception as e:
            print(f"Error connecting to Snowflake: {str(e)}")
            return None
        



def retrieve_data_model():
    conn=connect_to_snowflake()
    cur = conn.cursor()
    query = f"SHOW TABLES IN {snowflake_schema}"
    cur.execute(query)
    tables = [row[1] for row in cur.fetchall()]
    data_model = {}
    for table in tables:
        column_query = f"SHOW COLUMNS IN {snowflake_schema}.{table}"
        cur.execute(column_query)
        columns = [f"{row[2]}" for row in cur.fetchall()]
        data_model[table] = columns
    cur.close()
    data_model_str = ""
    for table, columns in data_model.items():
        data_model_str += f"{table} ({', '.join(columns)});\n"

    data_model = {}
    for line in data_model_str.strip().split('\n'):
                table, columns = re.match(r'^(\w+)\s*\((.*?)\)\s*;', line).groups()
                data_model[table] = [col.strip() for col in columns.split(',')]
    return data_model_str



def analyticsinsights(user_question):
    data_model=retrieve_data_model()

    system_prompt="""-- You are a Snowflake expert, and your job is to return a valid Snowflake executable query. \n -- Use the following data model of the schema to answer the user's question, use only valid column name provide in:\n\n{0}.\n -- If data type of column is varchar, convert the column values into lowercase""".format(data_model)
    analytics_message=[{
                        "role": "system", 
                        "content": system_prompt
                    }]
    analytics_message.append({"role": "user", "content":user_question})


    get_query = AzureOpenAIclient.chat.completions.create(
                                            messages=analytics_message,
                                            model=chat_deployment,
                                            temperature=0,
                                            max_tokens=150,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            stop=["#", ";", "User Query"]
                                        )
    
    response_text = get_query.choices[0].message.content
    print('response_text:{0}'.format(response_text))
    response_text = response_text.lower()


    # Define a custom stop pattern based on your use case
    custom_stop_pattern = r'\bexplanation:.*'

    # Find the index of the custom stop pattern
    custom_stop_match = re.search(custom_stop_pattern, response_text, re.IGNORECASE)

    if custom_stop_match:
        stop_index = custom_stop_match.start()
        print('stop_index:{0}'.format(str(stop_index)))
        response_text = response_text[:stop_index]
        print(response_text)

    # Find the index of the first SQL query appearance
    start_index = response_text.find("SELECT".lower())


    # Extract the SQL query
    try:
        generated_sql_query = response_text[start_index:]
        generated_sql_query = generated_sql_query.replace('`', '')  # Remove trailing semicolon
        generated_sql_query = generated_sql_query.rstrip()

        sql_result=execute_sql_query(generated_sql_query)

        result_df = pd.DataFrame.from_dict(sql_result)
        sql_query_result=result_df.to_string()
        analytics_message.append({"role":"user","content":"{0}. Respond in complete sentence. Use the dataframe provided to generate response".format(user_question)})
        analytics_message.append({"role":"assistant","content":sql_query_result})
    except Exception as e:
        analytics_message.append({"role":"assistant","content":"this can not be answered with available information. Please let me know if I can help you in anything else"})
        
    output=AzureOpenAIclient.chat.completions.create(
            messages=analytics_message,
            model=chat_deployment,
            temperature=0.7,
            max_tokens=2048,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    
    response=output.choices[0].message.content


    response_text = re.sub(r'<\|.*?\|>', '', response)
    response_text = response_text.strip()
    return response_text

    


# helper method used to check if the correct arguments are provided to a function
'''
def check_args(function, args):
    sig = inspect.signature(function)
    print(sig)
    params = sig.parameters
    print(params)

    # Check if there are extra arguments
    for name in args:
        print('\nname: {0}'.format(name))
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True
'''

def execute_sql_query(query, params=None):
    """Execute a SQL query and return the results."""
    
    results = []
    conn=connect_to_snowflake()

    # Establish the connection
    with conn as db_conn_sf, db_conn_sf.cursor() as db_cursor_sf:
                
        if params:
            db_cursor_sf.execute(query, params)
        else:
            db_cursor_sf.execute(query)
        # If the query is a SELECT statement, fetch results
        if query.strip().upper().startswith('SELECT'):
            results = db_cursor_sf.fetchall()
            # results = db_cursor_sf.fetch_pandas_all()
         
        conn.commit()

    print(results)
    return results
        
def generate_embeddings(text):
    """ Generate embeddings for an input string using embeddings API """

    url = f"{AOAI_embd_endpoint}/openai/deployments/{embeddings_deployment}/embeddings?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_embd_key,
    }

    data = {"input": text}

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    return response['data'][0]['embedding']


def referCkaKnowledgeBase(user_question):
    embedded_query = generate_embeddings(user_question)
    client = QdrantClient(host=QRANT_HOST, port=QRANT_PORT)

    query_results = client.search(
                collection_name=collection_name,
                query_vector=(vector_name, embedded_query),
                limit=top_k,
            )
    
    context_result = query_results[0].payload["content"]
    return context_result


def skillmandatory(skill):
    responseResult = "Looks like this question is very specific, Can you please select a relevant Skill ?"
    return responseResult


def chat_complete(messages, functions, function_call='auto'):
    """  Return assistant chat response based on user query. Assumes existing list of messages """
    
    url = f"{AOAI_chat_endpoint}/openai/deployments/{chat_deployment}/chat/completions?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_chat_key
    }

    data = {
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "temperature" : 0,
        "max_tokens": 300
    }

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()

    return response
