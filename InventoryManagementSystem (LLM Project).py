#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install langchain sqlalchemy pymysql sentence-transformers chromadb langchain_community google-api-python-client datasets


# In[4]:


from langchain_community.llms import GooglePalm


# In[5]:


api_key = ""
llm = GooglePalm(google_api_key=api_key, temperature=1.0)
response = llm("write about football in 2 sentences")
print(response)


# In[6]:


from langchain.utilities import SQLDatabase


db_user="root"
db_password="root1"
db_host="localhost"
db_name="atliq_tshirts"

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

print(db.table_info)


# In[31]:


from langchain_experimental.sql import SQLDatabaseChain
db_chain=SQLDatabaseChain.from_llm(llm,db,verbose=True)
qns1 = db_chain("How many t-shirts do we have left for nike in extra small size and white color?")


# In[32]:


qns2 = db_chain.run("How much is the price of the inventory for all small size t-shirts?")


# In[33]:


qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'")


# In[34]:


qns3 = db_chain.run("If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?")


# In[35]:


#feeding sql code to resolve the error and produce the right answer
sql_code = """
select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """

qns3 = db_chain.run(sql_code)


# In[36]:


qns4=db_chain.run("SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'")


# In[20]:


q5 = db_chain.run("How many white color Levi's t shirts we have available?")


# In[37]:


qns5 = db_chain.run("SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'")


# In[41]:


#using few shot learning to feed the complex queries gave the wrong output to fix the errors 
# Sample data
few_shots = [
    {'query': 'How many tshirts do we have for nike in small size and white color', 'sql_query': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'S'", 'result_label': 'Result of the SQL query', 'result': {'query': 'How many tshirts do we have for nike in small size and white color', 'result': '91'}},
    {'query': 'How much is the total price of the inventory for all S-size t-shirts?', 'sql_query': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'", 'result_label': 'Result of the SQL query', 'result': '14113'},
    {'query': 'If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?', 'sql_query': "SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from\n(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'\ngroup by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id\n ", 'result_label': 'Result of the SQL query', 'result': '17262.2'},
    {'query': 'If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?', 'sql_query': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'", 'result_label': 'Result of the SQL query', 'result': '17742'},
    {'query': "How many white color Levi's shirt I have?", 'sql_query': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'", 'result_label': 'Result of the SQL query', 'result': '90'}
]

def flatten_dict_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(flatten_dict_values(value))
        else:
            values.append(value)
    return values




# In[42]:


# Sample data

# Flatten and merge the values into single strings
to_vectorize = [" ".join(map(str, flatten_dict_values(example))) for example in few_shots]

print(to_vectorize)


# In[19]:


pip install -U langchain-huggingface


# In[39]:


get_ipython().system('pip install --upgrade urllib3')


# In[23]:


get_ipython().system('pip install urllib3==1.26.7')


# In[24]:


pip install --upgrade --force-reinstall botocore


# In[46]:


from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

e=embeddings.embed_query("How many white color Levi's shirt I have")
e[:5]


# In[40]:


[#example.values() for example in few_shots]


# In[23]:


#little tweaking has been done
# Ensure only string values are concatenated into to_vectorize


# In[43]:


to_vectorize[0]


# In[51]:


to_vectorize


# In[53]:


# Assuming few_shots is defined as per the previous example
# Convert dictionary items to strings
converted_few_shots = [
    {key: str(value) for key, value in example.items()}
    for example in few_shots
]




# In[56]:


converted_few_shots


# In[57]:


# Now use converted_few_shots as metadata
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=converted_few_shots)


# In[58]:


vectorstore 


# In[60]:


#semanticsimilairty selector is used to pull a similar vector data to the inpput query
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

example_selector.select_examples({"Question": "How many Adidas T shirts I have left in my store?"})


# In[63]:


#instructing langchain to use these instructions to create mysql queries
mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""


# In[62]:


from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

print(PROMPT_SUFFIX)


# In[64]:


from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)


# In[65]:


print(_mysql_prompt)


# In[69]:


few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
)


# In[67]:


new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)


# In[71]:


#we are executing this process to improve the efficiency and result accuracy of the llm by creating chains

new_chain("How many white color Levi's shirt I have?")


# In[ ]:




