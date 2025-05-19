from langchain_openai import OpenAI
from secret_key import openapi_key
from langchain.chains import SequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

# Set the API key
os.environ['OPENAI_API_KEY'] = openapi_key

# Initialize the LLM
llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name Generation
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template='I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.'
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')

    # Chain 2: Menu Items Suggestion
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest some menu items for {restaurant_name}.'
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='menu_items')

    # Combine chains sequentially
    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items'],
        verbose=True
    )

    response = chain.invoke({'cuisine': cuisine})
    return response

if __name__ == '__main__':
    print(generate_restaurant_name_and_items('Italian'))

