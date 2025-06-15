from re import template
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field


load_dotenv()

class AnaliseSentimento(BaseModel):
  comentario: str = Field(description="Comentário do usuário no e-commerce")
  sentimento: str = Field(description="Sentimento do comentário, positivo ou negativo")
  score: float = Field(description="Score do sentimento do comentário, de 0 a 10")
  justificativa_score: str = Field(description="Justificativa do score do sentimento do comentário")

parser = PydanticOutputParser(pydantic_object=AnaliseSentimento)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.7,
)

prompt = """
Você é um analista de sentimentos na seção de comentários de um e-commerce
Analuse o seguinte texto e extraia o sentimento:
{texto}

A resposta deve ser no seguinte formato:
{format_instructions}
"""

prompt_template = PromptTemplate(input_variables=["texto"], template=prompt)

chain = prompt_template | llm

# Peça para o usuário digitar um comentário
comentario = input("Digite um comentário: ")

resultado = chain.invoke({"texto": comentario, "format_instructions":parser.get_format_instructions()})

analise_sentimento = parser.parse(resultado.content)


print(analise_sentimento.comentario)
print(analise_sentimento.sentimento)
print(analise_sentimento.score)
print(analise_sentimento.justificativa_score)