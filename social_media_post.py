import logging
import sys
import os
from venv import logger

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

api_key= os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    logging.error("Please set your Open API key first, check .env file")
    sys.exit(1)

llm = ChatOpenAI(
    model="gpt-5.4-nano",
    temperature="0.5",
    verbose=True,
)

@tool
def write_formal_announcemnet(content: str) -> str:
    """
    Write a formal announcement for the given content.
    """
    content_prompt = PromptTemplate(
        input_variables=["content"],
        template="""You are a professional corporate communications writer.
Given the following content, write a structured announcement.

content: {content}
Write the announcement with:
- professional tone suitable for a company blog or press release
- Create a compelling headline
- Follow with 2 to 3 well-structured paragraphs
- The content should include:
- A clear introduction of the announcement
- Key features, benefits, or purpose
- The impact or value to customers or users
- Keep the language concise, clear, and professional
-Avoid overly promotional or casual language

Return ONLY the announcement, nothing else.""",
    )

    formatted_prompt = content_prompt.format(content=content)
    logger.info("Sending prompt to LLM - write_formal_announcement tool")

    response =llm.invoke(formatted_prompt)

    return response.content

@tool
def convert_to_social_media_post(content: str) ->str:
    """
    Convert the given content into a social media post.
    """
    social_media_prompt = PromptTemplate(
        input_variables=["content"],
        template="""You are an expert at creating engaging social media posts.

Rewrite the provided formal announcement into an engaging social media post tailored to the specified platform.
Formal announcement text
Target platform (LinkedIn, Instagram, or Twitter/X)
Rules:
- LinkedIn: Professional yet engaging, thought-leadership style
-Instagram: Casual, visually engaging, emoji-friendly
-Twitter/X: Concise, punchy, attention-grabbing
-Start with a strong hook opening line to capture attention
-Simplify key points from the announcement
-Highlight benefits or impact clearly
-Use relevant emojis appropriately (do not overuse)
-Add 2 to 5 relevant hashtags
-Ensure the content feels natural and platform-appropriate
-Keep it concise and engaging

content: {content}
Return ONLY the social media post, nothing else.""",
    )
    formatted_prompt = social_media_prompt.format(content=content)
    logger.info("Sending prompt to LLM - convert_to_social_media_post tool")

    response = llm.invoke(formatted_prompt)
    return response.content

tools = [write_formal_announcemnet, convert_to_social_media_post]
logger.info(f"Tools registered: {[t.name for t in tools]}")

SYSTEM_PROMPT = """You are a Social Media Post Generator assistant. Your job is to help users
create engaging social media posts. Your primary role is to adapt structured announcements into compelling, audience-focused content optimized for different social media platforms while preserving the original message and intent.

Core Responsibilities
-Analyze formal announcements and extract key information (purpose, value, highlights)
-Rewrite content in a way that is engaging, concise, and platform-appropriate
-Maintain brand professionalism while increasing readability and audience appeal
-Ensure clarity, accuracy, and consistency with the original content

When the user gives you content, follow these steps:
1. First, use the write_formal_announcement tool to create a structured announcement.
2. Then, use the convert_to_social_media_post tool to transform the announcement into a social media post.
3. Return the final social media post to the user.

Always use both tools in order: write formal announcement first, then convert to social media post."""

agent_graph = create_agent(
   tools = tools,
   system_prompt=SYSTEM_PROMPT,
   model=llm,
   debug=True,
)

def run_social_media_post(content: str) -> str:
   
    """
    Main function to run the social media post generator agent.
   
     Args:
        content: The formal announcement content to be converted into a social media post.
                     
    Returns:
        A engaging, concise, and platform-appropriate social media post.
    """
   
    logger.info("Running social media post generator agent")
    
    response = agent_graph.invoke(
       {"messages": [HumanMessage(content=content)]}
        
    
    )
   
    final_post = response["messages"][-1].content

    return final_post


if __name__ == "__main__":
   
   while True:
      
      content = input("Your content:").strip()

      if not content:
         print("Please enter some content to generate a social media post.\n")
         continue
      if content.lower() in ("quit", "exit","q","stop","close"):
            print("\nGoodbye, Have a great day!")
            break
      try:
         
         final_post = run_social_media_post(content)
         print("\n" + "=" * 60)
         print("YOUR SOCIAL MEDIA POST:")
         print("=" * 60)
         print(final_post)
         print("=" * 60 + "\n")

      except Exception as e:
            logger.error(f"Error generating social media post: {e}")
            print("An error occurred while generating the social media post. Please try again or check your api key\n")



