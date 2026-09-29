from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = (
        "You are a NASA subject matter expert that is knowledgeable about NASA missions\n"
        "Apollo 11, Apollo 13, and Challenger.\n"
        "You are helping users with their questions about NASA.\n"
        "If the context is missing or insufficient, say so instead of guessing or using\n"
        "information that you have learned from other sources.\n"
        "Cite the sources you use.\n"
        "If the user asks a question that is not related to NASA tell the user to ask questions about NASA."
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    for history_item in conversation_history:
        messages.append({
            "role": history_item["role"],
            "content": history_item["content"]
        })

    if context:
        user_content = f"Context: {context}\n\nUser Question: {user_message}"
    else:
        user_content = (
            f"User Question: {user_message}\n\n"
            "No context retrieved from the archives.\n"
            "Tell the user that you cannot answer the question from the archives."
        )

    messages.append({
        "role": "user", "content": user_content})


    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    text_response = response.choices[0].message.content

    return text_response
