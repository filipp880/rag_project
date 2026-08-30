# %%
import os
import json
import ollama
import re
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from retrieval import hybrid_search

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

file_path = os.path.join("data", "article.txt")
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_weather(city: str) -> dict:
    return {"city": city, "temp": 22, "condition": "sunny"}

def calculate(expression: str) -> float:
    try:
        return eval(expression)
    except (SyntaxError, ZeroDivisionError, TypeError, NameError):
        return 0.0

def dispatch(function_name: str, arguments: dict):
    if function_name == 'get_weather':
        return get_weather(**arguments)
    elif function_name == 'calculate':
        return calculate(**arguments)
    elif function_name == 'search_article':
        return search_article(**arguments)
    else: raise ValueError()

def search_article(query: str) -> str:
    return "\n\n".join(f"[источник: {src}] {doc}" for doc, src in hybrid_search(query))

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Получить погоду в городе',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'Название города'}
                },
        'required': ['city']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate',
            'description': "Вычислять математическое выражение",
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {"type": "string", 'description': "Математическое выражение"}
                },
        'required': ['expression'],
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_article',
            'description': 'База знаний с ответами на вопросы',
            'parameters':{
                'type': 'object',
                'properties': {
                    'query': {"type": "string", 'description': "Вопрос, ответ на который нужно найти в базе знаний"}
                },
        'required': ['query'],
            }
        }
    },
]

def agent(user_query: str, max_iterations: int = 3):
    messages = [
        {'role': 'system', 'content': 'Ты - полезный ассистент. Используй инструменты, только если они нужны. '
         'Сложные задачи решай последовательно, вызывая инструменты по шагам.'
         ' Отвечай на русском. '
         'Если отвечаешь по базе знаний, тогда обязательно укажи источник: "согласно <имя файла>"'},
         {'role': 'user', 'content': user_query}
    ]
    sources = set()
    for step in range(max_iterations):
        response = ollama.chat(model='llama3.2', messages=messages, tools=tools, options={'num_ctx': 8192})
        message=response['message']

        if not message.get('tool_calls'):
            print(f"Агент (финал, шаг{step+1}):{message['content']}")
            print(f"Источники: {', '.join(sorted(sources)) or 'не использовались'}")
            history = load_history()
            history.append({
                'timestamp': datetime.now().isoformat(),
                'query': user_query,
                'response': message['content'],
                'sources': sorted(sources)
            })
            save_history(history)
            return message['content']

        print(f"Шаг {step+1}: вызовы{[tc['function']['name']for tc in message['tool_calls']]}")
        messages.append(message)

        for tool_call in message['tool_calls']:
            name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            try:
                result = dispatch(name, args)
            except (ValueError, TypeError) as e:
                result = {'error': str(e)}
            if name == 'search_article':
                sources.update(re.findall(r'\[источник: ([^\]]+)\]', result))
            messages.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)})
    print(f"Источники: {', '.join(sorted(sources)) or 'не использовались'}")
    print("Превышено max_iterations - останавливаюсь.")
    return None

# agent("Что такое overlap?")
# agent("Что такое multi-head attention?")
# agent("Погода в Москве")
# agent("Какая температура в Москве, умноженная на 2?")
# agent("Какая температура в Москве, умноженная на 2?", max_iterationёёs=1)


