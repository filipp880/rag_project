# %%
import json
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

with open("article.txt", 'r', encoding='utf-8') as f:
    text = f.read()

def chunk_by_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current = ''
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current.strip())
            current = ''
        current += p + '\n\n'
    if current.strip():
        chunks.append(current.strip())
    return chunks

def search(query: str, top_k: int = 3):
    query_vec = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k
    )
    print(f"\n=== Запрос: {query} ===")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"[dist={dist:.4f}] {doc[:150]}")
    return results['documents'][0]

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
    return "\n\n".join(search(query))

def ask_llm(query: str):
    print(f"\n{'='*50}")
    print(f"Пользователь: {query}")
    print(f"{'='*50}")  
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'Ты — полезный ассистент. Используй инструменты ТОЛЬКО если они действительно нужны для ответа на вопрос пользователя. Если вопрос не требует вызова функции — отвечай сам, без вызова инструментов.Отвечай на русском языке'
            },
                {'role': 'user', 
                 'content': f"Вопрос:{query}"}],
        tools=tools
    )

    message = response['message']

    if message.get('tool_calls'):
        tool_call = message['tool_calls'][0]
        func_name = tool_call['function']['name']
        func_args = tool_call['function']['arguments']

        print(f'LLM хочет вызвать: {func_name}({func_args})')

        result = dispatch(func_name, func_args)
        print(f'Результат функции: {result}')

        final_response = ollama.chat(
            model= 'llama3.2',
            messages=[
                {'role': 'user', 'content': query},
                message,
                {
                    'role': 'tool',
                    'content': json.dumps(result, ensure_ascii=False)
                }
            ],
            tools=tools
        )

        print(f"LLM (final): {final_response['message']['content']}")
    else:
        print(f"LLM (Без функции): {message['content']}")

def ask_rag(query: str):
    chunks = search(query)
    context = "\n\n".join(chunks)
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'Ты — умный ассистент. Ответь на вопрос пользователя, опираясь СТРОГО на предоставленный контекст.Если в контексте нет ответа на вопрос, так и скажи: В предоставленных данных нет ответа. Отвечай на русском'
            },
            {'role': 'user',
              'content': f"Контекст:\n{context}\n\nВопрос:{query}"}],
    )
    print(f"LLM ans:{response['message']['content']}")
    return chunks

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

client = chromadb.PersistentClient(path="./chromadb")
try:
    client.delete_collection('article')
except:
    pass

collection = client.create_collection(
    name='article',
    metadata={"hnsw:space": "cosine"}
)
chunks = chunk_by_paragraphs(text)
emb = model.encode(chunks)
collection.upsert(
    documents = chunks,
    embeddings = emb.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# while True:
#     user_data = input("Задай вопрос или напиши 'стоп'")
#     if user_data.lower() == 'стоп':
#         break
#     try:
#         ask_llm(user_data)
#     except (ValueError,TypeError) as e:
#         print(f"Ошибка: {e}")
#         result = {"error": "Функция не найдена"}


def agent(user_query: str, max_iterations: int = 3):
    messages = [
        {'role': 'system', 'content': 'Ты - полезный ассистент. Используй инструменты, только если они нужны. '
         'Сложные задачи решай последовательно, вызывая инструменты по шагам. Отвечай на русском. '},
         {'role': 'user', 'content': user_query}
    ]

    for step in range(max_iterations):
        response = ollama.chat(model='llama3.2', messages=messages, tools=tools)
        message=response['message']

        if not message.get('tool_calls'):
            print(f"Агент (финал, шаг{step+1}):{message['content']}")
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
            messages.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)})

    print("Превышено max_iterations - останавливаюсь.")
    return None

agent("Что такое overlap?")
agent("Какая температура в Москве, умноженная на 2?")
agent("Какая температура в Москве, умноженная на 2?", max_iterationёёs=1)


