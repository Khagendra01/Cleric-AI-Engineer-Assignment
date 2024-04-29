import requests

# Test the /submit_question_and_documents endpoint
payload = {
    'question': 'What is pratidin?',
    'documents': ['https://pratidin.org/about']
}
response = requests.post('http://localhost:5000/submit_question_and_documents', data=payload)
data = response.json()

print(data['message'])

# Test the /get_question_and_facts endpoint
response = requests.get('http://localhost:5000/get_question_and_facts')
print(f"Status code: {response.status_code}")
print(f"Response content: {response.content}")

# Try to parse the response as JSON
try:
    response_json = response.json()
    print(response_json)
except ValueError as e:
    print(f"Error decoding JSON response: {e}")

    