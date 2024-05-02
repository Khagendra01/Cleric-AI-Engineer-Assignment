import requests

# Test the /submit_question_and_documents endpoint
payload = {
    'question': 'What are our product design decisions?',
    'documents': ['http://localhost:5000/test/call_log_fdadweq.txt']
}
response = requests.post('http://localhost:5000/submit_question_and_documents', data=payload)
data = response.json()
# Test the /get_question_and_facts endpoint
response = requests.get('http://localhost:5000/get_question_and_facts')
# Try to parse the response as JSON
try:
    response_json = response.json()
    print(response_json)
except ValueError as e:
    print(f"Error decoding JSON response: {e}")

    