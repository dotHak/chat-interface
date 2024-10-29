# chat-interface

## Installation
Make sure to have python3 installed on your system. Then run the following commands:

- Install the required packages using the required:
```bash
pip install -r requirements.txt
```

## Usage
- Run the following command to start the chat interface:
```bash
python -m chainlit run app.py -h
```
or
```bash
python3 -m chainlit run app.py -h
```

- Run the following command to start the interface in development mode:
```bash
python -m chainlit run app.py -h -w
```

- Open the interface in your browser by visiting the following URL:
http://localhost:8000

## Environment Variables
- Create a `.env` file in the root directory of the project and add the following environment variables:
```bash
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
MOCK_HOSPITAL_SYSTEM_BASE_URL=https://mock-hospital-system.onrender.com
```
