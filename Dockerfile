FROM python:3.12

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
COPY ./.env.production /app/.env

EXPOSE 80

CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]
