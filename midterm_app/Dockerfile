FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]