# ==============================================================================
# 🐳 Dockerfile
# ------------------------------------------------------------------------------
FROM practicalai/practicalai:latest

# ==============================================================================
# 🚢 Ports
# ------------------------------------------------------------------------------
EXPOSE 8888 6006 5000

# ==============================================================================
# 📖 Document classification
# ------------------------------------------------------------------------------
ARG DIR
COPY . $DIR/
WORKDIR $DIR/
RUN pip install -r requirements.txt && \
    python3.6 setup.py develop
ENV DIR ${DIR}
WORKDIR $DIR/document_classification
CMD /usr/bin/python3.6 /usr/local/bin/gunicorn --log-level ERROR --workers 4 --bind 0.0.0.0:5000 --access-logfile - --error-logfile - --reload wsgi
#ENTRYPOINT ["tail", "-f", "/dev/null"]