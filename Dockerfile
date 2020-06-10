# Create essential base image
FROM python:3.7 as base
COPY requirements.txt /home/
WORKDIR /home/
RUN pip install -U pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -m spacy download en
ADD src/ /home/src/
ENV PYTHONPATH=/home/src
ENV PYTHONUNBUFFERED=0

# Label image with git commit url
ARG GIT_URL=unspecified
ARG VERSION=unspecified
LABEL org.label-schema.schema-version=1.0
LABEL org.label-schema.url=$GIT_URL
LABEL org.label-schema.version=$VERSION
ENV VERSION=$VERSION

# Run unittests
FROM base as tests
RUN pip install nose && \
    pip install pytest && \
    pip install coverage && \
    pip install hypothesis && \
    pip install testfixtures
COPY tests /home/tests
ARG cachebust=0
# ^ Change this to avoid using cached results. These are tests, so we may want to run them.
RUN nosetests --with-coverage --cover-package dsconcept

# Deployment ready image
FROM base as pipeline
COPY Makefile /home/
ENTRYPOINT ["make"]
