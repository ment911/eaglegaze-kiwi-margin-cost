FROM python:3.6-slim-buster

ARG PATH_TO_SCRIPT
ENV PATH_TO_SCRIPT=$PATH_TO_SCRIPT
WORKDIR ${PATH_TO_SCRIPT}

ARG GIT_SSL_NO_VERIFY=$GIT_SSL_NO_VERIFY

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl && \
    apt-get install -y python3-dev  python3-pip && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get install -y libyaml-dev libpq-dev && \
    apt-get install -y postgresql-server-dev-all postgresql-client && \
    apt-get clean

ARG MAIN_DIR
ENV MAIN_DIR=$MAIN_DIR
COPY ./${MAIN_DIR} .
COPY requirements.txt .

RUN echo dependency repos succeed pip3 installed
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

ARG LOG_FILE
ENV LOG_FILE=$LOG_FILE
RUN echo $LOG_FILE

ARG LOG_PATH
ENV LOG_PATH=$LOG_PATH
RUN echo $LOG_PATH

ARG GENERAL_BRANCH
ENV GENERAL_BRANCH=$GENERAL_BRANCH
RUN echo $GENERAL_BRANCH

ARG GIT_TOKEN
ENV GIT_TOKEN=$GIT_TOKEN
RUN echo $GIT_TOKEN

ARG GIT_HOST
ENV GIT_HOST=$GIT_HOST
RUN echo $GIT_HOST

ARG GIT_URN
ENV GIT_URN=$GIT_URN
RUN echo $GIT_URN

RUN env

###### Copying repo dependencies
#RUN echo Copying eaglegaze-common...
#COPY uintei-common ./eaglegaze-common
#
RUN echo Git cloning eaglegaze-common...
RUN git clone -b ${GENERAL_BRANCH} https://oauth2:${GIT_TOKEN}@${GIT_HOST}${GIT_URN}/eaglegaze-common
RUN echo Installing eaglegaze-common...
RUN cd eaglegaze-common && python3 setup.py install && cd .. && rm -rf eaglegaze-common
#
#RUN echo pip3 installing eaglegaze-common...
#RUN pip3 install ${GENERAL_BRANCH} -i https://${GIT_TOKEN}@${GIT_HOST}${GIT_URN}/eaglegaze-common

#RUN pip3 list
#RUN pip3 -V

# start with log level 8 in foreground, output to stderr
ENTRYPOINT ["python3", "main.py"]
# CMD [ "python", "main.py" ]