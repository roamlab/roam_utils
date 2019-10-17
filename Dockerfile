FROM python:3.6

COPY . /root/roam_utils

WORKDIR /root/roam_utils

RUN pip install .

CMD /bin/bash
