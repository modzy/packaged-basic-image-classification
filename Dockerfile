# This Dockerfile is used to build a container image that contains everything
# necessary to run the model application.
#
# It must define a `CMD` that starts the application server on the port
# specified by the `PSC_MODEL_PORT` environment variable.
#
# Other than for simple CPU-only models, this file will likely need to be updated
# to support different model requirements.
#
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

# ensure UTF-8 locale; at the moment setting "LANG=C" on Linux can break Python 3
# http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# prevent Python from buffering stdin/stdout for immediate logging output
ENV PYTHONUNBUFFERED 1
# prevent Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# create the application working directory
WORKDIR /opt/app

# install python requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy application files into the container image
# NOTE: to avoid overly large container size, only copy the files actually needed by
#       explicitely specifying the needed files with the `COPY` command or adjusting
#       the `.dockerignore` file to ignore unneeded files
COPY . ./

# ensure the entrypoint is executable
# NOTE: this isn't necessary if the entrypoint.sh script is properly marked and checked
#       into git as executable
RUN chmod +x entrypoint.sh

# environment variable to specify model server port
ENV PSC_MODEL_PORT 80

# start the application server
# NOTE: the _exec_ syntax with square brackets should always be used when defining
#       the `CMD`
CMD ["./entrypoint.sh"]
