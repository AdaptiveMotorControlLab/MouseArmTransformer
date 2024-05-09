FROM pytorch/pytorch

# stuff for opencv to work
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /build
COPY . /build 
RUN pip install --no-cache-dir '.[dev]' && rm -rf /build

WORKDIR /app
RUN chmod a+rwx /app

