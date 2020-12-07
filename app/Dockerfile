FROM python:3.7 

WORKDIR /app 

COPY . /app 

# streamlit files to the .streamlit folder
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --upgrade protobuf


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install curl -y 

# gather and install coral-edgetpu package and install runtime #
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
            | tee /etc/apt/sources.list.d/coral-edgetpu.list 
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - 
RUN apt-get update
RUN apt-get install libedgetpu1-std -y
# # ?????? #
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_x86_64.whl

EXPOSE 8501

CMD streamlit run app.py 



