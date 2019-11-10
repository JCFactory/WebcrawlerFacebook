FROM continuumio/miniconda3

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN mkdir /app

EXPOSE 5000 8080 8081
WORKDIR /app

# Install myapp requirements
COPY ./app/environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -n myapp -f environment.yml \
    && rm -rf /opt/conda/pkgs/*

# Copy all files after to avoid rebuild the conda env each time
COPY ./app/ /app/

# activate the myapp environment
ENV PATH /opt/conda/envs/myapp/bin:$PATH
RUN ls -latr
# Launch the API
#ENTRYPOINT ["python"]
CMD ["bash", "-c", "source activate myapp && python ./reporting/app.py"]

#CMD [ "source activate flask_env && python ./reporting/app.py" ]

