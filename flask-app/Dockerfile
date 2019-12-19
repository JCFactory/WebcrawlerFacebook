FROM continuumio/miniconda3

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN mkdir /app

EXPOSE 5000 8080 8081
WORKDIR /app

# Install fbcrawler requirements
COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -n fbcrawler -f environment.yml \
    && rm -rf /opt/conda/pkgs/*

# Copy all files after to avoid rebuild the conda env each time
COPY ./fapp/ /app/

# activate the fbcrawler environment
ENV PATH /opt/conda/envs/fbcrawler/bin:$PATH
RUN ls -latr
# Launch the API
#ENTRYPOINT ["python"]
CMD ["bash", "-c", "source activate fbcrawler && python ./app.py"]

#CMD [ "source activate flask_env && python ./reporting/app.py" ]

