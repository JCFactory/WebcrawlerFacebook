#!/bin/bash
PATH_TO_PROJECT="https://github.com/JCFactory/WebcrawlerFacebook/archive/master.zip"
ZIP_FILE="project.zip"
APP_FOLDER="WebcrawlerFacebook-master/flask-app/"
wget -O $ZIP_FILE $PATH_TO_PROJECT
unzip $ZIP_FILE
cd ./$APP_FOLDER
docker-compose up --build
