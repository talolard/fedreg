FROM python:3.8
RUN pip install beautifulsoup4 lxml jsonlines
ADD ./download_all_fedreg_data.sh .
ADD ./convert_to_json.py .
CMD ./download_all_fedreg_data.sh